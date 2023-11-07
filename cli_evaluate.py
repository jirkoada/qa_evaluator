import os
import time
import random
import itertools
import pandas as pd
import argparse
import yaml
from types import SimpleNamespace    
from langchain.llms import MosaicML
from langchain.llms import Anthropic
from langchain.llms import Replicate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.evaluation.qa import QAEvalChain
from langchain.document_loaders import UnstructuredPDFLoader, PyMuPDFLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI, QA_CHAIN_PROMPT, QA_CHAIN_PROMPT_LLAMA
from text_utils import GENERATION_PROMPT_SELECTOR
from integrations import FTembeddings

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_file", default="eval_config.yml", help="Path to custom config file")


def create_llama_pipeline(model_id='meta-llama/Llama-2-7b-chat-hf', temperature=0.1):
    from torch import cuda, bfloat16
    from transformers import StoppingCriteria, StoppingCriteriaList
    import torch
    import transformers
    from langchain.llms import HuggingFacePipeline
    
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, you need an access token
    hf_auth = os.getenv('HF_TOKEN')
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )

    # enable evaluation mode to allow model inference
    model.eval()

    print(f"Model loaded on {device}")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    
    stop_list = ['\nHuman:', '\n```\n']
    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    # define custom stopping criteria object
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=1024,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    
    return HuggingFacePipeline(pipeline=generate_text)


def make_llm(model):
    """
    Make LLM
    @param model: LLM to use
    @return: LLM
    """

    if model in ("gpt-3.5-turbo", "gpt-4"):
        # TODO: Try langchain.llms.OpenAI instead
        llm = ChatOpenAI(model_name=model, temperature=0)
    elif model == "anthropic":
        llm = Anthropic(temperature=0)
    elif model == "Anthropic-100k":
        llm = Anthropic(model="claude-v1-100k",temperature=0)
    elif model == "vicuna-13b":
        llm = Replicate(model="replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e",
                input={"temperature": 0.75, "max_length": 3000, "top_p":0.25})
    elif model == "mosaic":
        llm = MosaicML(inject_instruction_format=True,model_kwargs={'do_sample': False, 'max_length': 3000})
    elif model == "llama2":
        #llm = LlamaCpp(model_path="/lscratch/poludmik/llama2/from_hf_2-7b/ggml-model-q4_0.bin", n_ctx=4096)
        llm = create_llama_pipeline()
    else:
        raise Exception("Invalid model choice")
    return llm


def make_embeddings(embeddings):
    if embeddings == "OpenAI":
        embd = OpenAIEmbeddings()
    # Note: Still WIP (can't be selected by user yet)
    elif embeddings == "LlamaCppEmbeddings":
        embd = LlamaCppEmbeddings(model_path="/lscratch/poludmik/llama2/from_hf_2-7b/ggml-model-q4_0.bin", n_ctx=2048)
    elif embeddings == "FastText":
        embd = FTembeddings("models/cc.en.300.bin")
    elif embeddings == "gte-large":
        model_name = "thenlper/gte-large"
        embd = HuggingFaceEmbeddings(model_name=model_name)
    else:
        model_name = embeddings
        embd = HuggingFaceEmbeddings(model_name=model_name)
    return embd


def make_chain(llm, retriever, model):

    """
    Make retrieval chain
    @param llm: model
    @param retriever: retriever
    @return: QA chain
    """

    # Select prompt 
    if model == "vicuna-13b":
        # Note: Better answer quality using default prompt 
        # chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT_LLAMA}
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
    else: 
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}

    # Select model 
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           chain_type_kwargs=chain_type_kwargs,
                                           input_key="question")
    return qa_chain


def grade_model_answer(predicted_dataset, predictions, grade_answer_prompt):
    """
    Grades the answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @return: A list of scores for the distilled answers.
    """

    #print("`Grading model answer ...`")
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Note: GPT-4 grader is advised by OAI 
    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def grade_model_retrieval(gt_dataset, predictions, grade_docs_prompt):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading.
    @return: list of scores for the retrieved documents.
    """

    #print("`Grading relevance of retrieved docs ...`")
    if grade_docs_prompt == "Fast":
        prompt = GRADE_DOCS_PROMPT_FAST
    else:
        prompt = GRADE_DOCS_PROMPT

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(gt_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def run_eval(chain, retriever, eval_qa_pair, grade_prompt, text):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chain: Model chain used for answering questions
    @param retriever:  Document retriever used for retrieving relevant documents
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param grade_prompt: String prompt used for grading model's performance
    @param text: full document text
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_grade: A dictionary containing scores for the model's document retrieval.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """

    #print("`Running eval ...`")
    predictions = []
    retrieved_docs = []
    gt_dataset = []
    latency = []

    # Get answer and log latency
    start_time = time.time()
    predictions.append(chain(eval_qa_pair))
    gt_dataset.append(eval_qa_pair)
    end_time = time.time()
    elapsed_time = end_time - start_time
    latency.append(elapsed_time)

    # Extract text from retrieved docs
    retrieved_doc_text = ""
    
    docs = retriever.get_relevant_documents(eval_qa_pair["question"])
    for i, doc in enumerate(docs):
        retrieved_doc_text += "Doc %s: " % str(i+1) + \
            doc.page_content + " "

    # Log
    retrieved = {"question": eval_qa_pair["question"],
                 "answer": eval_qa_pair["answer"], "result": retrieved_doc_text}
    retrieved_docs.append(retrieved)

    # Grade
    graded_answers = grade_model_answer(
        gt_dataset, predictions, grade_prompt)
    graded_retrieval = grade_model_retrieval(
        gt_dataset, retrieved_docs, grade_prompt)
    return graded_answers, graded_retrieval, latency, predictions


def run_evaluator(retriever, test_dataset, num_eval_questions, model_version, grade_prompt):
    #print("Making LLM")
    llm = make_llm(model_version)

    #print("Making chain")
    qa_chain = make_chain(llm, retriever, model_version)
    
    results = pd.DataFrame()
    #print()
    num_q = min(num_eval_questions, len(test_dataset))
    correct = 0
    acc = float("nan")
    print("Evaluating...")
    for i in range(num_q):
        print(f"\rQuestion {i+1} / {num_q}, Correct: {correct} / {i}, ACC: {acc}", end="")
        eval_pair = test_dataset[i]
        
        # Run eval
        graded_answers, graded_retrieval, latency, predictions = run_eval(
            qa_chain, retriever, eval_pair, grade_prompt, "")

        # Assemble output
        d = pd.DataFrame(predictions)
        d['answerScore'] = 1 if "Incorrect" not in graded_answers[0]["text"] else 0
        d['answerComment'] = graded_answers[0]["text"]
        d['retrievalScore'] = 1 if "Incorrect" not in graded_retrieval[0]["text"] else 0
        d['retrievalComment'] = graded_retrieval[0]["text"]
        d['latency'] = latency
        
        correct += d['answerScore'][0]
        acc = correct / (i+1)

        # Convert dataframe to dict
        d_dict = d.to_dict('records')
        results = pd.concat([results, d], ignore_index=True)
    return results


def create_dataset(paths, num_questions, context_length):
    text = ""
    for path in paths:
        loader = PyMuPDFLoader(path)
        local_pages = loader.load_and_split()
        for page in local_pages:
            text += page.page_content
    
    pairs = []
    section_length = len(text) // num_questions
    print(f"Chars per section: {section_length}")
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    prompt = GENERATION_PROMPT_SELECTOR.get_prompt(llm)
    chain = QAGenerationChain.from_llm(llm, prompt)
    
    for i in range(num_questions):
        print(f"\rGenerating question {i+1} out of {num_questions}", end="")
        start_index = random.randint(0, section_length//2) + i * section_length
        sub_sequence = text[start_index : start_index + context_length]
        eval_set = []
        # Catch any QA generation errors and re-try until QA pair is generated
        awaiting_answer = True
        while awaiting_answer:
            try:
                qa_pair = chain.run(sub_sequence)
                eval_set.append(qa_pair)
                awaiting_answer = False
            except Exception as e:
                print("Exception: ", e)
                start_index = random.randint(0, section_length//1.4) + i * section_length
                sub_sequence = text[start_index : start_index + context_length]
        eval_pair = list(itertools.chain.from_iterable(eval_set))
        pairs.append(eval_pair[0])
    print()
    return pairs


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Load config
    with open(args.config_file, "r") as cf:
        try:
            config = yaml.safe_load(cf)
            cfg = SimpleNamespace(**config)
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception("Config file could not be parsed")

    # Load dataset
    test_dataset = pd.read_csv(cfg.dataset_file)[["question", "answer"]].to_dict('records')
    print(f"Loaded {len(test_dataset)} questions")

    # Create or load vector store
    embd = make_embeddings(cfg.embeddings)
    if cfg.create_new_vs:
        splits = []
        for path in cfg.file_paths:
            loader = PyMuPDFLoader(path) # Fast, Good for metadata
            if cfg.split_method == "RecursiveTextSplitter":
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_chars,
                                                            chunk_overlap=cfg.overlap)
            elif cfg.split_method == "CharacterTextSplitter":
                text_splitter = CharacterTextSplitter(separator=" ",
                                                    chunk_size=cfg.chunk_chars,
                                                    chunk_overlap=cfg.overlap)
            else:
                raise Exception("Invalid text splitter")
        
            local_pages = loader.load_and_split(text_splitter)
            splits.extend(local_pages)
        vs = FAISS.from_documents(splits, embd)
        print("New vector store created")
        if cfg.save_vs_path is not None:
            vs.save_local(cfg.save_vs_path)
            print(f"Vector store saved to {cfg.save_vs_path}")
    
    elif cfg.vector_store is not None:
        vs = FAISS.load_local(cfg.vector_store, embd)
        print(f"Loaded {cfg.vector_store}")
    
    else:
        raise Exception("Invalid vector store settings")
    
    # Run evaluation
    retriever = vs.as_retriever(k=cfg.num_neighbors)
    results = run_evaluator(retriever, test_dataset, cfg.num_eval_questions, cfg.model_version, cfg.grade_prompt)
    
    score = results["answerScore"].mean()
    score_str = f"Total score: {score} ({len(results['answerScore'][results['answerScore'] == 1])} / {len(results['answerScore'])})"
    print("\n")
    print(score_str)
    lat = results[results['latency'] < results['latency'].quantile(0.99)]['latency'].mean()
    print(f"Typical latency: {lat}")

    # Save results
    if cfg.incorrect_first:
        results.index.name = "idx"
        results = results.sort_values(by = ["answerScore", "idx"])

    results[score_str] = ""
    results[f"Parameters: {str(config)}"] = ""
    results.loc[results.index[0], score_str] = f"Typical latency: {lat}"
    results.to_csv(cfg.result_path)

