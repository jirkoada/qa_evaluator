import random
import itertools
import pandas as pd
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.document_loaders import PyMuPDFLoader
from text_utils import GENERATION_PROMPT_SELECTOR


parser = argparse.ArgumentParser()
parser.add_argument("source_pdfs", nargs="*", default="[]", help="Paths to source pdf documents")
parser.add_argument("-o", "--output_path", default="new_dataset.csv", help="Path to target .csv file")
parser.add_argument("-n", "--num_questions", default=10, type=int, help="Number of questions to generate")
parser.add_argument("-c", "--context_length", default=3000, type=int, help="Length of context for question generation")


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
    dataset = create_dataset(args.source_pdfs, args.num_questions, args.context_length)
    df = pd.DataFrame(dataset)
    df.to_csv(args.output_path)
    print(f"Dataset stored to {args.output_path}")
    