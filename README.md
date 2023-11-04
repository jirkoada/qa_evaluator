# A tool for evaluation of Langchain QA model performance

Based on https://github.com/langchain-ai/auto-evaluator

## Usage
Clone this repo:

    git clone https://gitlab.com/alquist/ciirc-projects/porsche/qa_eval.git

Install required packages:

    cd qa_evaluator
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt

Generate datasets:

    python3 cli_create_dataset.py --help
    python3 cli_create_dataset.py docs/fee_brochure.pdf docs/fee_rules.pdf [docs/file3.pdf ...] -n 10 -o datasets/example.csv

Optionally, adjust the custom task definition for the generation prompts inside text_utils.py to improve the relevance of generated questions.

Adjust evaluation settings in eval_config.yml. Create more versions of the file if needed.

Run evaluation:

    python3 cli_evaluate.py --help
    python3 cli_evaluate.py [-c path_to_custom_config.yml]

Use Interactive_eval.ipynb or Standalone_interactive.ipynb for better control over code execution.
