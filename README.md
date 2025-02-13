<div align= "center">
    <h1>Meta-Analysis</h1>
</div>

<p align="center">
   <a href="https://huggingface.co/collections/iafcy/meta-analysis-67acaf78f9de76315b8fe199" target="_blank">🤗 Dataset</a>
</p>

## 📖 Overview

This repositry contains:
- Code implementation for running experiments for LLM on Meta-Analysis
- Test dataset in `./data`. The full dataset, including training, testing and validation is available on [Huggingface 🤗](https://huggingface.co/collections/iafcy/meta-analysis-67acaf78f9de76315b8fe199).

## 💻️ Code

### Installation
1. Clone the repositry
```bash
git clone https://github.com/iafcy/Meta-Analysis
cd Meta-Analysis
```
2. Create virtual environment and download libraries
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

#### Project structure
```
.
├── data                  # Test dataset
│   ├── b2k_test.json     # Base to Key
│   ├── qa_test.json      # Quality Assessment
│   ├── cc_test.json      # Characteristics Classification
│   └── grade_test.json   # GRADE
├── evaluation            # For calculating performance
├── prediction            # For generating predictions
├── main.py               # Example
├── model.py              # Classes for LLM
└── prompt.py             # Prompt templates for each task
```

#### Set up models
1. Copy .env_sample
```bash
cp .env.sample .env
```

2. Fill in your API keys
```
OPENAI_API_KEY=<your_openai_api_key>
ANTHROPIC_API_KEY=<your_anthropic_api_key>
LLAMA_CLOUD_API_KEY=<your_llam_cloud_api_key> # For LlamaParse
```

#### Generate predictions
Example code is is available in `main.py`.

First, initialize the model by specifiying the model name and hyperparameters. You can use models from OpenAI, Anthropic and Huggingface.
```python
from model import GPT, Claude, HuggingFace

model = GPT(model_name='gpt-4o')
model = Claude(model_name='claude-3-5-sonnet-20241022')
model = HuggingFace(model_name='Qwen/Qwen2.5-72B-Instruct')
```

To generate predictions, you need to  specific the initialized model and the directory where the prediction results will be saved.
Executing the `predict_<task_name>` function will generate a JSON file with the test data and the predictions from the model.
Then, you can execute the `evaluate_<task_name>` function to calculate and print the performance.

##### Base to Key
```python
from experiment import *
dir = 'test'

predict_b2k(model, dir) # predictions saved in ./test/b2k.json
evaluate_b2k(dir) # performance will be printed and saved to ./test/b2k.json
```

##### Quality Assessment
```python
from experiment import *
dir = 'test'

predict_qa(model, dir) # predictions saved in ./test/qa.json
evaluate_qa(dir) # performance will be printed and saved to ./test/qa.json
```

##### Characteristics Classification
```python
from experiment import *
dir = 'test'

predict_cc(model, dir) # predictions saved in ./test/cc.json
evaluate_cc(dir) # performance will be printed and saved to ./test/cc.json
```

##### GRADE
```python
from experiment import *
dir = 'test'

predict_grade(model, dir) # predictions saved in ./test/grade.json
evaluate_grade(dir) # performance will be printed and saved to ./test/grade.json
```

#### Test your LLM
If you want to test a model which is not from OpenAI, Anthropic or Huggingface, you need to modify `model.py`.
1. Create a class in `model.py` that inherit from the `Bot` class
2. Overwrite the `query` method, where the input is the prompt message, and the output is the response (string) of the model
3. Overwrite the `get_llamaindex_llm` and `get_llamaindex_embedding` methods, which are used to initialize model and embedding model for llamaindex.
4. Instantiate the model, pass it to `predict_<task_name>` and `evaluate_<task_name>` to run the experiment.
