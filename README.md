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
git clone https://github.com/iafcy/MetaAnalysis
cd MetaAnalysis
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
cp .env_sample .env
```

2. Fill in your API keys
```
OPENAI_API_KEY=<your_openai_api_key>
ANTHROPIC_API_KEY=<your_anthropic_api_key>
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
Then, you can execute the `evaluate_<task_name>` function to calculate and view the performance.

##### Base to Key
```python
from experiment import *
dir = 'test'

# predictions saved in ./test/b2k.json
predict_b2k(model, dir)
evaluate_b2k(dir)
```

##### Quality Assessment
```python
from experiment import *
dir = 'test'

# predictions saved in ./test/qa.json
predict_qa(model, dir)
evaluate_qa(dir)
```

##### Characteristics Classification
```python
from experiment import *
dir = 'test'

# predictions saved in ./test/cc.json
predict_cc(model, dir)
evaluate_cc(dir)
```

##### GRADE
```python
from experiment import *
dir = 'test'

# predictions saved in ./test/grade.json
predict_grade(model, dir)
evaluate_grade(dir)
```

#### Test your LLM
If you want to test a model which is not from OpenAI, Anthropic or Huggingface, you need to modify `model.py`.
1. Create a class in `model.py` that inherit from the `Bot` class
2. Overwrite the `query` method, where the input is the prompt message, and the output is the response (string) of the model
3. Instantiate the model, pass it to `predict_<task_name>` and `evaluate_<task_name>` to run the experiment.