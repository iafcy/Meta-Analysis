from experiment import *
from model import GPT, Claude, HuggingFace

dir = 'test'
model = GPT(model_name='gpt-4o')
# model = Claude(model_name='claude-3-5-sonnet-20241022')
# model = HuggingFace(model_name='Qwen/Qwen2.5-72B-Instruct')

predict_b2k(model, dir)
evaluate_b2k(dir)
# predict_cc(model, dir)
# evaluate_cc(dir)
# predict_qa(model, dir)
# evaluate_qa(dir)
# predict_grade(model, dir)
# evaluate_grade(dir)