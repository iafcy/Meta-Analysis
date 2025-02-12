from openai import OpenAI
import anthropic
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import pipeline
from dotenv import load_dotenv
from os import getenv
load_dotenv()

class Bot():
    def __init__(self, model_name, temperature=0.0, max_tokens=2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens=max_tokens

    def query(self, messages):
        raise NotImplementedError
    
class GPT(Bot):
    def __init__(self, model_name='gpt-4o', temperature=0.0, max_tokens=2048):
        super().__init__(model_name, temperature, max_tokens)

        self.client = OpenAI(api_key=getenv('OPENAI_API_KEY'))

    def query(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        response = response.choices[0].message.content

        return response

class Claude(Bot):
    def __init__(self, model_name='claude-3-5-sonnet-20241022', temperature=0.0, max_tokens=2048):
        super().__init__(model_name, temperature, max_tokens)

        self.client = anthropic.Anthropic()

    def query(self, messages):
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        response = response.content.text

        return response
    
class HuggingFace(Bot):
    def __init__(self, model_name='Qwen/Qwen2.5-72B-Instruct', temperature=0.0, max_tokens=2048):
        super().__init__(model_name, temperature, max_tokens)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype="auto",
        )
        self.pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer
        )

    def query(self, messages):
        response = self.pipe(
            messages,
            do_sample=self.temperature > 0,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            return_full_text=False
        )[0]['generated_text']

        return response
