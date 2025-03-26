from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import ClassVar, List, Optional
from simulator import get_simulation_summary

import torch

# if torch.cuda.is_available():
#     print(f"Available CUDA devices: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("No CUDA devices available")

class qwen2(LLM):
    max_new_tokens: int = 1920
    temperature: float  = 0.9
    top_p: float = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    device: ClassVar[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,max_new_tokens = 1920,
                      temperature = 0.9,
                      top_p = 0.8):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _llm_type(self) -> str:
        return "qwen2"
    
    # load pretrained model
    def load_model(self, model_name_or_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model =AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )

    # handles conversation
    def chat_stream(self, model, tokenizer, query: str, history: list):
        special_responses = {
            "Who are you?": "I am an Intelligent Emergency Department chatbot designed to provide assistance to patients, including basic medical knowledge, and patient condition analysis.",
            "What is my current status?": "I dont know yet"
        }

        if query in special_responses:
            response = special_responses[query]
            messages = history + [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
            return response, messages

        with torch.no_grad():
            # add system role instructions for the model's behavior
            messages = [{
                'role': 'system', 
                'content': (
                    '###role\n'
                    'You are a general practitioner with a wealth of general medical knowledge, and you are very good at answering health questions accurately in an easy-to-understand way.\n'
                    '###goal\n'
                    'You provide accurate, professional, and easy-to-understand answers through your medical knowledge.\n'
                    'All your answers should have a medical evidence base to ensure accuracy and reliability.'
                )
            }]

            for item in history:
                if item['role'] == 'user':
                    if item.get('content'):
                        messages.append({'role': 'user', 'content': item['content']})
                if item['role'] == 'assistant':
                    if item.get('content'):
                        messages.append({'role': 'assistant', 'content': item['content']})
                                 
            # reconstructs conversation history
            messages.append({'role': 'user', 'content': query})
            
            # formats messages into a single string for model input.
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
                                    
            # tokenizes and sends inputs to device
            model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
            
            # generates response using model parameters.
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                temperature=self.temperature
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # generates response using model parameters.
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            messages.append({'role': 'assistant', 'content': response})

        return response ,messages

    # chat logic
    def _call(self, prompt: str ,stop: Optional[List[str]] = ["<|user|>"]):
        response, self.history = self.chat_stream(self.model, self.tokenizer, prompt, self.history)
        return response
    
    def query_only(self, query):
        if self.history[-2]['role'] == 'user':
            self.history[-2]['content'] = query
            
    def get_history(self) -> List:
        return self.history
    
    def delete_history(self):
        del self.history 
        self.history = []