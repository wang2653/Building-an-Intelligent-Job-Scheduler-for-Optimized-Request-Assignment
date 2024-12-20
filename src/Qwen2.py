from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import ClassVar, List, Optional
import torch

# if torch.cuda.is_available():
#     print(f"Available CUDA devices: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("No CUDA devices available")

class Qwen2(LLM):
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

    @property
    def _llm_type(self) -> str:
        return "Qwen2"
    
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
    def chat_stream(self, model, tokenizer, query:str, history:list):
        with torch.no_grad(): 
            messages = [
                {
                    "role": "system",
                    "content": "You are an emergency room nurse, providing professional insights into managing emergency departments."
                },
                {
                    "role": "user",
                    "content": "How can scheduling in emergency departments be improved?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Emergency departments are highly complex and often face serious challenges of overcrowding and long patient waiting times. "
                        "A number of factors contribute to the complexity of patient flow, including patients with varying levels of severity and the availability of medical resources [1]. "
                        "The unpredictability of patient arrivals and changing resource constraints further exacerbate this complexity, making it difficult for decision makers to effectively manage patient scheduling. "
                        "Scheduling errors can lead to longer patient waiting times, overcrowding, and potential deterioration in patient health.\n\n"
                        "For several decades, substantial efforts have been made to resolve the crowding caused by the complexities in emergency departments. "
                        "Traditionally, this problem is solved by rule-based algorithms (e.g., assigning patients to the next available doctor). "
                        "However, such algorithms often result in suboptimal performance and long delays due to an inability to adapt to variable patient demand. "
                        "Recently, advancement in deep reinforcement learning has enabled new algorithms to leverage trends in past data to forecast patient demand and schedule accordingly to significantly reduce wait time. "
                        "Currently, the state-of-the-art scheduler for teleconsultation utilizes Deep Q Network (DQN) and achieved 9%–41% reduction in average waiting time.\n\n"
                        "Our scheduler will implement other deep reinforcement algorithms such as Deep Deterministic Policy Gradient (DDPG) and Proximal Policy Optimization (PPO) to further increase the performance of the scheduler. "
                        "Encasing the novel algorithm, our scheduler will include a built-in chatbot for collecting request information, a network prediction component to forecast incoming request loads, and an efficient scheduling system to minimize patient wait times. "
                        "In addition, our capstone product will include a user-friendly graphical user interface (GUI) to ensure accessibility for all users, thereby enhancing the overall efficiency and effectiveness of patient service interactions."
                    )
                }
            ]
            for item in history:
                if item['role'] == 'user':
                    if item.get('content'):
                        messages.append({'role': 'user', 'content': item['content']})
                if item['role'] == 'assistant':
                    if item.get('content'):
                        messages.append({'role': 'assistant', 'content': item['content']})         
            messages.append({'role': 'user', 'content': query})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
                        
            model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                temperature=self.temperature
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  
            messages.append({'role': 'assistant', 'content': response})

        return response ,messages

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