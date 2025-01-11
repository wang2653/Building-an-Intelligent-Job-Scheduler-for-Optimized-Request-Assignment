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

    @property
    def _llm_type(self) -> str:
        return "qwen2"
    
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
    def chat_stream(self, model, tokenizer, query: str, history: list):
        with torch.no_grad():
            # retrieve simulation data
            simulation_summary = get_simulation_summary()
            patients = simulation_summary["patients"]
            resources = simulation_summary["resources"]

            # Sort patients by acuity
            patients.sort(key=lambda p: p["acuity_level"], reverse=True)

            # Generate a concise summary
            top_patients = [
                f"Patient #{p['id']} (Acuity: {p['acuity_level']}, Remaining Time: {p['remaining_treatment_time']})"
                for p in patients[:3]
            ]
            top_resources = [
                f"{r['type']} (Available: {r['available_slots']})"
                for r in resources
            ]

            system_content = f"""
            Current Status:
            Top Patients: {', '.join(top_patients)}
            Resources: {', '.join(top_resources)}
            """

            # Build conversation
            messages = [
                {"role": "system", "content": system_content},
                *[
                    {"role": msg['role'], "content": msg['content']}
                    for msg in history
                ],
                {"role": "user", "content": query}
            ]

            # Generate model output
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
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response, messages

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