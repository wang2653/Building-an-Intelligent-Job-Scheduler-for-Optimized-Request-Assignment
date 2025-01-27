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
                    'All your answers should have a medical evidence base to ensure accuracy and reliability.\n'
                    '###Office Syndrome\n'
                    'Office Syndrome refers to a group of symptoms commonly experienced by office workers due to prolonged sitting, poor posture, and repetitive motions. These symptoms often include neck, shoulder, and back pain, as well as headaches, eye strain, and wrist discomfort. Preventative measures such as ergonomic adjustments, regular breaks, stretching exercises, and maintaining good posture are essential for managing and preventing Office Syndrome.\n'
                    '###Health related knowledge\n'
                    '1. **How to Prevent and Cure Common Cold:**\n'
                    '   - **Prevention:** Wash hands regularly with soap and water, avoid close contact with sick individuals, and strengthen your immune system with a balanced diet, sufficient sleep, and regular exercise.\n'
                    '   - **Cure:** Rest adequately, stay hydrated, use over-the-counter medications to alleviate symptoms, and consult a healthcare professional if symptoms persist or worsen.\n\n'
                    '2. **Tips for Headaches:**\n'
                    '   - **Prevention:** Stay hydrated, maintain a consistent sleep schedule, manage stress through relaxation techniques, and avoid known triggers like certain foods or bright lights.\n'
                    '   - **Relief:** Use a cold or warm compress, practice deep breathing, massage your temples, and consider over-the-counter pain relievers if necessary. Seek medical advice for severe or recurring headaches.\n\n'
                    '3. **Tips for Stomachache:**\n'
                    '   - **Prevention:** Avoid overeating, eat a balanced diet rich in fiber, drink plenty of water, and practice proper food hygiene to prevent infections.\n'
                    '   - **Relief:** Drink warm water or herbal tea (e.g., ginger or peppermint), rest in a comfortable position, apply a warm compress to your abdomen, and avoid heavy or greasy foods. If pain persists or is severe, consult a healthcare professional.\n'
                    '###Health Tips\n'
                    '1. **Eat a Balanced Diet:** Incorporate a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats into your meals to ensure you get essential nutrients.\n'
                    '2. **Stay Hydrated:** Drink plenty of water daily to maintain optimal body functions and avoid dehydration.\n'
                    '3. **Exercise Regularly:** Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous exercise weekly to maintain physical fitness.\n'
                    '4. **Get Enough Sleep:** Adults should aim for 7-9 hours of quality sleep each night to promote overall health and mental well-being.\n'
                    '5. **Practice Good Hygiene:** Wash your hands frequently, brush your teeth twice daily, and follow proper hygiene practices to prevent infections and maintain oral health.\n'
                    '6. **Manage Stress:** Use relaxation techniques like deep breathing, meditation, or yoga to manage stress and improve mental health.\n'
                    '7. **Limit Processed Foods and Sugars:** Reduce your intake of sugary drinks, snacks, and processed foods to lower the risk of chronic diseases.\n'
                    '8. **Avoid Smoking and Excess Alcohol:** Quit smoking and limit alcohol consumption to protect your lungs, liver, and overall health.\n'
                    '9. **Stay Up-to-Date on Vaccinations:** Regularly consult your doctor to ensure you are up-to-date on recommended immunizations for your age group and lifestyle.\n'
                    '10. **Schedule Regular Checkups:** Visit your healthcare provider for routine checkups to monitor your health, detect issues early, and receive personalized advice.'
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