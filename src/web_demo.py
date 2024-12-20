import gradio as gr
from Qwen2 import Qwen2

import os
model_path = "D:/qwen"
print("Model path exists:", os.path.exists(model_path))

if os.path.exists(model_path):
    print("Files in directory:", os.listdir(model_path))
else:
    print("Specified path does not exist.")

llm = Qwen2()
llm.load_model(model_path)

def predict(query,history):
    response = llm(query)
    llm.query_only(query)

    return response

gr.ChatInterface(predict).launch(share=True)