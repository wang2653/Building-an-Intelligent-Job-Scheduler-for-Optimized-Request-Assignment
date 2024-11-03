import random
import gradio as gr

def respond(message, chat_history):
        #No LLM here, just respond with a random pre-made message
        bot_message = random.choice(["This project is Building an Intelligent Job Scheduler for Optimized Request Assignment", 
                                     "Emergency departments are highly complex and often face serious challenges of overcrowding and long patient waiting times. A number of factors contribute to the complexity of patient flow, including patients with varying levels of severity and the availability of medical resources. The unpredictability of patient arrivals and changing resource constraints further exacerbate this complexity, making it difficult for decision makers to effectively manage patient scheduling. Scheduling errors can lead to longer patient waiting times, overcrowding, and potential deterioration in patient health.", 
                                     "Hi, this is a chatbot"]) 
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
gr.close_all()
demo.launch(share=True)