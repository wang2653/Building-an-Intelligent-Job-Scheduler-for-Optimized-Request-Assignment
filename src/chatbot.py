import json
import gradio as gr
import openai
import asyncio

# openai.api_key = "APIHERE"

def add_message(history, message):
    if message:
        history.append((message, None))
    return history, ""

async def bot(history):
    messages = []
    for human, assistant in history[:-1]:
        if human:
            messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": history[-1][0]})

    er_data = load_er_data()

    system_prompt = f"""
    You are an AI assistant providing up-to-date information about emergency room wait times, resource availability, and other related updates.
    Use the following data for accurate responses: {er_data}
    Please answer the patient's questions based on the data provided.
    """

    messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )

        assistant_reply = response['choices'][0]['message']['content']
    except Exception as e:
        assistant_reply = "I'm sorry, but I'm unable to process your request at the moment."
        print(f"Error: {e}")

    history[-1][1] = assistant_reply
    return history

def load_er_data():
    try:
        with open('er_data.json', 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading ER data: {e}")
        return {
            "wait_times": "Data not available.",
            "resource_availability": "Data not available.",
            "updates": "Data not available."
        }

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()

    chat_input = gr.Textbox(
        placeholder="Enter your message here...",
        show_label=False,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(
        bot, inputs=[chatbot], outputs=[chatbot]
    )

demo.queue()
demo.launch(share=True)
