import gradio as gr

# Creating the main interface using Gradio Blocks
with gr.Blocks(css="../assets/styles.css") as interface:
    with gr.Row():
        gr.Markdown("# Intelligent Scheduler for Emergence Department", elem_id="page_title")
        gr.Button("Home", elem_id="home_btn", size="lg")
        gr.Button("About", elem_id="about_btn", size="lg")
        gr.Button("Contact", elem_id="contact_btn", size="lg")
        gr.Button("Login", elem_id="login_btn", size="lg")
    with gr.Row():
        gr.Markdown("# Explore Better Decision Making", elem_id="main_text")
        gr.Image(value="../static/emergency1.jpg", elem_id="main_image")

interface.launch()