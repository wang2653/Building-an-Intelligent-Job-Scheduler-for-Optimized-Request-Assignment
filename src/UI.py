import gradio as gr
from PIL import Image

# Read the CSS file
with open("../assets/styles.css", "r") as css_file:
    styles_css = f"<style>{css_file.read()}</style>"

# Load the background image
"""background_image = Image.open("../static/background1.jpg")

def edit_image(image):
    return image"""

with gr.Blocks() as interface:
    gr.HTML(styles_css)  # Adding custom styles here
    with gr.Row(equal_height=True, elem_id="header_row"):
        gr.Markdown("# Intelligent Scheduler for Emergence Department", elem_id="page_title")
        gr.Button("Home", elem_id="home_btn", size="lg")
        gr.Button("About", elem_id="about_btn", size="lg")
        gr.Button("Contact", elem_id="contact_btn", size="lg")
        gr.Button("Login", elem_id="login_btn", size="lg")
    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown('<h1 style="font-size: 4em;">Explore Better Decision Making</h1>', elem_id="main_text")
            with gr.Row():
                gr.Button("Get Started", elem_id="get_start_btn")
                gr.Button("Learn More", elem_id="learn_more_btn")
        gr.Image(value="../static/emergency1.jpg", elem_id="main_image")

interface.launch()