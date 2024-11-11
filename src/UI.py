import gradio as gr

# Injecting CSS directly into the HTML
custom_css = """
<style>
        #page_title {
        font-size: 3em;
        flex-grow: 3;
    }

    #home_btn,
    #about_btn,
    #contact_btn,
    #login_btn {
        flex-grow: 1;
    }
</style>
"""


# Read the CSS file
with open("../assets/styles.css", "r") as css_file:
    styles_css = f"<style>{css_file.read()}</style>"

with gr.Blocks() as interface:
    gr.HTML(styles_css)  # Adding custom styles here
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