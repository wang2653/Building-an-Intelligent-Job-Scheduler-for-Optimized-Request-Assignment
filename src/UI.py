import gradio as gr
from PIL import Image
from oauth2client import client, file, tools
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
import integration


# Read the CSS file
with open("../assets/styles.css", "r") as css_file:
    styles_css = f"<style>{css_file.read()}</style>"

# Load the background image
"""background_image = Image.open("../static/background1.jpg")

def edit_image(image):
    return image"""

# Basic test function for doctor interface
def test_func(Patient_name, Arrival_time, Acuity_level, Treatment_plan):
    return ("Hello, " + Patient_name + ", you arrived at time " 
            + Arrival_time + ". Your acuity level is " + Acuity_level
            + ", and your treatment plan is " + Treatment_plan + '.')


def login_with_google():
    # Define the scopes required for the Google login
    SCOPES = ['https://www.googleapis.com/auth/userinfo.profile']
    creds = None

    # Load credentials if they exist
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # Refresh or create credentials if they are not valid or do not exist
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Return a success message if the user is logged in
    if creds:
        return "Login successful with Google!"
    else:
        return "Login failed. Please try again."


with gr.Blocks() as interface:
    gr.HTML(styles_css)  # Adding custom styles here

    with gr.Row(equal_height=True, elem_id="header_row"):
        gr.Markdown("# Intelligent Scheduler for Emergence Department", elem_id="page_title")
        gr.Button("Home", elem_id="home_btn", size="lg")
        gr.Button("About", elem_id="about_btn", size="lg")
        gr.Button("Contact", elem_id="contact_btn", size="lg")
        login_btn = gr.Button("Login", elem_id="login_btn", size="lg")

    login_btn.click(fn=login_with_google)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown('<h1 style="font-size: 4em;">Explore Better Decision Making</h1>', elem_id="main_text")
            with gr.Row():
                gr.Button("Get Started", elem_id="get_start_btn")
                gr.Button("Learn More", elem_id="learn_more_btn")

            # this defines the doctor interface
            doctor_interface = gr.Interface(
                fn = test_func,# will replace to process_and_simulate later
                inputs=["text", "text", "text", "text"],
                outputs=[gr.Textbox(label="Output", lines=5)],
                flagging_mode="never"
            )

        gr.Image(value="../static/emergency1.jpg", elem_id="main_image")

interface.launch()