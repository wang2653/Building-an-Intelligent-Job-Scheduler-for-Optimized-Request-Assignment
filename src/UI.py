import gradio as gr
import pandas as pd
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

# load the patient database
df = pd.read_csv('../data/patientdata7.csv')

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

# helper functions to open different pages
def open_main_page():
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def open_about_page():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def open_contact_page():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

with gr.Blocks() as interface:
    gr.HTML(styles_css)  # Adding custom styles here

    with gr.Row(equal_height=True, visible=True, elem_id="header_row") as main_header:
        gr.Markdown("# Intelligent Scheduler for Emergence Department", elem_id="page_title")
        home_btn = gr.Button("Home", elem_id="home_btn", size="lg")
        about_btn = gr.Button("About", elem_id="about_btn", size="lg")
        contact_btn = gr.Button("Contact", elem_id="contact_btn", size="lg")
        login_btn = gr.Button("Login", elem_id="login_btn", size="lg")

    login_btn.click(fn=login_with_google)
    
    with gr.Column(visible=True) as main_display:
        with gr.Row():
            with gr.Column():
                gr.Markdown('<h1 style="font-size: 4em;">Explore Better Decision Making</h1>', elem_id="main_text")
                
                with gr.Row():
                    gr.Button("Get Started", elem_id="get_start_btn")
                    gr.Button("Learn More", elem_id="learn_more_btn")
            gr.Image(value="../static/emergency1.jpg", elem_id="main_image")
        
        with gr.Row():
            # this defines the doctor interface
            doctor_interface = gr.Interface(
                fn = test_func,# will replace to process_and_simulate later
                inputs=["text", "text", "text", "text"],
                outputs=[gr.Textbox(label="Output", lines=5)],
                flagging_mode="never"
            )

            gr.Dataframe(value=df, label="Patient Data", elem_id="data_table")

    # The about page
    with gr.Column(visible=False) as about:
        gr.Markdown("# This is the about page, waiting for more contents.  \n" +
                    "For more info about our product, please see https://github.com/wang2653/Building-an-Intelligent-Job-Scheduler-for-Optimized-Request-Assignment.  \n" + 
                    "For any thoughts or problems regarding to the web design, please contact liruoyu996@gmail.com, thank you.")
    
    # The contact page
    with gr.Column(visible=False) as contact:
        gr.Markdown("# This is the contact page, waiting for more contents.  \n" +
                    "For more info about our product, please see https://github.com/wang2653/Building-an-Intelligent-Job-Scheduler-for-Optimized-Request-Assignment.  \n" + 
                    "For any thoughts or problems regarding to the web design, please contact liruoyu996@gmail.com, thank you.")
    
    # Set different page visibilities for different pages
    about_btn.click(fn=open_about_page, inputs=[], outputs=[main_header, main_display, about, contact])
    home_btn.click(fn=open_main_page, inputs=[], outputs=[main_header, main_display, about, contact])
    contact_btn.click(fn=open_contact_page, inputs=[], outputs=[main_header, main_display, about, contact])

interface.launch()