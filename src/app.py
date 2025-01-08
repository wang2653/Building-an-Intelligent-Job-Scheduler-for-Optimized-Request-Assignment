from bottle import Bottle, run, template, static_file, request, redirect, TEMPLATE_PATH
import pandas as pd
from oauth2client import client, file, tools
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle

# Create Bottle application
app = Bottle()

# Set templates path
TEMPLATE_PATH.insert(0, "../templates")

# Load the patient database
df = pd.read_csv('../data/patientdata7.csv')

# Route for serving static files
@app.route('/static/<filepath:path>')
def serve_static(filepath):
    return static_file(filepath, root='../static')

# Google login function
def login_with_google():
    SCOPES = ['https://www.googleapis.com/auth/userinfo.profile']
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    if creds:
        return "Login successful with Google!"
    else:
        return "Login failed. Please try again."

# Basic test function for doctor interface
def test_func(Patient_name, Arrival_time, Acuity_level, Treatment_plan):
    return f"Hello, {Patient_name}, you arrived at time {Arrival_time}. Your acuity level is {Acuity_level}, and your treatment plan is {Treatment_plan}."

# Home page
@app.route('/')
def home():
    # Generate html table from dataset
    table_html = df.to_html(index=False, classes='data-table', escape=False)
    return template('home.html', title="Intelligent Scheduler for Emergency Department", table_html=table_html)

# About page
@app.route('/about')
def about():
    return template('about', title="About")

# Contact page
@app.route('/contact')
def contact():
    return template('contact', title="Contact")

# Doctor interface
@app.route('/doctor', method=['GET', 'POST'])
def doctor_interface():
    if request.method == 'POST':
        Patient_name = request.forms.get('Patient_name')
        Arrival_time = request.forms.get('Arrival_time')
        Acuity_level = request.forms.get('Acuity_level')
        Treatment_plan = request.forms.get('Treatment_plan')
        result = test_func(Patient_name, Arrival_time, Acuity_level, Treatment_plan)
        return template('doctor', title="Doctor Interface", result=result)
    return template('doctor', title="Doctor Interface", result=None)

# Run the application
if __name__ == '__main__':
    run(app, host='localhost', port=8080, debug=True)
