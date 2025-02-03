from bottle import Bottle, run, template, static_file, request, redirect, TEMPLATE_PATH, response
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
users_df = pd.read_csv('../data/user_accounts.csv')

# Route for serving static files
@app.route('/static/<filepath:path>')
def serve_static(filepath):
    return static_file(filepath, root='../static')


# Basic test function for doctor interface
def test_func(Patient_ID, Arrival_time, Acuity_level, Treatment_plan):
    return f"Hello, {Patient_ID}, you arrived at time {Arrival_time}. Your acuity level is {Acuity_level}, and your treatment plan is {Treatment_plan}."

# Home page
@app.route('/', method=['GET', 'POST'])
def home():
    return template('index.html')

@app.route('/login', method=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Retrieve user input
        username = request.forms.get('username')
        password = request.forms.get('password')

        # Check credentials in CSV file
        if ((users_df['username'] == username) & (users_df['password'] == password)).any():
            # Store login status in a session cookie
            response.set_cookie("username", username, secret="your_secret_key", path="/")
            redirect('/doctor')  # Redirect to the doctor interface
        else:
            return template('login.html', error="Invalid username or password")

    return template('login.html', error=None)

# Doctor interface
@app.route('/doctor', method=['GET', 'POST'])
def doctor():
    return template('doc.html')

# Run the application
if __name__ == '__main__':
    run(app, host='localhost', port=8080, debug=True)
