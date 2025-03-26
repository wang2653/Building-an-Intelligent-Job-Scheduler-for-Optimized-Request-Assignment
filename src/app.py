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

# Load the accounts database
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
    if request.method == 'POST':
        action = request.forms.get('action')
        print(f"Action: {action}")

        if action == 'add':
            patient_id = request.forms.get('id')
            arrival_time = request.forms.get('arrival_time')
            acuity_level = request.forms.get('acuity_level')
            treatment_plan = request.forms.get('treatment_plan')
            treatment_time = request.forms.get('treatment_time')

            print(f"Received: ID={patient_id}, Arrival={arrival_time}, Acuity={acuity_level}, Plan={treatment_plan}, Time={treatment_time}")

            treatment_plan_arr = "[" + treatment_plan + "]"
            treatment_time_arr = "[" + treatment_time + "]"

            new_row = {
                "patient_id": patient_id,
                "arrival_time": arrival_time,
                "acuity_level": acuity_level,
                "treatment_plan_arr": treatment_plan_arr,
                "treatment_totaltime_arr": treatment_time_arr
            }

            df = pd.read_csv('../data/NEWpatientdata10.csv')
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv('../data/NEWpatientdata10.csv', index=False)
            print("Row added.")

        elif action == 'remove':
            pass

    return template('doc.html')

@app.route('/get_patients')
def get_patients():
    df = pd.read_csv('../data/current_patients_info.csv')
    return df.to_json(orient="records")

@app.route('/get_doctors')
def get_doctors():
    # Skip the first row and only read the next two rows (which have 19 columns)
    df = pd.read_csv('../data/current_doctors_info.csv', header=None, skiprows=1, nrows=2)
    # Row 0 (after skipping) is the statuses; row 1 is the tasks.
    status_list = df.iloc[0].tolist()
    task_list   = df.iloc[1].tolist()

    resources = []
    # Loop over 19 columns
    for i in range(19):
        resources.append({
            "status": int(status_list[i]),  # convert numpy.int64 to int
            "task": str(task_list[i])         # ensure task is a string
        })
    return {"resources": resources}

# Run the application
if __name__ == '__main__':
    run(app, host='localhost', port=8080, debug=True)
