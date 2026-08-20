# pyrefly: ignore [missing-import]
from bottle import Bottle, run, template, static_file, request, redirect, TEMPLATE_PATH, response
import pandas as pd
import os

# Define absolute directories for robustness
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')

# Create Bottle application
app = Bottle()

# Set templates path
if TEMPLATE_DIR not in TEMPLATE_PATH:
    TEMPLATE_PATH.insert(0, TEMPLATE_DIR)
if '../templates' not in TEMPLATE_PATH:
    TEMPLATE_PATH.insert(0, '../templates')

def get_users_df():
    """Dynamically load user accounts CSV with string data types."""
    csv_path = os.path.join(DATA_DIR, 'user_accounts.csv')
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path, dtype=str).fillna('')
        except Exception:
            return pd.DataFrame(columns=['username', 'password'])
    return pd.DataFrame(columns=['username', 'password'])

# Route for serving static files
@app.route('/static/<filepath:path>')
def serve_static(filepath):
    return static_file(filepath, root=STATIC_DIR)

# Basic test function for doctor interface
def test_func(Patient_ID, Arrival_time, Acuity_level, Treatment_plan):
    return f"Hello, {Patient_ID}, you arrived at time {Arrival_time}. Your acuity level is {Acuity_level}, and your treatment plan is {Treatment_plan}."

# Home page
@app.route('/', method=['GET', 'POST'])
def home():
    return template('index.html')

# Login and Register Route
@app.route('/login', method=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Retrieve user inputs
        username = (request.forms.get('username') or '').strip()
        password = (request.forms.get('password') or '').strip()
        action = (request.forms.get('action') or 'signin').strip().lower()

        csv_path = os.path.join(DATA_DIR, 'user_accounts.csv')
        users_df = get_users_df()

        # Handle Registration (Sign Up)
        if action == 'signup':
            if not username or not password:
                return template('login.html', error="Username and password cannot be empty", success=None)

            # Check if user already exists (case-insensitive username check)
            if not users_df.empty and (users_df['username'].astype(str).str.strip().str.lower() == username.lower()).any():
                return template('login.html', error=f"Account '{username}' already exists. Please Sign In.", success=None)

            # Append new user to user_accounts.csv
            new_user_row = pd.DataFrame([{'username': username, 'password': password}])
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                new_user_row.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                new_user_row.to_csv(csv_path, mode='w', header=True, index=False)

            print(f"Registered new account: {username}")
            # Set cookie and log in to doctor interface
            response.set_cookie("username", username, secret="your_secret_key", path="/")
            redirect('/doctor')

        # Handle Authentication (Sign In)
        else:
            if not username or not password:
                return template('login.html', error="Please enter username and password", success=None)

            # Exact match check with string conversion
            if not users_df.empty:
                match = users_df[
                    (users_df['username'].astype(str).str.strip() == username) & 
                    (users_df['password'].astype(str).str.strip() == password)
                ]
            else:
                match = pd.DataFrame()

            if not match.empty:
                # Store login status in a session cookie
                response.set_cookie("username", username, secret="your_secret_key", path="/")
                redirect('/doctor')  # Redirect to the doctor interface
            else:
                return template('login.html', error="Invalid username or password", success=None)

    return template('login.html', error=None, success=None)

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
            treatment_plan = request.forms.get('treatment_plan') or ''
            treatment_time = request.forms.get('treatment_time') or ''

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

            new_patient_csv = os.path.join(DATA_DIR, 'NEWpatientdata10.csv')
            if os.path.exists(new_patient_csv):
                df = pd.read_csv(new_patient_csv)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
            df.to_csv(new_patient_csv, index=False)
            print("Row added.")

        elif action == 'remove':
            pass

    return template('doc.html')

@app.route('/get_patients')
def get_patients():
    df = pd.read_csv(os.path.join(DATA_DIR, 'current_patients_info.csv'))
    response.content_type = 'application/json'
    return df.to_json(orient="records")

@app.route('/get_doctors')
def get_doctors():
    df = pd.read_csv(os.path.join(DATA_DIR, 'current_doctors_info.csv'), header=None, skiprows=1, nrows=2)
    status_list = df.iloc[0].tolist()
    task_list   = df.iloc[1].tolist()

    resources = []
    for i in range(min(len(status_list), 19)):
        resources.append({
            "status": int(status_list[i]),
            "task": str(task_list[i])
        })
    response.content_type = 'application/json'
    return {"resources": resources}

@app.route('/get_current_time')
def get_current_time():
    df = pd.read_csv(os.path.join(DATA_DIR, 'current_patients_info.csv'))
    current_time_value = df.iloc[0]['current_time'] if not df.empty and 'current_time' in df.columns else 0
    return {"current_time": int(current_time_value)}

@app.route('/search_patients', method=['GET'])
def search_patients():
    patient_id = request.query.get('patient_id')
    if not patient_id:
        return "[]"

    df = pd.read_csv(os.path.join(DATA_DIR, 'all_patients_info.csv'))
    try:
        filtered_df = df[df['patient_id'].astype(str) == str(patient_id).strip()]
    except Exception:
        return "[]"

    filtered_df.to_csv(os.path.join(DATA_DIR, 'search.csv'), index=False)
    response.content_type = 'application/json'
    return filtered_df.to_json(orient="records")

# Run the application
if __name__ == '__main__':
    run(app, host='localhost', port=8080, debug=True)
