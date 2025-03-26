import csv
from queue import PriorityQueue
import ast
import time
from collections import deque

import numpy as np
import math
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

'''Environment'''
# according to the paper
MEDICAL_RESOURCE_TYPE_ARR = ["DOCTOR", "NURSE", "XRAY", "CT"]
MEDICAL_RESOURCE_NUM_ARR = [8, 7, 2, 2]  # Number of each resource type
MEDICAL_TREATMENT_TYPE_ARR = [1,2,3,4,5,6,7,8,9]
RESOURCE_TREATMENT_MAP = {
    1: 1,     # 'DOCTOR',    #Triage
    2: 2,     # 'NURSE',     #Registration
    3: 1,     # 'DOCTOR',    #Evaluation
    4: 2,     # 'NURSE',     #Laboratory
    5: 3,     # 'XRAY',      #X-ray
    6: 2,     # 'NURSE',     #Consultation
    7: 4,     # 'CT',        #CT scan
    8: 2,     # 'NURSE',     #Discharge
    9: 2,     # 'NURSE'      #Admission
}
TREATMENT_MAP = {
    1: "Triage",
    2: "Registration",
    3: "Evaluation",
    4: "Laboratory",
    5: "X-ray",
    6: "Consultation",
    7: "CT scan",
    8: "Discharge",
    9: "Admission"
}

TREATMENT_TIME = {
    1: 1,
    2: 1,
    3: 14,
    4: 35,
    5: 1,
    6: 15,
    7: 29,
    8: 30,
    9: 1
}

ACUITY_WEIGHT_MAP = {
    1: 1,  # Acuity level 1
    2: 1,  # Acuity level 2
    3: 1,  # Acuity level 3
    4: 15, # Acuity level 4
    5: 30  # Acuity level 5
}

TOTAL_SIMULATION_TIME = 1500 # minutes

# record the last paitent id to check new added patient 
LAST_PATIENT_ID = [0]


# Patient object list 
# {patient_id:Patient, ...} 
global_patient = {}
# {arrival_time , Patient, ...} 
global_patient_by_time = PriorityQueue()

# current patients in the ED
current_patients = set()
all_current_patients = set()

# run_simulation
global_medical_resource = []

# weighted waiting time of each patient
global_weighted_waiting_time = []

'''
example Patient object, data read from simulated data:
patient_id = 1 #first patient in the simulated data
arrival_time = 10 #arrives 10 minutes since start of simulation time
acuity_level = 2 #assigned how severe the patient is
treatment_plan_arr = ["A", "B", "G"] #assigned treatment plan 
treatment_time_arr = [10, 15, 5] #assigned simulated duration time for each treatment
wait_time_arr = [] #minutes patient waited for each treatment, to be recorded after simulation start
'''
class Patient:
    def __init__(self, patient_id, arrival_time, acuity_level, treatment_plan_arr, treatment_totaltime_arr):
        # Preassigned in simulated data 
        self.patient_id = patient_id                            # Unique identifier for the patient
        self.arrival_time = arrival_time                        # Time of arrival
        self.acuity_level = acuity_level                        # Acuity level (integer 1 to 5)
        self.treatment_plan_arr = treatment_plan_arr            # List of treatments plan (integer 1 to 7)
        self.treatment_totaltime_arr = treatment_totaltime_arr  # List of treatments time (simulated in minutes)
        # Dynamically updating during simulation
        self.treatment_starttime_arr =[None] * len(treatment_plan_arr)          # List of wait times for each treatment (to be recorded)
        self.current_treatment_index = 0   # initilize to be 0 < len(treatment_plan_arr)
        self.current_treatment_time = 0 #initialize to be 0
        self.treatment_remaining_time = 0 #initialize to be 0



'''
example
medical_resource_type = DOCTOR
medical_resource_num = 3
patient_intreatment= [1,3,5] # patient index
patient_waiting = [6,7] # patient index 
'''
class Resource:
    def __init__(self, medical_resource_type, medical_resource_num):
        # Preassigned
        self.medical_resource_type = medical_resource_type
        self.medical_resource_num = medical_resource_num
        # Dynamically changing during simulation
        self.patientid_in_treatment_arr = [] # Patient id list
        self.patientid_waiting_arr = [] # Patient id list

        self.doctor_arr = [0, 0, 0, 0, 0, 0, 0, 0] # Patient id list
        self.doctor_treatment_arr = [0, 0, 0, 0, 0, 0, 0, 0] # treatment name list

        self.nurse_arr = [0, 0, 0, 0, 0, 0, 0] # Patient id list
        self.nurse_treatment_arr = [0, 0, 0, 0, 0, 0, 0] # treatment name list

        self.xray_arr = [0, 0] # Patient id list
        self.xray_treatment_arr = [0, 0] # treatment name list

        self.ct_arr = [0, 0] # Patient id list
        self.ct_treatment_arr = [0, 0] # treatment name list



    def resource_is_available(self):
        if len(self.patientid_in_treatment_arr) >= self.medical_resource_num:
            return False
        else:
            return True
        
    def waiting_is_empty(self):
        if len(self.patientid_waiting_arr) > 0:
            return False
        else:
            return True
    
    def update_patient_in_treatment(self):
        patient_finish_treatment = []
        patients_to_remove = []
        
        for patient_id in self.patientid_in_treatment_arr:
            current_treatment_index = global_patient[patient_id].current_treatment_index
            current_treatment_time = global_patient[patient_id].current_treatment_time + 1
            global_patient[patient_id].current_treatment_time = current_treatment_time
            current_treatment_totaltime = global_patient[patient_id].treatment_totaltime_arr[current_treatment_index]

            if current_treatment_time >= current_treatment_totaltime:
                global_patient[patient_id].current_treatment_index += 1
                global_patient[patient_id].current_treatment_time = 0
                patient_finish_treatment.append(patient_id)
                patients_to_remove.append(patient_id)
        
        
        for patient_id in patients_to_remove:
            self.patientid_in_treatment_arr.remove(patient_id)

            if self.medical_resource_type == "DOCTOR":  
                index_to_remove = self.doctor_arr.index(patient_id)
                self.doctor_arr[index_to_remove] = 0
                self.doctor_treatment_arr[index_to_remove] = 0

            if self.medical_resource_type == "NURSE":  
                index_to_remove = self.nurse_arr.index(patient_id)
                self.nurse_arr[index_to_remove] = 0
                self.nurse_treatment_arr[index_to_remove] = 0

            if self.medical_resource_type == "XRAY":  
                index_to_remove = self.xray_arr.index(patient_id)
                self.xray_arr[index_to_remove] = 0
                self.xray_treatment_arr[index_to_remove] = 0
                
            if self.medical_resource_type == "CT":  
                index_to_remove = self.ct_arr.index(patient_id)
                self.ct_arr[index_to_remove] = 0
                self.ct_treatment_arr[index_to_remove] = 0
            
        return patient_finish_treatment

    
    def add_patient_to_waiting(self, patient_id):
        self.patientid_waiting_arr.append(patient_id)
            
    def add_patient_to_treatment(self, patient_id, current_time):
        if self.resource_is_available() == False:
            print("ERROR: not enough medical resource to schedule this patient")   
            return

        self.patientid_in_treatment_arr.append(patient_id)
        self.patientid_waiting_arr.remove(patient_id)
        current_treatment_index = global_patient[patient_id].current_treatment_index
        global_patient[patient_id].treatment_starttime_arr[current_treatment_index] = current_time

        current_treatment_id = global_patient[patient_id].treatment_plan_arr[current_treatment_index]

        if self.medical_resource_type == "DOCTOR":
            available_idx = []
            for i in range(8):
                if self.doctor_arr[i] == 0:
                    available_idx.append(i)
            
            random_idx = random.choice(available_idx)

            self.doctor_arr[random_idx] = patient_id
            self.doctor_treatment_arr[random_idx] = TREATMENT_MAP[current_treatment_id]
        
        if self.medical_resource_type == "NURSE":
            available_idx = []
            for i in range(7):
                if self.nurse_arr[i] == 0:
                    available_idx.append(i)
            
            random_idx = random.choice(available_idx)

            self.nurse_arr[random_idx] = patient_id
            self.nurse_treatment_arr[random_idx] = TREATMENT_MAP[current_treatment_id]
        
        if self.medical_resource_type == "XRAY":
            available_idx = []
            for i in range(2):
                if self.xray_arr[i] == 0:
                    available_idx.append(i)
            
            random_idx = random.choice(available_idx)

            self.xray_arr[random_idx] = patient_id
            self.xray_treatment_arr[random_idx] = TREATMENT_MAP[current_treatment_id]
        
        if self.medical_resource_type == "CT":
            available_idx = []
            for i in range(2):
                if self.ct_arr[i] == 0:
                    available_idx.append(i)
            
            random_idx = random.choice(available_idx)

            self.ct_arr[random_idx] = patient_id
            self.ct_treatment_arr[random_idx] = TREATMENT_MAP[current_treatment_id]

        
    def get_waiting_state(self):
        """
        Build the state representation.
        
        Args:
            self

        Return
            state vector, each state vector is reflected as the ratio to standardize vector numbers, which is that the vector numbers are expressed between 0 and 1.
        """
        total_patient_waiting = len(self.patientid_waiting_arr)
        if total_patient_waiting == 0:
            return [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # acuity distrbution
        acuity_counts = [0,0,0,0,0]
        # values are proportions (e.g., 0.5 means 50% of patients are at that acuity level)
        for patient_id in self.patientid_waiting_arr:
            acuity_index = global_patient[patient_id].acuity_level - 1
            acuity_counts[acuity_index] += 1

        # treatment Types: Represented by the next seven elements (positions 5 to 11).
        treatment_type_counts = [0,0,0,0,0,0,0,0,0]
        # Each position corresponds to a treatment type from 1 to 7.
        for patient_id in self.patientid_waiting_arr:
            current_treatment_index = global_patient[patient_id].current_treatment_index
            current_treatment_type = global_patient[patient_id].treatment_plan_arr[current_treatment_index]-1
            treatment_type_counts[current_treatment_type] += 1

        return [float(distribution / total_patient_waiting) for distribution in (acuity_counts + treatment_type_counts)]

    def get_inTreatment_state(self):
        total_patient_inTreatment = len(self.patientid_in_treatment_arr)
        if total_patient_inTreatment == 0:
            return [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # acuity distrbution
        acuity_counts = [0,0,0,0,0]
        for patient_id in self.patientid_in_treatment_arr:
            acuity_index = global_patient[patient_id].acuity_level - 1
            acuity_counts[acuity_index] += 1

        #treatment type distribution
        treatment_type_counts = [0,0,0,0,0,0,0,0,0]
        for patient_id in self.patientid_in_treatment_arr:
            current_treatment_index = global_patient[patient_id].current_treatment_index
            current_treatment_type = global_patient[patient_id].treatment_plan_arr[current_treatment_index]-1
            treatment_type_counts[current_treatment_type] += 1

        return [float(distribution / total_patient_inTreatment) for distribution in (acuity_counts + treatment_type_counts)]


def load_patient_data():

    file_path = "./data/NEWpatientdata10.csv"

    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader: 
            patient_id = int(row['patient_id'])
            arrival_time = int(row['arrival_time'])
            acuity_level = int(row['acuity_level'])
            # Convert to a list of integers
            treatment_plan_arr = ast.literal_eval(row['treatment_plan_arr'])
            treatment_totaltime_arr = ast.literal_eval(row['treatment_totaltime_arr'])

            global_patient[patient_id] = Patient(patient_id, arrival_time, acuity_level, treatment_plan_arr, treatment_totaltime_arr)
            global_patient_by_time.put((arrival_time, patient_id))
            LAST_PATIENT_ID[0] = patient_id

def add_new_manually():
    file_path = "./data/NEWpatientdata10.csv"

    with open(file_path, mode='r') as file:
        # only read the last line
        last_line  = deque(csv.DictReader(file), maxlen=1)
        row = last_line[0]
        patient_id = int(row['patient_id'])
        # print("last line: ", patient_id)

        last_id = LAST_PATIENT_ID[0]
        if patient_id != last_id:
            # print("asdasd/n")
            arrival_time = int(row['arrival_time'])
            acuity_level = int(row['acuity_level'])
            # Convert to a list of integers
            treatment_plan_arr = ast.literal_eval(row['treatment_plan_arr'])
            treatment_totaltime_arr = ast.literal_eval(row['treatment_totaltime_arr'])

            global_patient[patient_id] = Patient(patient_id, arrival_time, acuity_level, treatment_plan_arr, treatment_totaltime_arr)
            global_patient_by_time.put((arrival_time, patient_id))

            LAST_PATIENT_ID[0] = patient_id



def init_simulation():
    
    #initialize all Patient object
    load_patient_data()

    #initialize all Resource object
    for i in range(len(MEDICAL_RESOURCE_TYPE_ARR)):
        resource = Resource(MEDICAL_RESOURCE_TYPE_ARR[i], MEDICAL_RESOURCE_NUM_ARR[i])
        global_medical_resource.append(resource)

def evaluation():
    for key, value in global_patient.items():
        # print(f"{key}: {value.arrival_time}")
        # print(f"{key}: {value.treatment_starttime_arr}")
        # print(f"{key}: {value.treatment_totaltime_arr}")

        start_time = value.treatment_starttime_arr
        total_time = value.treatment_totaltime_arr
        arrival_time = value.arrival_time
        acuity_level = value.acuity_level

        # first treatment waiting time
        waiting_time = 0
        waiting_time += start_time[0] - arrival_time

        # remaining treatments waiting time
        for i in range(1, len(start_time)):
            waiting_time += start_time[i] - (total_time[i-1] + start_time[i-1])
            
        # print(f"{key}: {waiting_time} * {ACUITY_WEIGHT_MAP[acuity_level]}\n")
        global_weighted_waiting_time.append(ACUITY_WEIGHT_MAP[acuity_level] * waiting_time)

    # print(f"weighted waiting time: {global_weighted_waiting_time}")
    # print(f"sum of weighted waiting time: {sum(global_weighted_waiting_time)}")
    # print(f"max of weighted waiting time: {max(global_weighted_waiting_time)}")
    # print(f"average of weighted waiting time: {sum(global_weighted_waiting_time) / len(global_weighted_waiting_time)}")
    return sum(global_weighted_waiting_time) / len(global_weighted_waiting_time)

def evaluation_final():
    global_weighted_waiting_time = []
    acuity_1 = []
    acuity_2 = []
    acuity_3 = []
    acuity_4 = []
    acuity_5 = []

    for key, value in global_patient.items():
        # print(f"{key}: {value.arrival_time}")
        # print(f"{key}: {value.treatment_starttime_arr}")
        # print(f"{key}: {value.treatment_totaltime_arr}")

        start_time = value.treatment_starttime_arr
        total_time = value.treatment_totaltime_arr
        arrival_time = value.arrival_time
        acuity_level = value.acuity_level

        # first treatment waiting time
        waiting_time = 0
        waiting_time += start_time[0] - arrival_time

        # remaining treatments waiting time
        for i in range(1, len(start_time)):
            waiting_time += start_time[i] - (total_time[i-1] + start_time[i-1])

        # print(f"{key}: {waiting_time} * {ACUITY_WEIGHT_MAP[acuity_level]}\n")
        global_weighted_waiting_time.append(ACUITY_WEIGHT_MAP[acuity_level] * waiting_time)
        if (acuity_level == 1):
            acuity_1.append(ACUITY_WEIGHT_MAP[acuity_level] * waiting_time)
        elif (acuity_level == 2):
            acuity_2.append(ACUITY_WEIGHT_MAP[acuity_level] * waiting_time)
        elif (acuity_level == 3):
            acuity_3.append(ACUITY_WEIGHT_MAP[acuity_level] * waiting_time)
        elif (acuity_level == 4):
            acuity_4.append(ACUITY_WEIGHT_MAP[acuity_level] * waiting_time)
        elif (acuity_level == 5):
            acuity_5.append(ACUITY_WEIGHT_MAP[acuity_level] * waiting_time)

    # print(f"weighted waiting time: {global_weighted_waiting_time}")
    # print(f"sum of weighted waiting time: {sum(global_weighted_waiting_time)}")
    print(f"acuity_1 average of weighted waiting time: {sum(acuity_1) / len(acuity_1)}")
    print(f"acuity_2 average of weighted waiting time: {sum(acuity_2) / len(acuity_2)}")
    print(f"acuity_3 average of weighted waiting time: {sum(acuity_3) / len(acuity_3)}")
    print(f"acuity_4 average of weighted waiting time: {sum(acuity_4) / len(acuity_4)}")
    print(f"acuity_5 average of weighted waiting time: {sum(acuity_5) / len(acuity_5)}")

    print(f"average of weighted waiting time: {sum(global_weighted_waiting_time) / len(global_weighted_waiting_time)}")
    print(global_weighted_waiting_time)

# Reward function
def compute_reward(patient_id, current_time):

    start_time = global_patient[patient_id].treatment_starttime_arr
    total_time = global_patient[patient_id].treatment_totaltime_arr
    arrival_time = global_patient[patient_id].arrival_time
    acuity_level = global_patient[patient_id].acuity_level

    # first treatment waiting time
    if start_time[0] == None:
        return current_time - arrival_time

    waiting_time = 0
    waiting_time += start_time[0] - arrival_time

    # remaining treatments waiting time
    for i in range(1, len(start_time)):
      if start_time[i] != None:
        waiting_time += start_time[i] - (total_time[i-1] + start_time[i-1])
      else:
        waiting_time += max((current_time - (total_time[i-1] + start_time[i-1])), 0)
        break

    # print(f"{patient_id}: {waiting_time} * {ACUITY_WEIGHT_MAP[acuity_level]}\n")
    return ACUITY_WEIGHT_MAP[acuity_level] * waiting_time
    
def compute_waiting_time(patient_id, current_time):
    start_time = global_patient[patient_id].treatment_starttime_arr
    total_time = global_patient[patient_id].treatment_totaltime_arr
    arrival_time = global_patient[patient_id].arrival_time


    # first treatment waiting time
    if start_time[0] == None:
        return current_time - arrival_time

    waiting_time = 0
    waiting_time += start_time[0] - arrival_time

    # remaining treatments waiting time
    for i in range(1, len(start_time)):
      if start_time[i] != None:
        waiting_time += start_time[i] - (total_time[i-1] + start_time[i-1])
      else:
        waiting_time += max((current_time - (total_time[i-1] + start_time[i-1])), 0)
        break

    
    return waiting_time


# Define the DQN Agent
class DQNAgent:
    def __init__(self, gamma=0.9999, epsilon=1.0, epsilon_min=1e-8, epsilon_decay=0.99, learning_rate=1e-7, batch_size=100):
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Main Q-Network
        self.model = self._build_model()
        # Target Q-Network
        self.target_model = self._build_model()
        # Initialize target network weights
        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Loss function
        self.loss_fn = nn.MSELoss()

    def build_model():
        model = nn.Sequential(
            # Conv Layer 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (16, 9)

            # Conv Layer 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (32, 4)

            # Conv Layer 3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (64, 2)

            # Flatten
            nn.Flatten(),  # (64 * 2 = 128)

            # Fully Connected Layers
            nn.Linear(64 * 2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            )

        return model

    def update_target_network(self):
        # Copy weights from main model to target model
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, all_patient_info, reward, next_state):
        self.memory.append((state, all_patient_info, reward, next_state))

    def save_model(self, file_path):
        """Save the model's state_dict to a file."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load the model's state_dict from a file."""
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {file_path}")

    def _build_model(self):
        return nn.Sequential(nn.Linear(3, 10),nn.LeakyReLU(negative_slope=0.01),nn.Linear(10, 4),nn.LeakyReLU(negative_slope=0.01),nn.Linear(4, 1),)

    def act(self, resource, state, current_time):
        waiting_patients = resource.patientid_waiting_arr
        if not waiting_patients:
            # print("no one waiting", waiting_patients)
            return None, None, None

        # Return the patient id with maximum q_value
        q_values_arr = []
        all_patient_info = []

        for patient_id in waiting_patients:
            patient_info = []

            # acuity_level
            patient_info.append(global_patient[patient_id].acuity_level / 5)
            # waiting time so far
            # patient_info.append(compute_reward(patient_id, current_time) / 10000)  # need to be normalized
            waiting_time_so_far = compute_reward(patient_id, current_time)
            patient_info.append(math.log1p(waiting_time_so_far) / 9)
            # current treatment type
            current_treatment_index = global_patient[patient_id].current_treatment_index
            current_treatment_type = global_patient[patient_id].treatment_plan_arr[current_treatment_index]
            # patient_info.append(current_treatment_type / 9) 废信息
            # current treatment average processing time
            patient_info.append(TREATMENT_TIME[current_treatment_type] / 35)

            # input_data = state + patient_info
            input_data = patient_info
            input_tensor = torch.tensor(input_data, dtype=torch.float32)

            # Get the Q-value predictions from the model
            q_values = self.model(input_tensor.unsqueeze(0))  # Add batch dimension
            q_values_arr.append(q_values.item())

            all_patient_info.append(patient_info)

        # print(f"q values {q_values_arr}")
        # Get the patient ID and info associated with the highest Q-value
        max_q_value_index = np.argmax(q_values_arr)
        # print(f"max q values {q_values_arr[max_q_value_index]}\n")
        patient_id_to_serve = waiting_patients[max_q_value_index]
        patient_info_to_serve = all_patient_info[max_q_value_index]

        # # Return random patient id based on epsilon-greedy strategy
        # if np.random.rand() <= self.epsilon:
        #     random_index = random.randrange(len(waiting_patients))
        #     return waiting_patients[random_index], all_patient_info[random_index], all_patient_info

        return patient_id_to_serve, patient_info_to_serve, all_patient_info

    def replay(self):
        if len(self.memory) >= self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            for state, patient_info, reward, next_state in minibatch:
                # target = -reward / 10000
                target = -(math.log1p(reward) / 9)


                # Calculate the maximum Q-value for the next state
                # input_data = next_state + patient_info
                input_data = patient_info
                input_tensor = torch.tensor(input_data, dtype=torch.float32)

                # Get Q-value predictions for the next state
                with torch.no_grad():
                    q_values = self.target_model(input_tensor.unsqueeze(0))  # Add batch dimension
                # q_values_arr.append(q_values.item())

                target += self.gamma * q_values.item()

                # Convert state to tensor and perform a single step of gradient descent
                # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state_tensor = torch.tensor(patient_info, dtype=torch.float32).unsqueeze(0)
                target_tensor = torch.tensor([target], dtype=torch.float32)

                # Forward pass and compute loss
                self.optimizer.zero_grad()
                predicted_q_value = self.model(state_tensor).squeeze(1)  # Flatten the output
                loss = self.loss_fn(predicted_q_value, target_tensor)

                # Backpropagate the loss
                loss.backward()
                self.optimizer.step()

            # Decay epsilon after every replay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

def run_simulation(agent):
    # reset the environment
    init_simulation()
    current_time = 0

    while current_time <= TOTAL_SIMULATION_TIME:
        # print('time ',current_time)
        add_new_manually()
        # update patient in treatment, look at each resource to see if patient finish treatment
        patient_new = []

        for resource in global_medical_resource:
            patient_finished = resource.update_patient_in_treatment()
            patient_new.extend(patient_finished)

        # look at simulation data to see if we have new patient at current time, add to patient_new
        while not global_patient_by_time.empty():
            arrival_time, patient_id = global_patient_by_time.get()
            if arrival_time == current_time:
                # print(f'New patient arrived: [Patient ID: {patient_id}]')
                patient_new.append(patient_id)
            else:
                global_patient_by_time.put((arrival_time, patient_id))
                break

        # assign patient to resources waiting que (according to mapping) if not reached end of treatment plan
        for patient_id in patient_new:
            # index of treatment_plan_arr
            patient = global_patient[patient_id]
            current_treatment_index = global_patient[patient_id].current_treatment_index
            # not finish all treatments
            if current_treatment_index < len(patient.treatment_plan_arr):
                # upcoming treatment id 
                current_treatment_id = patient.treatment_plan_arr[current_treatment_index]
                # the resource id of the upcoming treatment
                resource_id = RESOURCE_TREATMENT_MAP[current_treatment_id]
                # assign patient to waiting queue of this resource
                resource = global_medical_resource[resource_id - 1]
                # print(f'Patient {patient_id} assigned to resource {resource.medical_resource_type}')
                resource.add_patient_to_waiting(patient_id)

                # calculate the total treatment pattern remaining time
                remaining_time = 0
                for i in range(current_treatment_index, len(patient.treatment_plan_arr)):
                    remaining_time += patient.treatment_totaltime_arr[i]
                patient.treatment_remaining_time = remaining_time
                # print(f'Patient {patient_id} remains {remaining_time}')

                current_patients.add(patient_id)
                all_current_patients.add(patient_id)
            else:
                current_patients.discard(patient_id)


        # update state 
        # if resource available and waiting line not empty then make an action for that resource
        for resource in global_medical_resource:
            while (resource.resource_is_available() == True) and (resource.waiting_is_empty() == False):
                # s_t, a_t
                state = resource.get_waiting_state()
                action, patient_info_to_serve, _ = agent.act(resource, state, current_time)
                select_patient_id = action
                # print(f'Patient {select_patient_id} with info {patient_info_to_serve}')

                # Execute a_t and Observe 𝑟_𝑡, 𝑠_𝑡+1
                resource.add_patient_to_treatment(select_patient_id, current_time)
                # print(f'Patient {select_patient_id} started treatment with {resource.medical_resource_type}')
                # if resource.medical_resource_type == "DOCTOR":  
                #     print(resource.doctor_arr)
                #     print(resource.treatment_arr)

                next_state = resource.get_waiting_state()
                reward = compute_reward(select_patient_id, current_time)
                # print(f'Patient {select_patient_id} earned reward {reward}')

                # get all patient info at t+1
                action, _, _ = agent.act(resource, next_state, current_time)
                if action == None:
                    continue

                # Save transition (𝑠_𝑡 + p_𝑡, p_𝑡+1, 𝑟_𝑡, 𝑠_𝑡+1) in 𝑀
                agent.remember(state + patient_info_to_serve, patient_info_to_serve, reward, next_state)
                



        current_patients_info = []
        for patient_id in current_patients:
            current_treatment_index = global_patient[patient_id].current_treatment_index
            current_treatment_id = global_patient[patient_id].treatment_plan_arr[current_treatment_index]

            # current_time, patient_id, acuity_level, waiting_time, current_treatment, remaining_time, status(1: in treatment; 0: waiting)
            patient_info = []
            patient_info.append(current_time)
            patient_info.append(patient_id)
            patient_info.append(global_patient[patient_id].acuity_level)
            patient_info.append(compute_waiting_time(patient_id, current_time))
            patient_info.append(TREATMENT_MAP[current_treatment_id])
            patient_info.append(global_patient[patient_id].treatment_remaining_time)

            if global_patient[patient_id].current_treatment_time > 0:
                patient_info.append(1)
            else:
                patient_info.append(0)

            current_patients_info.append(patient_info)

        # write the current patients info into the file
        with open("./data/current_patients_info.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["current_time", "patient_id", "acuity_level", "waiting_time", "current_treatment", "remaining_time", "status"])  # Writing header
            writer.writerows(current_patients_info)
            

        current_patients_info = []
        for patient_id in all_current_patients:
            current_treatment_index = global_patient[patient_id].current_treatment_index
            if current_treatment_index == len(global_patient[patient_id].treatment_plan_arr):
                treatment_name = "Done"
            else:
                current_treatment_id = global_patient[patient_id].treatment_plan_arr[current_treatment_index]
                treatment_name = TREATMENT_MAP[current_treatment_id]

            # current_time, patient_id, acuity_level, waiting_time, current_treatment, remaining_time, status(1: in treatment; 0: waiting)
            patient_info = []
            patient_info.append(current_time)
            patient_info.append(patient_id)
            patient_info.append(global_patient[patient_id].acuity_level)
            patient_info.append(compute_waiting_time(patient_id, current_time))
            patient_info.append(treatment_name)

            if global_patient[patient_id].current_treatment_time > 0:
                patient_info.append(1)
            else:
                patient_info.append(0)

            current_patients_info.append(patient_info)

        # write the current patients info into the file
        with open("./data/all_patients_info.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["current_time", "patient_id", "acuity_level", "waiting_time", "current_treatment", "status"])  # Writing header
            writer.writerows(current_patients_info)




        resources_info = []
        treatments_info = []
        for resource in global_medical_resource:    
            if resource.medical_resource_type == "DOCTOR":  
                resources_info.extend(resource.doctor_arr)
                treatments_info.extend(resource.doctor_treatment_arr)

            if resource.medical_resource_type == "NURSE":  
                resources_info.extend(resource.nurse_arr)
                treatments_info.extend(resource.nurse_treatment_arr)
            if resource.medical_resource_type == "XRAY":  
                resources_info.extend(resource.xray_arr)
                treatments_info.extend(resource.xray_treatment_arr)

            if resource.medical_resource_type == "CT":  
                resources_info.extend(resource.ct_arr)
                treatments_info.extend(resource.ct_treatment_arr)


        current_resources_info = []
        current_resources_info.append(resources_info)
        current_resources_info.append(treatments_info)

        # write the current doctors info into the file
        with open("./data/current_doctors_info.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["doctor_1", "doctor_2", "doctor_3", "doctor_4", "doctor_5", "doctor_6", "doctor_7", "doctor_8"])  # Writing header
            writer.writerows(current_resources_info)


        current_time = current_time + 1
        time.sleep(0.01)




new = DQNAgent()
new.model.load_state_dict(torch.load('data/dqn_weights_best_10_72.pth'))

run_simulation(new)
evaluation_final()

