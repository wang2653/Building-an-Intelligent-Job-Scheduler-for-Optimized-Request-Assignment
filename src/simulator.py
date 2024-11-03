# # Data handling and manipulation
# import numpy as np
# import pandas as pd

# # Deep learning with PyTorch
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms

import csv

'''Environment'''
MEDICAL_RESOURCE_TYPE_ARR = ["DOCTOR", "NURSE", "XRAY", "CT"]
MEDICAL_RESOURCE_NUM_ARR = [4, 3, 2, 1]  # Number of each resource type
MEDICAL_TREATMENT_TYPE_ARR = [1,2,3,4,5,6,7]
RESOURCE_TREATMENT_MAP = {
    1: 'DOCTOR',
    2: 'NURSE',
    3: 'XRAY',
    4: 'DOCTOR',
    5: 'NURSE',
    6: 'DOCTOR',
    7: 'DOCTOR'
}
ACUITY_WEIGHT_MAP = {
    1: 30,  # Acuity level 1
    2: 15,  # Acuity level 2
    3: 10,  # Acuity level 3
    4: 5, # Acuity level 4
    5: 1  # Acuity level 5
}

TOTAL_SIMULATION_TIME = 120 # minutes

# Initialize counts
num_acuity_levels = 5
num_treatment_type = 7

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
        self.patient_id = patient_id              # Unique identifier for the patient
        self.arrival_time = arrival_time          # Time of arrival
        self.acuity_level = acuity_level          # Acuity level (integer 1 to 5)
        self.treatment_plan_arr = treatment_plan_arr# List of treatments plan (integer 1 to 7)
        self.treatment_totaltime_arr = treatment_totaltime_arr# List of treatments time (simulated in minutes)
        # Dynamically updating during simulation
        self.treatment_starttime_arr =[None] * len(treatment_plan_arr)          # List of wait times for each treatment (to be recorded)
        self.current_treatment_index = 0   # initilize to be 0 < len(treatment_plan_arr)
        self.current_treatment_time = 0 #initialize to be 0


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
        self.patientid_in_treatment_arr = None # Patient id list
        self.patientid_waiting_arr = None # Patient id list

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
        
        for patient_id in self.patientid_in_treatment_arr:
            current_treatment_index = global_patient[patient_id].current_treatment_index
            current_treatment_time = global_patient[patient_id].current_treatment_time + 1
            current_treatment_totaltime = global_patient[patient_id].treatment_totaltime_arr[current_treatment_index]

            if current_treatment_time >= current_treatment_totaltime:
                # move on to the next treatment 
                global_patient[patient_id].current_treatment_index = global_patient[patient_id].current_treatment_index + 1
                global_patient[patient_id].current_treatment_time = 0
                patient_finish_treatment.append(patient_id)
                self.patientid_in_treatment_arr.remove(patient_id)
        
        return patient_finish_treatment
    
    def add_patient_to_waiting(self, patient_id_arr):
        self.patientid_waiting_arr.append(patient_id_arr)
            
    def add_patient_to_treatment(self, patient_id, current_time):
        if self.resource_is_available() == False:
            print("ERROR: not enough medical resource to schedule this patient")   
            return

        self.patientid_in_treatment_arr.append(patient_id)
        self.patientid_waiting_arr.remove(patient_id)
        current_treatment_index = global_patient[patient_id].current_treatment_index
        global_patient[patient_id].treatment_starttime_arr[current_treatment_index] = current_time
        
    def get_state(self):
        """
        Build the state representation.
        
        Args:
            self

        Return
            state vector, each state vector is reflected as the ratio to standardize vector numbers, which is that the vector numbers are expressed between 0 and 1.
        """
        total_patient_waiting = len(self.patientid_waiting_arr)
        # acuity distrbution
        acuity_counts = []
        for patient_id in self.patientid_waiting_arr:
            acuity_index = global_patient[patient_id].acuity_level - 1
            acuity_counts[acuity_index] += 1

        #treatment type distribution
        treatment_type_counts = []
        for patient_id in self.patientid_waiting_arr:
            current_treatment_index = global_patient[patient_id].current_treatment_index
            current_treatment_type = global_patient[patient_id].treatment_plan_arr[current_treatment_index]-1
            treatment_type_counts[current_treatment_type] += 1

        return (acuity_counts + treatment_type_counts) / total_patient_waiting

# Reward function
def compute_reward(patient, current_time):
    """
    Computes the reward when assigning a patient to a resource.
    
    Args:
        patient (Patient): The patient being assigned.
        current_time (float): Current time in minutes.
    
    Returns:
        float: Reward value.
    """
    reward = {}

    return reward

# Patient object list 
# {patient_id:Patient, ...} 
global_patient = {}
# pattern_id : [treatment_1, treatment_2, ...] 
global_treatment = {} 
# treatment_id : duration
global_treatment_duration = {}

# run_simulation
global_medical_resource = []

def load_patient_data(file_path):
    """
        load data
    """
    file_path = {
        "Treatment": "./data/Treatment.csv",
        "Treatment_Pattern": "./data/Treatment_Pattern.csv",
        "Patient": "./data/Patient.csv",
        "Medical_resources": "./data/Medical_resources.csv"
    }

    # initialize the global_treatment
    # pattern_id : [treatment_1, treatment_2]
    with open(file_path["Treatment_Pattern"], mode='r') as file:
        reader = csv.reader(file)

        for row in reader: 
            pattern_id = row['pattern_id']
            global_treatment[pattern_id] = row[2:]

    # initialize the global_treatment_duration
    # treatment_id : duration
    with open(file_path["Treatment"], mode='r') as file:
        reader = csv.reader(file)

        for row in reader: 
            treatment_id = row['treatment_id']
            global_treatment_duration[treatment_id] = row[2]
            
    # initialize the global_patient
    with open(file_path["Patient"], mode='r') as file:
        reader = csv.reader(file)

        for row in reader:
            patient_id = row['p_id']
            arrival_time = row['arrival_time']
            acuity_level = row['acuity_level']
            treatment_plan_arr = global_treatment[row['pattern_id']]
            treatment_totaltime_arr = [global_treatment_duration[pattern_id] for pattern_id in treatment_plan_arr]
            global_patient[patient_id] = Patient(patient_id, arrival_time, acuity_level, treatment_plan_arr, treatment_totaltime_arr)


# def random_assign():

# def action():
#     action = {}
    
#     return state, new_state, reward, action


def init_simulation():
    
    #initialize all Resource object
    Medical_Resource_List = []
    for i in range(len(MEDICAL_RESOURCE_TYPE_ARR)):
        resource = Resource(MEDICAL_RESOURCE_TYPE_ARR[i], MEDICAL_RESOURCE_NUM_ARR[i])
        Medical_Resource_List.append(resource)

    #initialize all Patient object
    load_patient_data()

    global_medical_resource = Medical_Resource_List
    
def run_simulation():

    current_time = 0

    while current_time <= TOTAL_SIMULATION_TIME:
        # store old state
        
        current_time = current_time + 1
        # update patient in treatment, look at each resource to see if patient finish treatment
        patient_new = []
        for resource in global_medical_resource:
            patient_new.extend(resource.update_patient_in_treatment())

        # look at simulation data to see if we have new patient at current time, add to patient_new

        # assign patient to resources waiting que (according to mapping) if not reached end of treatment plan

        # update state 

        # if resource available and waiting line not empty then make an action for that resource
        for resource in global_medical_resource:
            if (resource.resource_is_available() == True) and (resource.waiting_is_empty == False):
                #take action for this resource
                patient_id = action(resource)

                resource.add_patient_to_treatment(patient_id, current_time)
                
            # calculate reward

            # store old state, new state, reward and action

     
# class CNN(nn.Module):
#     def __init__(self):
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.fc = nn.Linear(64 * 3 * 3)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 64 * 3 * 3)
#         x = self.fc(x)
#         return x