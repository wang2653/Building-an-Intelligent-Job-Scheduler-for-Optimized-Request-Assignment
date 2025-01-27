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
from queue import PriorityQueue
import ast

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
ACUITY_WEIGHT_MAP = {
    1: 30,  # Acuity level 1
    2: 15,  # Acuity level 2
    3: 1,  # Acuity level 3
    4: 1, # Acuity level 4
    5: 1  # Acuity level 5
}

TOTAL_SIMULATION_TIME = 1500 # minutes

# Patient object list 
# {patient_id:Patient, ...} 
global_patient = {}
# {arrival_time , Patient, ...} 
global_patient_by_time = PriorityQueue()
# pattern_id : [treatment_id, treatment_id, ...] --
global_treatment = {} 
# treatment_id : duration
# int : float --
global_treatment_duration = {}


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

def load_patient_data():

    file_path = "../data/NEWpatientdata7.csv"

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



def action_AA(resource):

    # select the patient with the highest acuity level
    if resource.waiting_is_empty():
        return None
    waiting_patients = resource.patientid_waiting_arr

    # sort patients by acuity level
    waiting_patients.sort(key=lambda pid: global_patient[pid].acuity_level)

    # patients with same acuity level
    count = 0
    for i in range(len(waiting_patients)):
        if global_patient[waiting_patients[0]].acuity_level == global_patient[waiting_patients[i]].acuity_level:
            count += 1
    waiting_patients = waiting_patients[:count]
   
    waiting_patients.sort(key=lambda pid: global_patient[pid].arrival_time)

    return waiting_patients[0]  # return the patient with the highest acuity and the earliest arrival time

# select the patient with the earliest arrival time
def action_FCFS(resource):
    
    if resource.waiting_is_empty():
        return None
    
    waiting_patients = resource.patientid_waiting_arr

    # sort patients by acuity level
    waiting_patients.sort(key=lambda pid: global_patient[pid].arrival_time)

    return waiting_patients[0]  # return the patient with the earliest arrival time

# select the patient with the shortest pattern remaining time
def action_SRPT(resource):
    if resource.waiting_is_empty():
        return None
    
    waiting_patients = resource.patientid_waiting_arr

    # sort patients by pattern remaining time
    waiting_patients.sort(key=lambda pid: global_patient[pid].treatment_remaining_time)

    return waiting_patients[0]  # return the patient with the shortest pattern remaining time

# select the patient with the linear combination of weighted acuity and SRPT
def action_AS(resource):
    if resource.waiting_is_empty():
        return None
    
    waiting_patients = resource.patientid_waiting_arr

    # sort patients by pattern remaining time - weighted acuity level
    waiting_patients.sort(key=lambda pid: (global_patient[pid].treatment_remaining_time - global_patient[pid].acuity_level * ACUITY_WEIGHT_MAP[global_patient[pid].acuity_level]))

    return waiting_patients[0]  # return the patient with the linear combination of weighted acuity and SRPT

# select the patient with the linear combination of weighted acuity and weighted waiting time
def action_AW(resource, current_time):
    if resource.waiting_is_empty():
        return None
    
    waiting_patients = resource.patientid_waiting_arr
    
    # waiting_time_so_far = current_time - global_patient[pid].arrival_time - (sum(global_patient[pid].treatment_totaltime_arr) - global_patient[pid].treatment_remaining_time)
    
    # sort patients descending by weighted acuity level + weighted waiting time so far
    waiting_patients.sort(key=lambda pid: (ACUITY_WEIGHT_MAP[global_patient[pid].acuity_level]) * (current_time - global_patient[pid].arrival_time - (sum(global_patient[pid].treatment_totaltime_arr) - global_patient[pid].treatment_remaining_time) + 1), reverse=True)

    return waiting_patients[0]  # return the patient with the linear combination of weighted acuity and weighted waiting time


def init_simulation():
    
    #initialize all Patient object
    load_patient_data()

    #initialize all Resource object
    for i in range(len(MEDICAL_RESOURCE_TYPE_ARR)):
        resource = Resource(MEDICAL_RESOURCE_TYPE_ARR[i], MEDICAL_RESOURCE_NUM_ARR[i])
        global_medical_resource.append(resource)

def evaluation():
    for key, value in global_patient.items():
        print(f"{key}: {value.arrival_time}")
        print(f"{key}: {value.treatment_starttime_arr}")
        print(f"{key}: {value.treatment_totaltime_arr}")

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
            
        print(f"{key}: {waiting_time} * {ACUITY_WEIGHT_MAP[acuity_level]}\n")
        global_weighted_waiting_time.append(ACUITY_WEIGHT_MAP[acuity_level] * waiting_time)

    print(f"weighted waiting time: {global_weighted_waiting_time}")
    print(f"sum of weighted waiting time: {sum(global_weighted_waiting_time)}")
    print(f"average of weighted waiting time: {sum(global_weighted_waiting_time) / len(global_weighted_waiting_time)}")
    
def run_simulation():

    init_simulation()
    current_time = 0

    while current_time <= TOTAL_SIMULATION_TIME:

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

        if patient_new:
            print(f'\n=== Time {current_time} ===')
            print(f'New patients are: {patient_new}')

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
                print(f'Patient {patient_id} assigned to resource {resource.medical_resource_type}')
                resource.add_patient_to_waiting(patient_id)

                # calculate the total treatment pattern remaining time
                remaining_time = 0
                for i in range(current_treatment_index, len(patient.treatment_plan_arr)):
                    remaining_time += patient.treatment_totaltime_arr[i]
                patient.treatment_remaining_time = remaining_time
                print(f'Patient {patient_id} remains {remaining_time}')


        # update state 

        # if resource available and waiting line not empty then make an action for that resource
        for resource in global_medical_resource:
            while (resource.resource_is_available() == True) and (resource.waiting_is_empty() == False):
                # 135.71
                # patient_id = action_FCFS(resource)
                # patient_id = action_SRPT(resource)
                # patient_id = action_AS(resource)
                # 51
                patient_id = action_AW(resource, current_time)
                # patient_id = action_AA(resource)
            
                if patient_id is not None:
                    resource.add_patient_to_treatment(patient_id, current_time)
                    print(f'Patient {patient_id} started treatment with {resource.medical_resource_type}')

            # calculate reward

            # store old state, new state, reward and action
        
        # check state
        # for resource in global_medical_resource:
        #     waiting_state = resource.get_waiting_state()
        #     treating_state = resource.get_inTreatment_state()
        #     print(f'\nResource: {resource.medical_resource_type}')
        #     print(f'  Waiting State: {waiting_state}')
        #     print(f'  Treating State: {treating_state}')


        current_time = current_time + 1

run_simulation()
evaluation()

def get_simulation_summary():
    summary = {
        "patients": [],
        "resources": [],
    }
    
    # Add patient details
    for patient_id, patient in global_patient.items():
        patient_info = {
            "id": patient_id,
            "arrival_time": patient.arrival_time,
            "acuity_level": patient.acuity_level,
            "treatment_plan": patient.treatment_plan_arr,
            "remaining_treatment_time": patient.treatment_remaining_time
        }
        summary["patients"].append(patient_info)
    
    # Add resource details
    for resource in global_medical_resource:
        resource_info = {
            "type": resource.medical_resource_type,
            "available_slots": resource.medical_resource_num - len(resource.patientid_in_treatment_arr),
            "patients_in_treatment": resource.patientid_in_treatment_arr,
            "patients_waiting": resource.patientid_waiting_arr,
        }
        summary["resources"].append(resource_info)
    
    return summary
