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

'''Environment'''
MEDICAL_RESOURCE_TYPE_ARR = ["NURSE", "DOCTOR", "XRAY", "CT"]
MEDICAL_RESOURCE_NUM_ARR = [3, 3, 2, 1]  # Number of each resource type
MEDICAL_TREATMENT_TYPE_ARR = [1,2,3,4,5,6,7]
RESOURCE_TREATMENT_MAP = {
    1: 'NURSE',
    2: 'NURSE',
    3: 'DOCTOR',
    4: 'XRAY',
    5: 'CT',
    6: 'DOCTOR',
    7: 'NURSE'
}
ACUITY_WEIGHT_MAP = {
    1: 30,  # Acuity level 1
    2: 15,  # Acuity level 2
    3: 10,  # Acuity level 3
    4: 5, # Acuity level 4
    5: 1  # Acuity level 5
}

TOTAL_SIMULATION_TIME = 30 # minutes

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
        self.patient_id = patient_id                            # Unique identifier for the patient
        self.arrival_time = arrival_time                        # Time of arrival
        self.acuity_level = acuity_level                        # Acuity level (integer 1 to 5)
        self.treatment_plan_arr = treatment_plan_arr            # List of treatments plan (integer 1 to 7)
        self.treatment_totaltime_arr = treatment_totaltime_arr  # List of treatments time (simulated in minutes)
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
            return [0,0,0,0,0,0,0,0,0,0,0,0]
        # acuity distrbution
        acuity_counts = [0,0,0,0,0]
        for patient_id in self.patientid_waiting_arr:
            acuity_index = global_patient[patient_id].acuity_level - 1
            acuity_counts[acuity_index] += 1

        #treatment type distribution
        treatment_type_counts = [0,0,0,0,0,0,0]
        for patient_id in self.patientid_waiting_arr:
            current_treatment_index = global_patient[patient_id].current_treatment_index
            current_treatment_type = global_patient[patient_id].treatment_plan_arr[current_treatment_index]-1
            treatment_type_counts[current_treatment_type] += 1

        return [float(distribution / total_patient_waiting) for distribution in (acuity_counts + treatment_type_counts)]

    def get_inTreatment_state(self):
        total_patient_inTreatment = len(self.patientid_in_treatment_arr)
        if total_patient_inTreatment == 0:
            return [0,0,0,0,0,0,0,0,0,0,0,0]
        # acuity distrbution
        acuity_counts = [0,0,0,0,0]
        for patient_id in self.patientid_in_treatment_arr:
            acuity_index = global_patient[patient_id].acuity_level - 1
            acuity_counts[acuity_index] += 1

        #treatment type distribution
        treatment_type_counts = [0,0,0,0,0,0,0]
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

# Patient object list 
# {patient_id:Patient, ...} 
global_patient = {}
# {arrival_time , Patient, ...} 
global_patient_by_time = PriorityQueue()
# pattern_id : [treatment_id, treatment_id, ...] 
global_treatment = {} 
# treatment_id : duration
# int : float
global_treatment_duration = {}
# treatment_id : resource_id
# int : int
global_treatment_resource = {}

# run_simulation
global_medical_resource = []

def load_patient_data():
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

    with open('./data/Treatment_Pattern.csv', mode='r') as file:
        reader = csv.reader(file)

        for row in reader: 
            pattern_id = int(row[0])
            global_treatment[pattern_id] = list(map(int, row[2:]))
            

    # initialize the global_treatment_duration
    # treatment_id : duration
    with open(file_path["Treatment"], mode='r') as file:
        reader = csv.reader(file)

        for row in reader: 
            treatment_id = int(row[0])
            global_treatment_duration[treatment_id] = float(row[2])
            global_treatment_resource[treatment_id] = int(row[3])
            

    # initialize the global_patient
    with open(file_path["Patient"], mode='r') as file:
        reader = csv.reader(file)

        for row in reader:
            patient_id = int(row[0])
            arrival_time = int(row[2])
            acuity_level = int(row[3])
            treatment_plan_arr = list( map( int, global_treatment[int(row[4])] ) )
            treatment_totaltime_arr = [global_treatment_duration[treatment_id] for treatment_id in treatment_plan_arr]
            global_patient[patient_id] = Patient(patient_id, arrival_time, acuity_level, treatment_plan_arr, treatment_totaltime_arr)
            global_patient_by_time.put((arrival_time, patient_id))



def action(resource):
    """
    Selects the next patient to assign to a resource.

    Args:
        resource (Resource): The resource for which to select a patient.
    """
    # select the patient with the highest acuity level
    if resource.waiting_is_empty():
        return None
    waiting_patients = resource.patientid_waiting_arr
    print("waiting:", waiting_patients)
    # sort patients by acuity level
    waiting_patients.sort(key=lambda pid: global_patient[pid].acuity_level)
    print("most one:", waiting_patients[0])
    return waiting_patients[0]  # return the patient with the highest acuity

def init_simulation():
    
    #initialize all Patient object
    load_patient_data()

    #initialize all Resource object
    for i in range(len(MEDICAL_RESOURCE_TYPE_ARR)):
        resource = Resource(MEDICAL_RESOURCE_TYPE_ARR[i], MEDICAL_RESOURCE_NUM_ARR[i])
        global_medical_resource.append(resource)


def run_simulation():

    init_simulation()
    current_time = 0

    while current_time <= TOTAL_SIMULATION_TIME:
        # store old state
        print('\ncurrent_time: ', current_time)

        # update patient in treatment, look at each resource to see if patient finish treatment
        patient_new = []
        
        for resource in global_medical_resource:
            patient_new.extend(resource.update_patient_in_treatment())

        # look at simulation data to see if we have new patient at current time, add to patient_new
        while not global_patient_by_time.empty():
            patient = global_patient_by_time.get()
            print("Processing: (arrival_time, patient_id)")
            print(patient)

            if patient[0] == current_time:
                patient_new.append(patient[1])
            else:
                global_patient_by_time.put(patient)
                break

        print("new patients are:", patient_new)

        # assign patient to resources waiting que (according to mapping) if not reached end of treatment plan
        for patient_id in patient_new:
            # index of treatment_plan_arr
            current_treatment_index = global_patient[patient_id].current_treatment_index
            # not finish all treatments
            if (current_treatment_index < len(global_patient[patient_id].treatment_plan_arr)):
                # upcoming treatment id 
                current_treatment_id = global_patient[patient_id].treatment_plan_arr[current_treatment_index]
                # the resource id of the upcoming treatment
                resource_id = global_treatment_resource[current_treatment_id]
                print("treatment_id:", current_treatment_id, "resource_id:", resource_id)
                # assign patient to waiting queue of this resource
                global_medical_resource[resource_id-1].add_patient_to_waiting(patient_id)

        # check state
        for i in global_medical_resource:
            # print("distirbution of acuity(5) and treatment type(7)")
            print("waiting: ", i.medical_resource_type, ":", i.get_waiting_state())
            print("treating:", i.medical_resource_type, ":", i.get_inTreatment_state())

        # update state 

        # if resource available and waiting line not empty then make an action for that resource
        for resource in global_medical_resource:
            while (resource.resource_is_available() == True) and (resource.waiting_is_empty() == False):
                #take action for this resource
                patient_id = action(resource)

                resource.add_patient_to_treatment(patient_id, current_time)
                
            # calculate reward

            # store old state, new state, reward and action
        
        current_time = current_time + 1
            
run_simulation()