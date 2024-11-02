'''Environment'''
MEDICAL_RESOURCE_TYPE_ARR = ["DOCTOR", "NURSE", "XRAY", "CT"]
MEDICAL_RESOURCE_NUM_ARR = [4, 3, 2, 1]  # Number of each resource type
MEDICAL_TREATMENT_TYPE_ARR = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
RESOURCE_TREATMENT_MAP = {
    'A': 'DOCTOR',
    'B': 'NURSE',
    'C': 'XRAY',
    'D': 'DOCTOR',
    'E': 'NURSE',
    'F': 'DOCTOR',
    'G': 'DOCTOR'
}
ACUITY_WEIGHT_MAP = {
    1: 30,  # Acuity level 1
    2: 15,  # Acuity level 2
    3: 3,  # Acuity level 3
    4: 1, # Acuity level 4
    5: 1  # Acuity level 5
}

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
    def __init__(self, patient_id, arrival_time, acuity_level, treatment_plan_arr, treatment_time_arr, wait_time_arr=None):
        self.patient_id = patient_id              # Unique identifier for the patient
        self.arrival_time = arrival_time          # Time of arrival
        self.acuity_level = acuity_level          # Acuity level (integer 1 to 5)
        self.treatment_plan_arr = treatment_plan_arr# List of treatments plan (alphabet A to G)
        self.treatment_time_arr = treatment_time_arr# List of treatments time (simulated in minutes)
        self.wait_time_arr = wait_time_arr          # List of wait times for each treatment (to be recorded)


class Resource:
    

'''Algorithm Variables
state: 
action : 
reward :
state_new:
weight:
weight_old:
'''


def reward_function (action):

