import numpy as np


def add_to_csv(patient_name, acuity_level):
    # implement your function here
    # this function is to receive two strings and add them to the .csv file
    # remember to also record the time as arriving time when calling this function
    return f"Added to dataset: {patient_name}, {acuity_level}"

def get_simulatio_result():
    # function to get the simulation result
    result = simulator.run_simulation()
    return result

def process_and_simulate(patient_name, acuity_level):
    # this function will be placed in UI.py
    # use this function to process inputs from doctor and display result from backend
    add_to_csv(patient_name, acuity_level)
    return get_simulatio_result()