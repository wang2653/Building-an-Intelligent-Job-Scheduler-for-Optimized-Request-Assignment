DROP SCHEMA IF EXISTS Emergency CASCADE;
CREATE SCHEMA Emergency;
SET SEARCH_PATH TO Emergency;


-- A patient arrives at date-arrival_Time with acuity will 
-- perform treatment_p_id.
CREATE TABLE Patients (
	patient_id INT PRIMARY KEY,
	date DATE NOT NULL
	arrival_time TIME NOT NULL,
	acuity INT NOT NULL,
	pattern_id INT NOT NULL
);


-- A treatement with name will need treatment_duration and 
-- resource_id.
CREATE TABLE Treatments (
	treatment_id INT PRIMARY KEY,
	name VARCHAR(30) NOT NULL,
	treatment_duration INT NOT NULL,
	resource_id INT NOT NULL
);


-- A treatement pattern has treatments_num treatments and 
-- in the sequence of treatments_sequence.
CREATE TABLE Treatment_patterns (
	pattern_id INT PRIMARY KEY,
	treatments_num INT NOT NULL,
	treatments_sequence VARCHAR(200) NOT NULL
);


-- We have resources_num medical resources with type available.
CREATE TABLE Medical_resources (
	resource_id INT PRIMARY KEY,
	type VARCHAR(200) NOT NULL,
	resources_num INT NOT NULL,
);

