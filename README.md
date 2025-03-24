# 🏥Building an Intelligent Job Scheduler for Optimized Request Assignment in Hospital Emergency Department

Welcome to the Intelligent Job Scheduler project – an AI-driven solution designed to revolutionize patient triage and resource allocation in emergency departments (EDs).

---

## 📌Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Performance & Testing](#performance--testing)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)

---

## 🏥 Overview

The Intelligent Job Scheduler integrates a Deep Reinforcement Learning (DRL) model with a sleek, user-friendly web interface and an AI-powered chatbot to minimize patient waiting times and streamline ED operations. Designed to dynamically balance resource availability with patient needs, this system has demonstrated a significant reduction in weighted delays compared to traditional rule-based schedulers.

---

## 🚀 Features

- **Smart Scheduling:**  
  Utilizes a Deep Q-Network (DQN) to make real-time, data-driven decisions that optimize patient assignment based on acuity, arrival times, and resource constraints.

- **Intuitive Web Interface:**  
  A clean, responsive design built with HTML and CSS that allows doctors and patients to interact effortlessly. Features include:
  - A main website with interactive navigation.
  - A unified login page for both patients and doctors.
  - A dedicated doctor dashboard with real-time data updates.

- **AI-Powered Chatbot (MediGuide):**  
  Offers immediate, evidence-based responses to health inquiries, ensuring patients receive clear guidance and, when necessary, advice to consult healthcare professionals.

---

## 💻 Project Structure
📂 Building-an-Intelligent-Job-Scheduler-for-Optimized-Request-Assignment  
 ├── 📂 data/              # Simulated data  
 ├── 📂 src/  
   ├── 📂 simulator.py     # DRL scheduler & hospital simulation (Python)  
   ├── 📂 web_demo.py      # AI chatbot for patient assistance  
   ├── 📂 app.py           # Web interface (HTML, CSS, JavaScript)  
 ├── 📂 static/            # Model and UI testing scripts  
 ├── 📂 templates/         # Web sections  
 ├── 📜 README.md          # Project documentation  
 ├── 📜 requirements.txt   # Dependencies list  
 └── 📜 LICENSE            # Project license  

---

## ⚙️ Installation & Setup  

1️⃣ Clone the repository:  

'''bash
git clone https://github.com/wang2653/Building-an-Intelligent-Job-Scheduler-for-Optimized-Request-Assignment.git  
cd Building-an-Intelligent-Job-Scheduler-for-Optimized-Request-Assignment  
'''

2️⃣ Install dependencies:  

'''bash
pip install -r requirements.txt
'''

3️⃣ Run the frontend server:  

'''bash
cd src  
python app.py  
'''

4️⃣ Access the web app: Open http://localhost:8000 in your browser.

---

## System Block Diagram
<img src="https://github.com/user-attachments/assets/fc1b6349-249f-4e55-b936-fd3d01dad087" width="500">

---

## 📊 Performance & Testing

<img width="568" alt="image" src="https://github.com/user-attachments/assets/4429b672-a2f1-4141-94c8-8989ba432e4c" />

---

## 🤖 Future Enhancements
Future works will include the integration of multi-modal streams of data and testing advanced algorithmic strategies to enhance system flexibility and robustness. One such way is the use of electronic health records (EHRs), and continuous patient feedback to measure the full spectrum of drivers of patient flow.

---

## 👨‍⚕️ Contributors
- Yifan Wang: yfan.wang@mail.utoronto.ca
- Ruoyu Li: lambert.li@mail.utoronto.ca 
- Housen Zhu: benjamin.zhu@mail.utoronto.ca 
- Yuan Sui: yuan.sui@mail.utoronto.ca

---

## 📜 License

© 2025 University of Toronto
Edward S. Rogers Sr. Department of Electrical and Computer Engineering
ECE496 Final Project – Team 2024105. All rights reserved.
