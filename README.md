# Image-Based Skin Disease Detection and Clinical Metadata Analysis

This project focuses on developing a data science-based solution to identify 15 types of skin diseases using image classification and patient clinical metadata analysis. The project aims to address class imbalance, minimize model bias, and provide an interactive dashboard for comprehensive data exploration.

## Background and Problem Statement
Skin diseases are a common health issue in society. Some of the main challenges faced include:
* Lack of public access to dermatologists, especially in certain regions, resulting in limited diagnostic processes.
* Difficulty in distinguishing types of skin diseases because the symptoms often share similarities, potentially leading to early misdiagnosis.
* Lack of public education regarding the early signs of skin diseases.
* Manual diagnosis processes by medical personnel are time-consuming.

Key Solution: Building an image-based skin disease detection system capable of rapidly and accurately identifying skin disease patterns, providing an initial diagnosis to assist the public in early detection.

## Dataset and Metadata Information

### 1. Image Data
The dataset used is sourced from Kaggle: Skin Disease Detection Dataset by the publisher mgmitesh.
* Total Classes: 15 skin disease classes (Acne, Actinic Keratosis, Basal Cell Carcinoma, Chickenpox, Dermato Fibroma, Dyshidrotic Eczema, Melanoma, Nail Fungus, Nevus, Normal Skin, Pigmented Benign Keratosis, Ringworm, Seborrheic Keratosis, Squamous Cell Carcinoma, Vascular Lesion).
* Initial Data Volume: 48,162 images (46,334 train data and 1,828 validation data in the original structure).
* Format: JPEG/JPG.

### 2. Clinical Metadata (metadata_penyakit_kulit.csv)
This metadata contains additional information from 2,300 patients with the following features:
* id_pasien: Unique patient ID.
* disease_name: Name of the skin disease (contains several writing inconsistencies such as melanomaa that need to be cleaned).
* age: Patient's age (contains 2.61% missing values).
* gender: Patient's gender.
* body_part: Location of the affected body part (contains 2.04% missing values).
* severity: Disease severity level (Mild, Moderate, Severe with 12.39% missing values).

## Interactive EDA Dashboard (Streamlit Application)
This project provides a dashboard.py file, which is an interactive web application built with Streamlit and Plotly to facilitate data visualization and risk analysis. The dashboard includes the following main features:
* Executive Summary: Displays key metrics of the image data and patient medical records in a concise single view.
* Demographics and Clinical Analysis: Interactive visualization of age distribution using Shannon entropy calculations, gender, infection location on body parts, and disease severity.
* Image Characteristics Analysis: Exploration of average color channel values (Red, Green, Blue) to identify visual similarities between disease classes.
* Risk and Bias Assessment: A specialized feature to identify disease classes most at risk of inducing bias in artificial intelligence models based on automated Risk Score calculations.

## Core Business Questions
This project answers 4 core questions through the Exploratory Data Analysis (EDA) process:
1. Class Distribution and Patient Demographics: Analyzing class imbalance across the 15 image classes as well as the age and gender distribution patterns of the patients.
2. Visual and Clinical Characteristics per Class: Comparing the average RGB color channel values of the images and the clinical distribution based on body_part and severity features.
3. Split Consistency and Data Cleanliness: Assessing the impact of the data deduplication process and the consistency of class proportions after dataset splitting using the stratified split technique.
4. Identification of Classes at Risk of Bias: Determining skin disease classes at risk of inducing model bias based on a combination of minimal data volume, high RGB feature overlap, and the dominance of a specific severity level.

## Workflow and Data Preprocessing
* Dataset Merging and Resplitting: Since the initial train and validation split was imbalanced, the data was recombined and resplit proportionally using a Stratified Split with a ratio of 70% Train / 15% Validation / 15% Test.
* Metadata Cleaning: Handling duplicate data (deduplication), correcting inconsistencies in disease names, and handling missing values in the age, body_part, and severity columns.

## Model Evaluation and A/B Testing Results
Before integration into the production system, an A/B Test was conducted on a sample size of 1,000 patients to compare the performance of the legacy model against the newly optimized model:

| Test Group / Model | Total Patients Evaluated | Correctly Diagnosed | Detection Rate |
| :--- | :---: | :---: | :---: |
| Model A (Legacy Baseline) | 500 | 357 | 71.40% |
| Model B (Optimized Pipeline) | 500 | 407 | 81.40% |

### Key Findings:
* Performance Gain: Model B (new) achieved a detection rate of 81.40%, representing an absolute 10 percentage point increase (or a 14.01% relative improvement) over Model A.
* Clinical Impact: Out of 500 patients, Model B accurately identified 50 additional disease cases that were missed by Model A.
* Statistical Significance: A Two-Proportion Z-Test yielded a p-value of 0.0002 (smaller than alpha = 0.05). This indicates that H0 is rejected and the performance improvement of Model B is statistically significant, rather than a result of random chance.

## Installation and Setup Guide
Prerequisites
Ensure your computer has Python 3.9+ and Git installed.

### Setup Steps Dashboard Streamlit:
1. Clone the Repository
  - https://github.com/ayunitamaharani/skin_diseases.git
  - cd skin_diseases
2. Create a Virtual Environment
  - python -m venv env
3. Activate the Virtual Environment
  - for windows : env\Scripts\activate
  - for Linux/Mac: source env/bin/activate
4. Install Library Dependencies
  - python -m pip install --upgrade pip
  - pip install -r requirements.txt
5. Run the Streamlit Dashboard
  - streamlit run dashboard.py

## LINK dashboard streamlit online: 
https://skindiseases-capstone-project.streamlit.app/

## Repository Structure
```text
├── dataset/
│   ├── metadata_penyakit_kulit.csv
│   ├── metadata_penyakit_kulit_cleaned.csv
│   ├── skin_disease.txt
│   └── split_skin_disease.txt
├── notebooks/
│   └── Skin_Diseases_Detection_Notebook.ipynb
├── dashboard.py
├── .gitignore
├── requirements.txt
└── README.md

