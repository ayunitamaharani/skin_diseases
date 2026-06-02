# Skin Disease EDA Dashboard

An interactive dashboard for performing Exploratory Data Analysis (EDA) on a skin disease metadata dataset. This dashboard is designed to explore disease distributions, patient demographic characteristics, dataset quality, and potential biases that may affect machine learning model performance.

## Live Demo

https://skindiseases-capstone-project.streamlit.app/

---

## Overview

Data analysis is a crucial step before building a skin disease classification model. This dashboard provides various interactive visualizations to understand dataset characteristics, evaluate data quality, and identify potential biases that may arise during the model training process.

The dataset consists of skin disease patient metadata, including information such as age, gender, symptom location, disease severity, and disease class.

The dashboard is built using Streamlit and Plotly, allowing users to dynamically explore data through interactive filters.

---

## Features

* **Interactive Filters** — Filter data by disease class, age range, gender, and severity level
* **Disease Class Distribution** — Visualization of the number of samples for each disease class
* **Demographic Analysis** — Age distribution and gender proportion of patients
* **Clinical Characteristics** — Symptom location heatmap and severity distribution by disease class
* **Dataset Quality Assessment** — Dataset statistics, class distribution, and age boxplots
* **Bias Risk Analysis** — Identification of classes that may introduce bias into machine learning models
* **Automated Insights** — Summary of key findings for each dashboard tab

---

## Dashboard Structure

### 1. Distribution & Demographics

Evaluates:

* Distribution of samples across disease classes
* Patient age distribution
* Gender proportions
* Average age per disease class
* Gender composition within each class

### 2. Clinical Characteristics

Analyzes:

* Distribution of symptom locations on the body
* Distribution of disease severity levels
* Relationship between disease classes and clinical characteristics

### 3. Dataset Quality

Displays:

* Data cleaning statistics
* Age distribution by disease class
* Class proportions after filtering
* Statistical summary for each disease class

### 4. Bias Risk Analysis

Measures potential bias risks based on:

* Class imbalance
* Dominance of specific severity levels
* Limited age diversity within classes

---

## Methods

| Method                              | Description                                                              |
| ----------------------------------- | ------------------------------------------------------------------------ |
| **Exploratory Data Analysis (EDA)** | Initial analysis to understand data patterns and characteristics         |
| **Entropy Analysis**                | Measures the diversity of severity level distributions within each class |
| **Risk Scoring**                    | Calculates bias risk scores based on multiple dataset factors            |
| **Class Imbalance Analysis**        | Identifies classes with imbalanced sample distributions                  |

---

## Dataset

Dataset used:

```text
dataset/
└── metadata_penyakit_kulit_cleaned.csv
```

Main variables:

| Variable     | Description                  |
| ------------ | ---------------------------- |
| id_pasien    | Unique patient ID            |
| disease_name | Skin disease name            |
| age          | Patient age                  |
| gender       | Patient gender               |
| body_part    | Symptom location on the body |
| severity     | Disease severity level       |

---

## Generated Insights

The dashboard helps identify:

* Underrepresented disease classes
* Patient age and gender distributions
* Symptom location patterns for each disease
* Disease severity distributions
* Potential bias caused by class imbalance
* Classes that may require data augmentation or class weighting during model training

---

## Technology Stack

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* SciPy

---

## Local Setup

```bash
git clone https://github.com/ayunitamaharani/skin_diseases.git

cd skin_diseases

pip install -r requirements.txt

streamlit run dashboard.py
```

---

## Project Structure

```text
skin_diseases/
│
├── dataset/
│   └── metadata_penyakit_kulit_cleaned.csv
│
├── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md
```

---
