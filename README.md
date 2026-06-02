# Skin Disease EDA Dashboard 

Dashboard interaktif untuk melakukan Exploratory Data Analysis (EDA) pada dataset metadata penyakit kulit. Dashboard ini dirancang untuk mengeksplorasi distribusi penyakit, karakteristik demografi pasien, kualitas dataset, serta potensi bias yang dapat memengaruhi performa model machine learning.

## Live Demo

[Tambahkan link deployment Streamlit Cloud di sini apabila tersedia.](https://skindiseases-capstone-project.streamlit.app/)

---

## Overview

Analisis data merupakan tahap penting sebelum membangun model klasifikasi penyakit kulit. Dashboard ini menyediakan berbagai visualisasi interaktif untuk memahami karakteristik dataset, mengevaluasi kualitas data, dan mengidentifikasi risiko bias yang dapat muncul selama proses pelatihan model.

Dataset yang digunakan terdiri dari metadata pasien penyakit kulit yang mencakup informasi usia, jenis kelamin, lokasi gejala, tingkat keparahan penyakit, dan kelas penyakit.

Dashboard dikembangkan menggunakan Streamlit dan Plotly sehingga memungkinkan eksplorasi data secara dinamis melalui filter interaktif.

---

## Fitur

- **Filter Interaktif** — Filter berdasarkan kelas penyakit, rentang usia, jenis kelamin, dan tingkat keparahan
- **Distribusi Kelas Penyakit** — Visualisasi jumlah data pada masing-masing kelas penyakit
- **Analisis Demografi** — Distribusi usia dan proporsi jenis kelamin pasien
- **Karakteristik Klinis** — Heatmap lokasi gejala dan tingkat keparahan berdasarkan kelas penyakit
- **Kualitas Dataset** — Ringkasan statistik dataset, distribusi kelas, dan boxplot usia
- **Analisis Risiko Bias** — Identifikasi kelas yang berpotensi menyebabkan bias pada model machine learning
- **Insight Otomatis** — Ringkasan hasil analisis pada setiap tab

---

## Struktur Dashboard

### 1. Distribusi & Demografi

Mengevaluasi:

- Distribusi jumlah data per kelas penyakit
- Distribusi usia pasien
- Proporsi jenis kelamin
- Rata-rata usia per kelas penyakit
- Komposisi gender pada setiap kelas

### 2. Karakteristik Klinis

Menganalisis:

- Distribusi lokasi gejala pada tubuh
- Distribusi tingkat keparahan penyakit
- Hubungan antara kelas penyakit dan karakteristik klinis

### 3. Kualitas Dataset

Menampilkan:

- Statistik pembersihan data
- Distribusi usia per kelas
- Proporsi kelas setelah filtering
- Ringkasan statistik setiap kelas penyakit

### 4. Risiko Bias

Mengukur risiko bias berdasarkan:

- Ketidakseimbangan jumlah data (class imbalance)
- Dominasi tingkat keparahan tertentu
- Rendahnya variasi usia pasien

---

## Metode

| Metode | Deskripsi |
|----------|-----------|
| **Exploratory Data Analysis (EDA)** | Analisis awal untuk memahami pola dan karakteristik data |
| **Entropy Analysis** | Mengukur keberagaman distribusi tingkat keparahan pada setiap kelas |
| **Risk Scoring** | Menghitung skor risiko bias berdasarkan beberapa faktor dataset |
| **Class Imbalance Analysis** | Mengidentifikasi kelas dengan jumlah data yang tidak seimbang |

---

## Dataset

Dataset yang digunakan:

```
dataset/
└── metadata_penyakit_kulit_cleaned.csv
```

Variabel utama yang digunakan:

| Variabel | Deskripsi |
|-----------|-----------|
| id_pasien | ID unik pasien |
| disease_name | Nama penyakit kulit |
| age | Usia pasien |
| gender | Jenis kelamin pasien |
| body_part | Lokasi gejala pada tubuh |
| severity | Tingkat keparahan penyakit |

---

## Insight yang Dihasilkan

Dashboard membantu mengidentifikasi:

- Kelas penyakit yang underrepresented
- Distribusi usia dan gender pasien
- Pola lokasi gejala pada setiap penyakit
- Distribusi tingkat keparahan penyakit
- Potensi bias akibat class imbalance
- Kelas yang membutuhkan augmentasi data atau class weighting saat training model

---

## Tech Stack yang Digunakan

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- SciPy

---

## Setup Lokal

```bash
git clone https://github.com/ayunitamaharani/skin_diseases.git

cd skin_diseases

pip install -r requirements.txt

streamlit run dashboard.py
```

---

## Struktur Project

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
