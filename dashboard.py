import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Analisis Penyakit Kulit",
    page_icon="🩺",
    layout="wide"
)

# Judul Utama
st.title("Dashboard Monitoring Penyakit Kulit")
st.markdown("Dashboard ini digunakan untuk menganalisis persebaran penyakit berdasarkan demografi dan lokasi pada tubuh.")

# 2. Fungsi Load Data
@st.cache_data
def load_data():
    # Ganti dengan path file kamu jika perlu
    df = pd.read_csv("dataset/metadata_penyakit_kulit_cleaned.csv")
    # Penanganan Missing Value sederhana untuk visualisasi
    df['gender'] = df['gender'].fillna('Tidak Diketahui')
    df['body_part'] = df['body_part'].fillna('Lainnya')
    df['severity'] = df['severity'].fillna('Normal')
    return df

df = load_data()

# 3. Sidebar - Filter Interaktif
st.sidebar.header("🔍 Filter Data")
gender_filter = st.sidebar.multiselect(
    "Pilih Jenis Kelamin:",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

severity_filter = st.sidebar.multiselect(
    "Pilih Tingkat Keparahan (Severity):",
    options=df["severity"].unique(),
    default=df["severity"].unique()
)

# Terapkan Filter
df_filtered = df[(df["gender"].isin(gender_filter)) & (df["severity"].isin(severity_filter))]

# 4. Ringkasan Metrik (KPI)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pasien", len(df_filtered))
col2.metric("Rata-rata Umur", f"{int(df_filtered['age'].mean())} Tahun")
col3.metric("Jenis Penyakit", df_filtered['disease_name'].nunique())
col4.metric("Bagian Tubuh", df_filtered['body_part'].nunique())

st.divider()

# 5. Baris Pertama Visualisasi (Umur & Gender)
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("🎂 Distribusi Umur Pasien")
    fig_age = px.histogram(df_filtered, x="age", nbins=20, 
                           color_discrete_sequence=['#636EFA'],
                           labels={'age':'Umur', 'count':'Jumlah Pasien'},
                           template="plotly_white")
    st.plotly_chart(fig_age, use_container_width=True)

with row1_col2:
    st.subheader("🚻 Komposisi Jenis Kelamin")
    fig_gender = px.pie(df_filtered, names="gender", 
                        hole=0.4, 
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_gender, use_container_width=True)

# 6. Baris Kedua Visualisasi (Severity & Body Part)
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📉 Tingkat Keparahan (Severity)")
    fig_sev = px.bar(df_filtered['severity'].value_counts().reset_index(), 
                     x='severity', y='count',
                     color='severity',
                     labels={'count':'Jumlah', 'severity':'Tingkat Keparahan'},
                     template="plotly_white")
    st.plotly_chart(fig_sev, use_container_width=True)

with row2_col2:
    st.subheader("📍 Lokasi pada Tubuh (Body Part)")
    fig_body = px.bar(df_filtered['body_part'].value_counts().reset_index(), 
                      y='body_part', x='count', 
                      orientation='h',
                      color_discrete_sequence=['#00CC96'],
                      labels={'count':'Jumlah', 'body_part':'Bagian Tubuh'},
                      template="plotly_white")
    st.plotly_chart(fig_body, use_container_width=True)

# 7. Baris Ketiga - Top Disease
st.subheader("🦠 Top 10 Penyakit Terbanyak")
top_diseases = df_filtered['disease_name'].value_counts().nlargest(10).reset_index()
fig_dis = px.bar(top_diseases, x='count', y='disease_name', 
                 orientation='h', 
                 color='count',
                 color_continuous_scale='Viridis',
                 labels={'count':'Jumlah Pasien', 'disease_name':'Nama Penyakit'})
st.plotly_chart(fig_dis, use_container_width=True)

# 8. Tabel Data Terfilter
with st.expander("👁️ Lihat Detail Tabel Data"):
    st.dataframe(df_filtered, use_container_width=True)