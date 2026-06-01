import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import entropy

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Disease Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("metadata_penyakit_kulit_cleaned.csv")
    df["disease_name"] = df["disease_name"].str.strip().str.title()
    df["severity"] = df["severity"].fillna("Unknown")
    return df

df = load_data()

# ── Sidebar filters ─────────────────────────────────────────
st.sidebar.title("🔍 Filter Data")

all_classes = sorted(df["disease_name"].unique())
selected_classes = st.sidebar.multiselect(
    "Kelas Penyakit",
    options=all_classes,
    default=all_classes,
)

age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider(
    "Rentang Usia",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
)

selected_gender = st.sidebar.multiselect(
    "Jenis Kelamin",
    options=["Male", "Female"],
    default=["Male", "Female"],
)

selected_severity = st.sidebar.multiselect(
    "Tingkat Keparahan",
    options=sorted(df["severity"].unique()),
    default=sorted(df["severity"].unique()),
)

# ── Apply filters ───────────────────────────────────────────
df_filtered = df[
    (df["disease_name"].isin(selected_classes)) &
    (df["age"].between(age_range[0], age_range[1])) &
    (df["gender"].isin(selected_gender)) &
    (df["severity"].isin(selected_severity))
]

# ── Header ──────────────────────────────────────────────────
st.title("🔬 Dashboard EDA — Deteksi Penyakit Kulit")
st.markdown("Eksplorasi data metadata pasien penyakit kulit berdasarkan **4 pertanyaan bisnis**.")

if len(df_filtered) == 0:
    st.warning("Tidak ada data yang sesuai filter. Coba ubah filter di sidebar.")
    st.stop()

# ── Metrics ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pasien", f"{len(df_filtered):,}")
col2.metric("Jumlah Kelas", df_filtered["disease_name"].nunique())
col3.metric("Rata-rata Usia", f"{df_filtered['age'].mean():.1f} thn")
col4.metric("Rasio Gender M/F",
            f"{(df_filtered['gender']=='Male').sum()} / {(df_filtered['gender']=='Female').sum()}")

st.divider()

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 P1 — Distribusi & Demografi",
    "🏥 P2 — Karakteristik Klinis",
    "✅ P3 — Kualitas Data",
    "⚠️ P4 — Risiko Bias",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — Distribusi kelas & demografi
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("P1: Distribusi Kelas & Demografi Pasien")
    st.caption("Apakah distribusi jumlah data pada 15 kelas penyakit kulit seimbang, dan bagaimana pola demografi pasien?")

    # Distribusi kelas
    class_counts = df_filtered["disease_name"].value_counts().reset_index()
    class_counts.columns = ["Kelas", "Jumlah"]
    mean_count = class_counts["Jumlah"].mean()

    class_counts["Warna"] = class_counts["Jumlah"].apply(
        lambda x: "Di bawah rata-rata" if x < mean_count else "Di atas rata-rata"
    )

    fig_class = px.bar(
        class_counts.sort_values("Jumlah"),
        x="Jumlah", y="Kelas",
        color="Warna",
        color_discrete_map={"Di bawah rata-rata": "#e74c3c", "Di atas rata-rata": "#3498db"},
        orientation="h",
        title="Distribusi Jumlah Data per Kelas",
        text="Jumlah",
    )
    fig_class.add_vline(x=mean_count, line_dash="dash", line_color="gray",
                        annotation_text=f"Rata-rata ({mean_count:.0f})")
    fig_class.update_traces(textposition="outside")
    fig_class.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig_class, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Distribusi usia
        fig_age = px.histogram(
            df_filtered, x="age",
            nbins=30,
            color_discrete_sequence=["#2bc4c3"],
            title="Distribusi Usia Pasien",
            labels={"age": "Usia (tahun)", "count": "Jumlah"},
        )
        fig_age.update_layout(bargap=0.05)
        st.plotly_chart(fig_age, use_container_width=True)

    with col_b:
        # Distribusi gender
        gender_counts = df_filtered["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Jumlah"]
        fig_gender = px.pie(
            gender_counts, names="Gender", values="Jumlah",
            color_discrete_sequence=["#3498db", "#e74c3c"],
            title="Proporsi Jenis Kelamin",
            hole=0.4,
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    # Rata-rata usia per kelas
    avg_age = df_filtered.groupby("disease_name")["age"].mean().reset_index()
    avg_age.columns = ["Kelas", "Rata-rata Usia"]
    avg_age = avg_age.sort_values("Rata-rata Usia", ascending=False)

    fig_age_class = px.bar(
        avg_age, x="Kelas", y="Rata-rata Usia",
        color="Rata-rata Usia",
        color_continuous_scale="Blues",
        title="Rata-rata Usia Pasien per Kelas Penyakit",
        text=avg_age["Rata-rata Usia"].round(1),
    )
    fig_age_class.update_traces(textposition="outside")
    fig_age_class.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig_age_class, use_container_width=True)

    # Proporsi gender per kelas (stacked)
    gender_class = df_filtered.groupby(["disease_name", "gender"]).size().unstack(fill_value=0)
    gender_pct = gender_class.div(gender_class.sum(axis=1), axis=0) * 100
    gender_pct = gender_pct.reset_index()

    fig_gender_class = go.Figure()
    for g, color in zip(["Male", "Female"], ["#3498db", "#e74c3c"]):
        if g in gender_pct.columns:
            fig_gender_class.add_trace(go.Bar(
                name=g,
                x=gender_pct["disease_name"],
                y=gender_pct[g],
                marker_color=color,
            ))
    fig_gender_class.update_layout(
        barmode="stack",
        title="Proporsi Gender per Kelas (%)",
        xaxis_tickangle=-45,
        yaxis_title="Persentase (%)",
        height=420,
    )
    st.plotly_chart(fig_gender_class, use_container_width=True)

    # Insight box
    balance_ratio = class_counts["Jumlah"].min() / class_counts["Jumlah"].max()
    minority = class_counts[class_counts["Jumlah"] < mean_count]["Kelas"].tolist()
    status = "BALANCED" if balance_ratio >= 0.8 else "CUKUP BALANCED" if balance_ratio >= 0.5 else "IMBALANCED"

    st.info(f"""
**💡 Insight P1:**
- Balance ratio dataset: **{balance_ratio:.2f}** → Status: **{status}**
- **{len(minority)} kelas** berada di bawah rata-rata ({mean_count:.0f} data) dan berpotensi underrepresented
- Kelas paling sedikit: **{class_counts.iloc[-1]['Kelas']}** ({class_counts.iloc[-1]['Jumlah']} data)
- Pola usia konsisten: kelas kanker kulit (Melanoma, SCC) didominasi pasien lansia
    """)


# ─────────────────────────────────────────────────────────────
# TAB 2 — Karakteristik klinis
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("P2: Karakteristik Visual & Klinis per Kelas")
    st.caption("Apakah terdapat perbedaan karakteristik klinis (lokasi tubuh & tingkat keparahan) antar kelas?")

    col_a, col_b = st.columns(2)

    with col_a:
        # Heatmap body_part per kelas
        bp_matrix = df_filtered.groupby(["disease_name", "body_part"]).size().unstack(fill_value=0)
        bp_pct = bp_matrix.div(bp_matrix.sum(axis=1), axis=0) * 100

        fig_bp = px.imshow(
            bp_pct,
            color_continuous_scale="Blues",
            title="Distribusi Lokasi Tubuh per Kelas (%)",
            labels={"x": "Bagian Tubuh", "y": "Kelas Penyakit", "color": "% Pasien"},
            text_auto=".0f",
            aspect="auto",
        )
        fig_bp.update_layout(height=500)
        st.plotly_chart(fig_bp, use_container_width=True)

    with col_b:
        # Heatmap severity per kelas
        sev_order = ["Mild", "Moderate", "Severe", "Unknown"]
        sev_cols = [s for s in sev_order if s in df_filtered["severity"].unique()]

        sev_matrix = df_filtered.groupby(["disease_name", "severity"]).size().unstack(fill_value=0)
        sev_matrix = sev_matrix.reindex(columns=sev_cols, fill_value=0)
        sev_pct = sev_matrix.div(sev_matrix.sum(axis=1), axis=0) * 100

        fig_sev = px.imshow(
            sev_pct,
            color_continuous_scale="YlOrRd",
            title="Distribusi Tingkat Keparahan per Kelas (%)",
            labels={"x": "Severity", "y": "Kelas Penyakit", "color": "% Pasien"},
            text_auto=".0f",
            aspect="auto",
        )
        fig_sev.update_layout(height=500)
        st.plotly_chart(fig_sev, use_container_width=True)

    # Body part overall
    bp_overall = df_filtered["body_part"].value_counts().reset_index()
    bp_overall.columns = ["Bagian Tubuh", "Jumlah"]

    fig_bp_bar = px.bar(
        bp_overall, x="Bagian Tubuh", y="Jumlah",
        color="Jumlah",
        color_continuous_scale="Viridis",
        title="Distribusi Keseluruhan Lokasi Gejala pada Tubuh",
        text="Jumlah",
    )
    fig_bp_bar.update_traces(textposition="outside")
    fig_bp_bar.update_layout(height=380)
    st.plotly_chart(fig_bp_bar, use_container_width=True)

    # Severity overall
    sev_overall = df_filtered["severity"].value_counts().reset_index()
    sev_overall.columns = ["Severity", "Jumlah"]

    fig_sev_bar = px.bar(
        sev_overall, x="Severity", y="Jumlah",
        color="Severity",
        color_discrete_map={"Mild": "#2ecc71", "Moderate": "#f39c12",
                            "Severe": "#e74c3c", "Unknown": "#95a5a6"},
        title="Distribusi Tingkat Keparahan Keseluruhan",
        text="Jumlah",
    )
    fig_sev_bar.update_traces(textposition="outside")
    fig_sev_bar.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_sev_bar, use_container_width=True)

    st.info("""
**💡 Insight P2:**
- Terdapat predileksi lokasi yang kuat per kelas — Acne dominan di Face/Back, Nail Fungus di Hands/Feet
- Kelas kanker (Melanoma, SCC, BCC) didominasi severity Moderate–Severe
- Perbedaan karakteristik klinis antar kelas cukup signifikan → metadata berguna sebagai fitur tambahan model
    """)


# ─────────────────────────────────────────────────────────────
# TAB 3 — Kualitas data
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("P3: Konsistensi & Kualitas Dataset")
    st.caption("Seberapa bersih dataset ini dan bagaimana kualitas distribusinya?")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Data Bersih", f"{len(df):,}")
    col_b.metric("Duplikat Dihapus", "248", delta="-248", delta_color="inverse")
    col_c.metric("Missing Values Sisa", int(df.isnull().sum().sum()))

    st.markdown("#### Distribusi Statistik Usia per Kelas")

    fig_box = px.box(
        df_filtered, x="disease_name", y="age",
        color="disease_name",
        title="Boxplot Usia per Kelas Penyakit",
        labels={"disease_name": "Kelas", "age": "Usia"},
    )
    fig_box.update_layout(xaxis_tickangle=-45, height=450, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### Proporsi Kelas di Dataset")

    class_pct = df_filtered["disease_name"].value_counts(normalize=True).reset_index()
    class_pct.columns = ["Kelas", "Proporsi"]
    class_pct["Proporsi %"] = (class_pct["Proporsi"] * 100).round(2)

    fig_pie_class = px.pie(
        class_pct, names="Kelas", values="Proporsi %",
        title="Proporsi Setiap Kelas dalam Dataset (Setelah Filter)",
        hole=0.3,
    )
    fig_pie_class.update_traces(textinfo="label+percent")
    fig_pie_class.update_layout(height=500)
    st.plotly_chart(fig_pie_class, use_container_width=True)

    st.markdown("#### Tabel Ringkasan per Kelas")
    summary = df_filtered.groupby("disease_name").agg(
        Jumlah=("id_pasien", "count"),
        Usia_Min=("age", "min"),
        Usia_Max=("age", "max"),
        Usia_Mean=("age", "mean"),
    ).reset_index()
    summary["Usia_Mean"] = summary["Usia_Mean"].round(1)
    summary.columns = ["Kelas Penyakit", "Jumlah", "Usia Min", "Usia Maks", "Usia Rata-rata"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info("""
**💡 Insight P3:**
- Proses deduplication berhasil menghapus **248 gambar duplikat** (0.52% dari 48.000 data awal)
- Terdapat **94 label error** yang mengindikasikan inkonsistensi pelabelan di sumber dataset Kaggle
- Tidak ditemukan train-val leakage sama sekali
- Stratified split 70/15/15 menghasilkan proporsi kelas yang sangat konsisten (selisih < 1% per kelas)
    """)


# ─────────────────────────────────────────────────────────────
# TAB 4 — Risiko Bias
# ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("P4: Identifikasi Kelas Berisiko Bias")
    st.caption("Kelas mana yang paling berisiko menyebabkan bias pada model berdasarkan kombinasi faktor?")

    # Hitung risk score dari metadata
    class_cnt = df_filtered["disease_name"].value_counts()
    max_cnt = class_cnt.max()

    # Faktor 1: Imbalance risk
    imbalance_risk = (1 - class_cnt / max_cnt).rename("imbalance_risk")

    # Faktor 2: Severity entropy risk
    def calc_entropy(grp):
        counts = grp.value_counts(normalize=True)
        return entropy(counts)

    sev_ent = df_filtered.groupby("disease_name")["severity"].apply(calc_entropy)
    max_ent = sev_ent.max()
    severity_risk = (1 - sev_ent / max_ent).rename("severity_risk") if max_ent > 0 else sev_ent * 0

    # Faktor 3: Age variance (low variance = less diverse = risk)
    age_var = df_filtered.groupby("disease_name")["age"].std().fillna(0)
    max_var = age_var.max()
    age_risk = (1 - age_var / max_var).rename("age_risk") if max_var > 0 else age_var * 0

    df_risk = pd.concat([imbalance_risk, severity_risk, age_risk], axis=1).reset_index()
    df_risk.columns = ["Kelas", "Imbalance Data", "Dominasi Severity", "Variasi Usia Rendah"]
    df_risk["Risk Score"] = df_risk[["Imbalance Data", "Dominasi Severity", "Variasi Usia Rendah"]].mean(axis=1)
    df_risk = df_risk.sort_values("Risk Score", ascending=False).reset_index(drop=True)

    def risk_label(score):
        if score >= 0.6: return "🔴 Tinggi"
        elif score >= 0.4: return "🟡 Sedang"
        else: return "🟢 Rendah"

    df_risk["Level Risiko"] = df_risk["Risk Score"].apply(risk_label)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_risk = px.bar(
            df_risk.sort_values("Risk Score"),
            x="Risk Score", y="Kelas",
            color="Risk Score",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            orientation="h",
            title="Skor Risiko Bias per Kelas",
            text=df_risk.sort_values("Risk Score")["Risk Score"].round(3),
        )
        fig_risk.add_vline(x=0.4, line_dash="dash", line_color="#f39c12",
                           annotation_text="Risiko Sedang")
        fig_risk.add_vline(x=0.6, line_dash="dash", line_color="#e74c3c",
                           annotation_text="Risiko Tinggi")
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(height=500)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_b:
        heatmap_data = df_risk.set_index("Kelas")[["Imbalance Data", "Dominasi Severity", "Variasi Usia Rendah"]]
        fig_heat = px.imshow(
            heatmap_data,
            color_continuous_scale="YlOrRd",
            title="Breakdown Faktor Risiko per Kelas",
            labels={"color": "Skor (0–1)"},
            text_auto=".2f",
            aspect="auto",
            zmin=0, zmax=1,
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### 🏆 Ranking Risiko Bias")
    st.dataframe(
        df_risk[["Kelas", "Risk Score", "Imbalance Data", "Dominasi Severity",
                 "Variasi Usia Rendah", "Level Risiko"]].round(3),
        use_container_width=True,
        hide_index=True,
    )

    top3 = df_risk.head(3)["Kelas"].tolist()
    st.warning(f"""
**⚠️ Insight P4:**
- Top 3 kelas paling berisiko: **{', '.join(top3)}**
- Kelas dengan imbalance tinggi perlu diprioritaskan untuk **class weighting** saat training
- Kelas dengan dominasi severity tinggi perlu dicek apakah distribusinya mencerminkan kondisi klinis nyata
- Rekomendasi: terapkan augmentasi data pada kelas dengan risk score ≥ 0.4
    """)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("📊 Dashboard EDA — Deteksi Penyakit Kulit | Data: metadata_penyakit_kulit_cleaned.csv")
