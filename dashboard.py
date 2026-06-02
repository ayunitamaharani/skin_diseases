import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import entropy

st.set_page_config(
    page_title="Skin Disease — EDA Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BURGUNDY   = "#6B2737"
MAROON     = "#4A1520"
WARM_RED   = "#8B3A4A"
CREAM      = "#F5EFE6"
CREAM_DARK = "#EDE4D8"
SAND       = "#D4C5B0"
BROWN_MID  = "#8B7355"
CHARCOAL   = "#2C1810"
MUTED_TEXT = "#7A6558"
WHITE      = "#FDFAF7"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');

  /* Lock to light mode */
  :root {{ color-scheme: only light !important; }}

  html, body {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: {CREAM} !important;
    color: {CHARCOAL} !important;
  }}

  .stApp {{ background-color: {CREAM} !important; }}

  /* Surgical text targets — tidak menyentuh SVG/canvas plotly */
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] strong,
  [data-testid="stMarkdownContainer"] em,
  .stMarkdown p {{
    color: {CHARCOAL} !important;
    font-family: 'DM Sans', sans-serif !important;
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background-color: {MAROON} !important;
    border-right: 1px solid {BURGUNDY} !important;
  }}
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: {SAND} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
  }}
  [data-testid="stSidebar"] [data-baseweb="tag"] {{
    background-color: {BURGUNDY} !important;
  }}

  /* Metric cards */
  [data-testid="metric-container"] {{
    background-color: {WHITE} !important;
    border: 1px solid {SAND} !important;
    border-radius: 8px !important;
    padding: 20px 24px !important;
  }}
  [data-testid="stMetricLabel"] p {{
    color: {MUTED_TEXT} !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
  }}
  [data-testid="stMetricValue"] {{
    color: {MAROON} !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.8rem !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: transparent !important;
    border-bottom: 1px solid {SAND} !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {MUTED_TEXT} !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
  }}
  .stTabs [aria-selected="true"] {{
    color: {BURGUNDY} !important;
    border-bottom: 2px solid {BURGUNDY} !important;
    background: transparent !important;
  }}

  /* Alert/info boxes */
  [data-testid="stAlert"] {{
    background-color: {WHITE} !important;
    border: 1px solid {SAND} !important;
    border-left: 3px solid {BURGUNDY} !important;
    border-radius: 6px !important;
  }}
  [data-testid="stAlert"] p {{
    color: {CHARCOAL} !important;
  }}

  /* Caption */
  [data-testid="stCaptionContainer"] p {{
    color: {MUTED_TEXT} !important;
    font-size: 0.8rem !important;
  }}

  /* Headings */
  h1, h2, h3 {{
    font-family: 'Playfair Display', serif !important;
    color: {MAROON} !important;
  }}

  hr {{ border-color: {SAND} !important; opacity: 0.5 !important; }}

  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {CREAM_DARK}; }}
  ::-webkit-scrollbar-thumb {{ background: {SAND}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=WHITE,
    font=dict(family="DM Sans", color=CHARCOAL, size=12),
    title_font=dict(family="Playfair Display", color=MAROON, size=15),
    margin=dict(t=48, b=32, l=16, r=16),
    xaxis=dict(gridcolor=CREAM_DARK, linecolor=SAND, tickcolor=SAND, tickfont=dict(color=CHARCOAL)),
    yaxis=dict(gridcolor=CREAM_DARK, linecolor=SAND, tickcolor=SAND, tickfont=dict(color=CHARCOAL)),
)

WARM_SCALE = [
    [0.0,  "#F5EFE6"],
    [0.25, "#D4C5B0"],
    [0.5,  "#8B7355"],
    [0.75, "#8B3A4A"],
    [1.0,  "#4A1520"],
]

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/metadata_penyakit_kulit_cleaned.csv")
    df["disease_name"] = df["disease_name"].str.strip().str.title()
    df["severity"] = df["severity"].fillna("Unknown")
    return df

df = load_data()

st.sidebar.markdown(f"""
<div style='padding: 8px 0 20px 0;'>
  <div style='font-family: Playfair Display, serif; font-size: 1.2rem; color: {CREAM}; letter-spacing: -0.01em;'>Skin Disease</div>
  <div style='font-size: 0.7rem; color: {SAND}; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px;'>EDA Dashboard</div>
</div>
""", unsafe_allow_html=True)

all_classes = sorted(df["disease_name"].unique())
selected_classes = st.sidebar.multiselect("Kelas Penyakit", options=all_classes, default=all_classes)
age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider("Rentang Usia", min_value=age_min, max_value=age_max, value=(age_min, age_max))
selected_gender = st.sidebar.multiselect("Jenis Kelamin", options=["Male", "Female"], default=["Male", "Female"])
selected_severity = st.sidebar.multiselect("Tingkat Keparahan", options=sorted(df["severity"].unique()), default=sorted(df["severity"].unique()))

df_f = df[
    (df["disease_name"].isin(selected_classes)) &
    (df["age"].between(age_range[0], age_range[1])) &
    (df["gender"].isin(selected_gender)) &
    (df["severity"].isin(selected_severity))
]

st.markdown(f"""
<div style='padding: 32px 0 8px 0; border-bottom: 1px solid {SAND}; margin-bottom: 28px;'>
  <div style='font-family: Playfair Display, serif; font-size: 2rem; color: {MAROON}; letter-spacing: -0.03em; line-height: 1.1;'>
    Analisis Eksploratif Data<br>Penyakit Kulit
  </div>
  <div style='font-size: 0.82rem; color: {MUTED_TEXT}; margin-top: 10px;'>
    Eksplorasi metadata 2.300 pasien mulai dari distribusi, karakteristik klinis, kualitas data, dan risiko bias model.
  </div>
</div>
""", unsafe_allow_html=True)

if len(df_f) == 0:
    st.warning("Tidak ada data yang sesuai filter.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Pasien", f"{len(df_f):,}")
c2.metric("Jumlah Kelas", df_f["disease_name"].nunique())
c3.metric("Rata-rata Usia", f"{df_f['age'].mean():.1f} thn")
c4.metric("Rasio M / F", f"{(df_f['gender']=='Male').sum()} / {(df_f['gender']=='Female').sum()}")

st.markdown("<div style='margin: 28px 0;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Distribusi & Demografi",
    "Karakteristik Klinis",
    "Kualitas Data",
    "Risiko Bias",
])

def apply_layout(fig, **kwargs):
    layout = {**PLOT_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig

# ── TAB 1 ────────────────────────────────────────────────────
with tab1:
    st.markdown(f"<div style='font-family: Playfair Display, serif; font-size: 1.2rem; color: {MAROON}; margin: 20px 0 4px;'>Distribusi Kelas & Demografi Pasien</div>", unsafe_allow_html=True)
    st.caption("Apakah distribusi 15 kelas penyakit kulit seimbang, dan bagaimana pola demografi pasiennya?")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    class_counts = df_f["disease_name"].value_counts().reset_index()
    class_counts.columns = ["Kelas", "Jumlah"]
    mean_count = class_counts["Jumlah"].mean()
    class_counts["Status"] = class_counts["Jumlah"].apply(lambda x: "Di bawah rata-rata" if x < mean_count else "Di atas rata-rata")

    fig1 = px.bar(class_counts.sort_values("Jumlah"), x="Jumlah", y="Kelas", color="Status",
                  orientation="h", text="Jumlah",
                  color_discrete_map={"Di bawah rata-rata": WARM_RED, "Di atas rata-rata": MAROON})
    fig1.add_vline(x=mean_count, line_dash="dot", line_color=SAND,
                   annotation_text=f"rata-rata {mean_count:.0f}", annotation_font_color=MUTED_TEXT)
    fig1.update_traces(textposition="outside", textfont_size=10)
    apply_layout(fig1, title="Jumlah Data per Kelas Penyakit", height=480, showlegend=True,
                 legend=dict(orientation="h", y=-0.12, font=dict(size=11, color=CHARCOAL)))
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_age = px.histogram(df_f, x="age", nbins=28, color_discrete_sequence=[BURGUNDY])
        fig_age.update_layout(bargap=0.06)
        apply_layout(fig_age, title="Distribusi Usia Pasien", xaxis_title="Usia (tahun)", yaxis_title="Jumlah")
        st.plotly_chart(fig_age, use_container_width=True)
    with col_b:
        gc = df_f["gender"].value_counts().reset_index()
        gc.columns = ["Gender", "Jumlah"]
        fig_g = px.pie(gc, names="Gender", values="Jumlah", hole=0.45,
                       color_discrete_sequence=[MAROON, SAND])
        fig_g.update_traces(textinfo="label+percent", textfont_size=12, textfont_color=WHITE)
        apply_layout(fig_g, title="Proporsi Jenis Kelamin", showlegend=False)
        st.plotly_chart(fig_g, use_container_width=True)

    avg_age = df_f.groupby("disease_name")["age"].mean().reset_index()
    avg_age.columns = ["Kelas", "Usia"]
    avg_age = avg_age.sort_values("Usia", ascending=False)
    fig_aa = px.bar(avg_age, x="Kelas", y="Usia", text=avg_age["Usia"].round(1),
                    color="Usia", color_continuous_scale=WARM_SCALE)
    fig_aa.update_traces(textposition="outside", textfont_size=10)
    apply_layout(fig_aa, title="Rata-rata Usia per Kelas", xaxis_tickangle=-40, height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_aa, use_container_width=True)

    gcp = df_f.groupby(["disease_name", "gender"]).size().unstack(fill_value=0)
    gcp = gcp.div(gcp.sum(axis=1), axis=0) * 100
    fig_gs = go.Figure()
    for g, col in zip(["Male", "Female"], [MAROON, SAND]):
        if g in gcp.columns:
            fig_gs.add_trace(go.Bar(name=g, x=gcp.index, y=gcp[g], marker_color=col))
    fig_gs.update_layout(barmode="stack", xaxis_tickangle=-40, yaxis_title="Persentase (%)", height=400)
    apply_layout(fig_gs, title="Proporsi Gender per Kelas (%)")
    st.plotly_chart(fig_gs, use_container_width=True)

    balance_ratio = class_counts["Jumlah"].min() / class_counts["Jumlah"].max()
    status = "BALANCED" if balance_ratio >= 0.8 else "CUKUP BALANCED" if balance_ratio >= 0.5 else "IMBALANCED"
    minority_n = len(class_counts[class_counts["Jumlah"] < mean_count])
    st.info(f"""
**Insight**

Balance ratio: {balance_ratio:.2f}, Status: {status}
{minority_n} kelas berada di bawah rata-rata dan berpotensi underrepresented.
Kelas paling sedikit: {class_counts.sort_values('Jumlah').iloc[0]['Kelas']} ({class_counts.sort_values('Jumlah').iloc[0]['Jumlah']} data).
Pola usia konsisten, lalu kelas kanker kulit (Melanoma, SCC) didominasi pasien lansia.
    """)

# ── TAB 2 ────────────────────────────────────────────────────
with tab2:
    st.markdown(f"<div style='font-family: Playfair Display, serif; font-size: 1.2rem; color: {MAROON}; margin: 20px 0 4px;'>Karakteristik Klinis per Kelas</div>", unsafe_allow_html=True)
    st.caption("Perbedaan distribusi lokasi tubuh dan tingkat keparahan antar kelas penyakit.")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        bp_m = df_f.groupby(["disease_name", "body_part"]).size().unstack(fill_value=0)
        bp_pct = bp_m.div(bp_m.sum(axis=1), axis=0) * 100
        fig_bp = px.imshow(bp_pct, color_continuous_scale=WARM_SCALE, text_auto=".0f", aspect="auto",
                           labels={"color": "% Pasien"})
        apply_layout(fig_bp, title="Lokasi Tubuh per Kelas (%)", height=520)
        st.plotly_chart(fig_bp, use_container_width=True)
    with col_b:
        sev_order = [s for s in ["Mild", "Moderate", "Severe", "Unknown"] if s in df_f["severity"].unique()]
        sev_m = df_f.groupby(["disease_name", "severity"]).size().unstack(fill_value=0).reindex(columns=sev_order, fill_value=0)
        sev_pct = sev_m.div(sev_m.sum(axis=1), axis=0) * 100
        fig_sev = px.imshow(sev_pct, color_continuous_scale=WARM_SCALE, text_auto=".0f", aspect="auto",
                            labels={"color": "% Pasien"})
        apply_layout(fig_sev, title="Tingkat Keparahan per Kelas (%)", height=520)
        st.plotly_chart(fig_sev, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        bp_ov = df_f["body_part"].value_counts().reset_index()
        bp_ov.columns = ["Bagian Tubuh", "Jumlah"]
        fig_bpb = px.bar(bp_ov, x="Bagian Tubuh", y="Jumlah", text="Jumlah",
                         color="Jumlah", color_continuous_scale=WARM_SCALE)
        fig_bpb.update_traces(textposition="outside", textfont_size=10)
        apply_layout(fig_bpb, title="Distribusi Lokasi Gejala", height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_bpb, use_container_width=True)
    with col_d:
        sv_ov = df_f["severity"].value_counts().reset_index()
        sv_ov.columns = ["Severity", "Jumlah"]
        fig_svb = px.bar(sv_ov, x="Severity", y="Jumlah", text="Jumlah", color="Severity",
                         color_discrete_map={"Mild": SAND, "Moderate": WARM_RED, "Severe": MAROON, "Unknown": BROWN_MID})
        fig_svb.update_traces(textposition="outside", textfont_size=10)
        apply_layout(fig_svb, title="Distribusi Tingkat Keparahan", height=360, showlegend=False)
        st.plotly_chart(fig_svb, use_container_width=True)

    st.info("""
**Insight**

Terdapat predileksi lokasi yang kuat per kelas, untuk kelas Acne dominan di Face/Back, Nail Fungus di Hands/Feet.
Kelas kanker (Melanoma, SCC, BCC) didominasi severity Moderate–Severe.
Kombinasi fitur visual dan klinis memiliki komplementaritas tinggi untuk pemodelan multimodal.
    """)

# ── TAB 3 ────────────────────────────────────────────────────
with tab3:
    st.markdown(f"<div style='font-family: Playfair Display, serif; font-size: 1.2rem; color: {MAROON}; margin: 20px 0 4px;'>Konsistensi & Kualitas Dataset</div>", unsafe_allow_html=True)
    st.caption("Ringkasan hasil deduplication, distribusi statistik usia, dan proporsi kelas.")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Data Bersih", f"{len(df):,}")
    m2.metric("Duplikat Dihapus", "248")
    m3.metric("Label Error", "94")
    m4.metric("Train-Val Leakage", "0")

    palette_15 = [MAROON, BURGUNDY, WARM_RED, BROWN_MID, SAND,
                  "#C4A882", "#9B6B5A", "#7A4F3A", "#5C3317", "#A0856C",
                  "#D4B896", "#6B4C3B", "#8C6B52", "#B8956A", "#E8D5B7"]

    fig_box = px.box(df_f, x="disease_name", y="age", color="disease_name",
                     color_discrete_sequence=palette_15)
    fig_box.update_layout(showlegend=False, xaxis_tickangle=-40)
    apply_layout(fig_box, title="Distribusi Usia per Kelas (Boxplot)", height=440)
    st.plotly_chart(fig_box, use_container_width=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        cp = df_f["disease_name"].value_counts(normalize=True).reset_index()
        cp.columns = ["Kelas", "Proporsi"]
        cp["Persen"] = (cp["Proporsi"] * 100).round(2)
        fig_pie = px.pie(cp, names="Kelas", values="Persen", hole=0.35,
                         color_discrete_sequence=palette_15)
        fig_pie.update_traces(textinfo="label+percent", textfont_size=10)
        apply_layout(fig_pie, title="Proporsi Kelas (setelah filter)", height=460, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        summary = df_f.groupby("disease_name").agg(
            Jumlah=("id_pasien", "count"),
            Usia_Min=("age", "min"),
            Usia_Max=("age", "max"),
            Usia_Rata=("age", "mean"),
        ).reset_index()
        summary["Usia_Rata"] = summary["Usia_Rata"].round(1)
        summary.columns = ["Kelas Penyakit", "Jumlah", "Usia Min", "Usia Maks", "Usia Rata-rata"]
        summary = summary.sort_values("Jumlah", ascending=False)
        st.markdown(f"<div style='font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; color:{MUTED_TEXT}; margin-bottom:10px;'>Ringkasan per Kelas</div>", unsafe_allow_html=True)
        st.dataframe(summary, use_container_width=True, hide_index=True, height=420)

    st.info("""
**Insight**

248 gambar duplikat dihapus (0.52% dari 48.000 data awal) yang berisi 154 exact copy dan 94 label error.
Tidak ditemukan train-val leakage. Stratified split 70/15/15 menghasilkan selisih proporsi < 1% per kelas.
    """)

# ── TAB 4 ────────────────────────────────────────────────────
with tab4:
    st.markdown(f"<div style='font-family: Playfair Display, serif; font-size: 1.2rem; color: {MAROON}; margin: 20px 0 4px;'>Identifikasi Kelas Berisiko Bias</div>", unsafe_allow_html=True)
    st.caption("Kelas mana yang paling berisiko menyebabkan bias berdasarkan imbalance data, dominasi severity, dan variasi usia.")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    class_cnt = df_f["disease_name"].value_counts()
    max_cnt = class_cnt.max()
    imbalance_risk = (1 - class_cnt / max_cnt)

    def calc_entropy(grp):
        counts = grp.value_counts(normalize=True)
        return entropy(counts)

    sev_ent = df_f.groupby("disease_name")["severity"].apply(calc_entropy)
    max_ent = sev_ent.max()
    severity_risk = (1 - sev_ent / max_ent) if max_ent > 0 else sev_ent * 0
    age_var = df_f.groupby("disease_name")["age"].std().fillna(0)
    max_var = age_var.max()
    age_risk = (1 - age_var / max_var) if max_var > 0 else age_var * 0

    df_risk = pd.concat([imbalance_risk, severity_risk, age_risk], axis=1).reset_index()
    df_risk.columns = ["Kelas", "Imbalance Data", "Dominasi Severity", "Variasi Usia Rendah"]
    df_risk["Risk Score"] = df_risk[["Imbalance Data", "Dominasi Severity", "Variasi Usia Rendah"]].mean(axis=1)
    df_risk = df_risk.sort_values("Risk Score", ascending=False).reset_index(drop=True)
    df_risk["Level"] = df_risk["Risk Score"].apply(lambda s: "Tinggi" if s >= 0.6 else "Sedang" if s >= 0.4 else "Rendah")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_r = px.bar(df_risk.sort_values("Risk Score"), x="Risk Score", y="Kelas", orientation="h",
                       color="Risk Score", color_continuous_scale=WARM_SCALE,
                       text=df_risk.sort_values("Risk Score")["Risk Score"].round(3))
        fig_r.add_vline(x=0.4, line_dash="dot", line_color=SAND,
                        annotation_text="sedang", annotation_font_color=MUTED_TEXT)
        fig_r.add_vline(x=0.6, line_dash="dot", line_color=WARM_RED,
                        annotation_text="tinggi", annotation_font_color=WARM_RED)
        fig_r.update_traces(textposition="outside", textfont_size=10)
        apply_layout(fig_r, title="Skor Risiko Bias per Kelas", height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_r, use_container_width=True)
    with col_b:
        hm = df_risk.set_index("Kelas")[["Imbalance Data", "Dominasi Severity", "Variasi Usia Rendah"]]
        fig_hm = px.imshow(hm, color_continuous_scale=WARM_SCALE, text_auto=".2f",
                           aspect="auto", zmin=0, zmax=1, labels={"color": "Skor (0–1)"})
        apply_layout(fig_hm, title="Breakdown Faktor Risiko", height=500)
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown(f"<div style='font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; color:{MUTED_TEXT}; margin: 16px 0 8px;'>Ranking Lengkap</div>", unsafe_allow_html=True)
    st.dataframe(df_risk[["Kelas", "Risk Score", "Imbalance Data", "Dominasi Severity", "Variasi Usia Rendah", "Level"]].round(3),
                 use_container_width=True, hide_index=True)

    top3 = df_risk.head(3)["Kelas"].tolist()
    st.info(f"""
**Insight**

Top 3 kelas paling berisiko: {', '.join(top3)}
Kelas dengan imbalance tinggi diprioritaskan untuk class weighting saat training.
Rekomendasi: terapkan augmentasi data pada kelas dengan risk score >= 0.4.
    """)

st.markdown(f"""
<div style='margin-top: 48px; padding-top: 20px; border-top: 1px solid {SAND};
     display: flex; justify-content: space-between; align-items: center;'>
  <span style='font-size: 0.72rem; color: {MUTED_TEXT}; letter-spacing: 0.06em; text-transform: uppercase;'>
    Skin Disease EDA Dashboard
  </span>
  <span style='font-size: 0.72rem; color: {SAND}; font-family: DM Mono, monospace;'>
    dataset: metadata_penyakit_kulit_cleaned.csv — 2.300 pasien, 15 kelas
  </span>
</div>
""", unsafe_allow_html=True)
