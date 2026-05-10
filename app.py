import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DengueDx · Hematological Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #f4f7fb;
    color: #1a2e44;
}
.stApp { background-color: #f0f4fa; }

.header-bar {
    background: #ffffff;
    border-bottom: 2px solid #1565c0;
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.8rem;
    border-radius: 0 0 10px 10px;
}
.header-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: #1565c0;
    letter-spacing: -0.5px;
}
.header-logo span { color: #e53935; }
.header-sub {
    font-size: 0.78rem;
    color: #607d8b;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.header-badge {
    font-size: 0.7rem;
    background: #e3f2fd;
    color: #1565c0;
    border: 1px solid #90caf9;
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #1565c0;
    border-left: 3px solid #1565c0;
    padding-left: 8px;
    margin-bottom: 0.9rem;
}
.range-hint {
    font-size: 0.7rem;
    color: #90a4ae;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: -8px;
    margin-bottom: 6px;
}
.stNumberInput label, .stTextInput label, .stSelectbox label, .stSlider label {
    font-size: 0.82rem !important;
    color: #455a64 !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}
.stNumberInput input, .stTextInput input {
    background: #f8fafd !important;
    border: 1px solid #b0c8e8 !important;
    border-radius: 8px !important;
    color: #1a2e44 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stButton > button {
    background: #1565c0 !important;
    color: #ffffff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover { background: #0d47a1 !important; }

.result-positive {
    background: #fff5f5;
    border: 2px solid #e53935;
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    animation: fadeIn 0.4s ease;
}
.result-negative {
    background: #f1f8f4;
    border: 2px solid #2e7d32;
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

.result-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.result-desc { font-size: 0.85rem; color: #607d8b; }

.chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 5px 16px;
    border-radius: 20px;
    margin-top: 10px;
    text-transform: uppercase;
}
.chip-high     { background:#fde8e8; color:#c62828; border:1px solid #ef9a9a; }
.chip-moderate { background:#fff3e0; color:#e65100; border:1px solid #ffcc80; }
.chip-low      { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; }

.metric-row { display: flex; gap: 10px; margin: 1rem 0; }
.metric-card {
    flex: 1;
    background: #f0f6ff;
    border: 1px solid #dce8f5;
    border-radius: 10px;
    padding: 0.9rem 0.7rem;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #1565c0;
}
.metric-label { font-size: 0.7rem; color: #78909c; margin-top: 2px; letter-spacing: 0.5px; }

hr { border: none; border-top: 1px solid #dce8f5; margin: 1rem 0; }

.disclaimer {
    font-size: 0.72rem;
    color: #607d8b;
    background: #f8fafd;
    border: 1px solid #dce8f5;
    border-left: 3px solid #1565c0;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-top: 1rem;
    line-height: 1.6;
}

.cbc-table { width: 100%; font-size: 0.8rem; border-collapse: collapse; }
.cbc-table th {
    text-align: left;
    color: #78909c;
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1px;
    padding: 6px 8px;
    border-bottom: 1px solid #dce8f5;
}
.cbc-table td { padding: 7px 8px; color: #1a2e44; border-bottom: 1px solid #f0f4fa; }
.cbc-table tr:last-child td { border-bottom: none; }
.tag-normal { background:#e8f5e9; color:#2e7d32; border-radius:4px; padding:2px 8px; font-family:'IBM Plex Mono',monospace; font-size:0.68rem; }
.tag-low    { background:#fff3e0; color:#e65100; border-radius:4px; padding:2px 8px; font-family:'IBM Plex Mono',monospace; font-size:0.68rem; }
.tag-high   { background:#fde8e8; color:#c62828; border-radius:4px; padding:2px 8px; font-family:'IBM Plex Mono',monospace; font-size:0.68rem; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("dengue_simple_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
today = date.today().strftime("%d %b %Y")
st.markdown(f"""
<div class="header-bar">
    <div>
        <div class="header-logo">Dengue<span>Dx</span></div>
        <div class="header-sub">Hematological Dengue Screening System</div>
    </div>
    <div class="header-badge">🩺 &nbsp; ML · ExtraTrees · v1.0 &nbsp;·&nbsp; {today}</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model file `dengue_simple_model.pkl` not found. Place it in the same directory as this app.")
    st.stop()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="large")

# ══════════════════ LEFT ═════════════════════════════════════════════════════
with left:

    st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        patient_name = st.text_input("Full Name", placeholder="e.g. Priya Sharma")
    with c2:
        age = st.number_input("Age", min_value=1, max_value=120, value=28)
    with c3:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    c4, c5 = st.columns(2)
    with c4:
        fever_days = st.number_input("Fever Duration (days)", min_value=0, max_value=14, value=3)
    with c5:
        sample_id = st.text_input("Sample ID", placeholder="e.g. LAB-2024-001")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Complete Blood Count (CBC)</div>', unsafe_allow_html=True)

    c6, c7 = st.columns(2)
    with c6:
        platelet = st.number_input("Platelet Count (/cumm)", min_value=0.0, value=150000.0, step=1000.0)
        st.markdown('<div class="range-hint">Normal: 1,50,000 – 4,00,000</div>', unsafe_allow_html=True)
        neutrophils = st.number_input("Neutrophils (%)", min_value=0.0, max_value=100.0, value=60.0)
        st.markdown('<div class="range-hint">Normal: 50 – 70%</div>', unsafe_allow_html=True)
        hgb = st.number_input("Haemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=13.5)
        st.markdown('<div class="range-hint">Normal: 12 – 17 g/dL</div>', unsafe_allow_html=True)

    with c7:
        wbc = st.number_input("WBC Count (/cumm)", min_value=0.0, value=7000.0, step=100.0)
        st.markdown('<div class="range-hint">Normal: 4,000 – 11,000</div>', unsafe_allow_html=True)
        lymphocytes = st.number_input("Lymphocytes (%)", min_value=0.0, max_value=100.0, value=30.0)
        st.markdown('<div class="range-hint">Normal: 20 – 40%</div>', unsafe_allow_html=True)
        hct = st.number_input("Haematocrit / HCT (%)", min_value=0.0, max_value=100.0, value=40.0)
        st.markdown('<div class="range-hint">Normal: 36 – 50%</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    predict_btn = st.button("🔬  Run Dengue Screening")

# ══════════════════ RIGHT ════════════════════════════════════════════════════
with right:

    if predict_btn:
        input_data  = np.array([[platelet, wbc, neutrophils, lymphocytes]])
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        dengue_prob = probability[1] * 100
        normal_prob = probability[0] * 100

        if dengue_prob >= 70:
            risk, chip_cls = "High Risk", "chip-high"
        elif dengue_prob >= 40:
            risk, chip_cls = "Moderate Risk", "chip-moderate"
        else:
            risk, chip_cls = "Low Risk", "chip-low"

        name_display = patient_name if patient_name else "Patient"

        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div style="font-size:2.6rem; margin-bottom:6px;">⚠️</div>
                <div class="result-title" style="color:#c62828;">Dengue Positive</div>
                <div class="result-desc">{name_display} &nbsp;·&nbsp; Age {age} &nbsp;·&nbsp; {gender} &nbsp;·&nbsp; Day {fever_days} of fever</div>
                <span class="chip {chip_cls}">{risk}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div style="font-size:2.6rem; margin-bottom:6px;">✅</div>
                <div class="result-title" style="color:#2e7d32;">Non-Dengue</div>
                <div class="result-desc">{name_display} &nbsp;·&nbsp; Age {age} &nbsp;·&nbsp; {gender} &nbsp;·&nbsp; Day {fever_days} of fever</div>
                <span class="chip {chip_cls}">{risk}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{dengue_prob:.1f}%</div>
                <div class="metric-label">Dengue Probability</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{normal_prob:.1f}%</div>
                <div class="metric-label">Non-Dengue Prob.</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{fever_days}d</div>
                <div class="metric-label">Fever Duration</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # CBC summary table
        st.markdown('<div class="section-label" style="margin-top:1rem;">CBC Summary</div>', unsafe_allow_html=True)

        def status_tag(value, low, high):
            if value < low:    return '<span class="tag-low">Low ↓</span>'
            elif value > high: return '<span class="tag-high">High ↑</span>'
            else:              return '<span class="tag-normal">Normal</span>'

        rows = [
            ("Platelet Count", f"{platelet:,.0f} /cumm",  status_tag(platelet, 150000, 400000)),
            ("WBC Count",      f"{wbc:,.0f} /cumm",       status_tag(wbc, 4000, 11000)),
            ("Neutrophils",    f"{neutrophils:.1f}%",      status_tag(neutrophils, 50, 70)),
            ("Lymphocytes",    f"{lymphocytes:.1f}%",      status_tag(lymphocytes, 20, 40)),
            ("Haemoglobin",    f"{hgb:.1f} g/dL",         status_tag(hgb, 12, 17)),
            ("HCT",            f"{hct:.1f}%",              status_tag(hct, 36, 50)),
        ]
        table_rows = "".join(
            f"<tr><td>{r[0]}</td><td style='font-family:IBM Plex Mono,monospace'>{r[1]}</td><td>{r[2]}</td></tr>"
            for r in rows
        )
        st.markdown(f"""
        <table class="cbc-table">
            <thead><tr><th>Parameter</th><th>Value</th><th>Status</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>""", unsafe_allow_html=True)

        # Charts
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Probability & CBC Analysis</div>', unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
        fig.patch.set_facecolor("#ffffff")

        # Donut
        ax1.set_facecolor("#ffffff")
        ax1.pie([dengue_prob, normal_prob],
                colors=["#e53935", "#2e7d32"],
                startangle=90,
                wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2.5))
        ax1.text(0, 0.08, f"{dengue_prob:.0f}%", ha="center", va="center",
                 fontsize=20, fontweight="600", color="#1a2e44", fontfamily="monospace")
        ax1.text(0, -0.22, "dengue prob", ha="center",
                 fontsize=8, color="#90a4ae", fontfamily="monospace")
        ax1.set_title("Probability Split", color="#1565c0", fontsize=9,
                      fontfamily="monospace", pad=10, fontweight="600")
        ax1.legend(
            handles=[mpatches.Patch(color=c, label=l)
                     for c, l in zip(["#e53935","#2e7d32"], ["Dengue","Non-Dengue"])],
            loc="lower center", bbox_to_anchor=(0.5, -0.18),
            ncol=2, frameon=False, fontsize=8)

        # Bar chart
        ax2.set_facecolor("#f8fafd")
        params  = ["Platelets\n(÷1000)", "WBC\n(÷1000)", "Neutrophils\n(%)", "Lymphocytes\n(%)"]
        vals    = [platelet/1000, wbc/1000, neutrophils, lymphocytes]
        normals = [275, 7.5, 60, 30]
        clrs    = ["#e53935" if (v/n < 0.55 or v/n > 1.5) else "#1565c0"
                   for v, n in zip(vals, normals)]

        bars = ax2.barh(params, vals, color=clrs, height=0.45, edgecolor="white", linewidth=1.2)
        ax2.scatter(normals, params, color="#ff8f00", zorder=5, s=55, marker="D", label="Normal midpoint")
        ax2.set_xlabel("Value", color="#607d8b", fontsize=8, fontfamily="monospace")
        ax2.set_title("CBC vs Normal", color="#1565c0", fontsize=9,
                      fontfamily="monospace", pad=10, fontweight="600")
        ax2.tick_params(colors="#607d8b", labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color("#dce8f5")
        ax2.legend(loc="lower right", frameon=False, fontsize=7, labelcolor="#607d8b")
        for bar in bars:
            ax2.text(bar.get_width() + max(vals)*0.03,
                     bar.get_y() + bar.get_height()/2,
                     f"{bar.get_width():.1f}",
                     va="center", color="#455a64", fontsize=7.5, fontfamily="monospace")

        plt.tight_layout(pad=1.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class="disclaimer">
        ⚕️ <strong>Clinical Disclaimer:</strong> This tool is for educational and screening purposes only.
        Results must be interpreted by a qualified physician. Confirm with NS1 antigen, IgM/IgG serology,
        and full clinical evaluation before any diagnosis.
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            margin-top: 2rem;
            min-height: 480px;
            background: #ffffff;
            border: 1.5px dashed #b0c8e8;
            border-radius: 14px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #90a4ae;
            text-align: center;
            padding: 2rem;
        ">
            <div style="font-size:3.5rem; margin-bottom:16px;">🩸</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:#1565c0; margin-bottom:8px; font-weight:600;">
                Awaiting CBC Input
            </div>
            <div style="font-size:0.8rem; color:#90a4ae; max-width:260px; line-height:1.8;">
                Enter patient details and blood count values on the left,
                then click <strong style="color:#1565c0;">Run Dengue Screening</strong>
                to view the diagnostic report.
            </div>
        </div>
        """, unsafe_allow_html=True)