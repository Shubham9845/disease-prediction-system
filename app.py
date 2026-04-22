import streamlit as st

st.set_page_config(
    page_title="BioPredict — Disease Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import io
import warnings
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.big-title {
    font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4aa 0%, #0099ff 50%, #7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 0.2rem;
}
.sub-title { color: #8899aa; font-size: 1.05rem; font-weight: 300; margin-bottom: 2rem; }
.badge {
    display: inline-block; background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.35); color: #00d4aa;
    padding: 3px 12px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 500; margin-right: 6px; margin-bottom: 6px;
}
.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700;
    color: #e2e8f0; border-left: 3px solid #00d4aa;
    padding-left: 14px; margin: 1.5rem 0 1rem 0;
}
.risk-high {
    background: linear-gradient(135deg, #2d0a0a, #1a0606);
    border: 1px solid #dc2626; border-radius: 12px;
    padding: 1rem 1.5rem; color: #fca5a5; font-size: 1.1rem; font-weight: 600;
}
.risk-medium {
    background: linear-gradient(135deg, #2d1a00, #1a1000);
    border: 1px solid #d97706; border-radius: 12px;
    padding: 1rem 1.5rem; color: #fcd34d; font-size: 1.1rem; font-weight: 600;
}
.risk-low {
    background: linear-gradient(135deg, #002d1a, #001a10);
    border: 1px solid #059669; border-radius: 12px;
    padding: 1rem 1.5rem; color: #6ee7b7; font-size: 1.1rem; font-weight: 600;
}
.stat-box {
    background: #0d1520; border: 1px solid #1e293b;
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.stat-number {
    font-family: 'Syne', sans-serif; font-size: 2rem;
    font-weight: 800; color: #00d4aa;
}
.stat-label {
    color: #64748b; font-size: 0.8rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 1px;
}
.disease-card-high {
    background: linear-gradient(135deg,#1a0606,#0d1520);
    border: 1px solid #1e293b; border-left: 3px solid #dc2626;
    border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
}
.disease-card-medium {
    background: linear-gradient(135deg,#1a1000,#0d1520);
    border: 1px solid #1e293b; border-left: 3px solid #d97706;
    border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
}
.disease-card-low {
    background: linear-gradient(135deg,#001a10,#0d1520);
    border: 1px solid #1e293b; border-left: 3px solid #059669;
    border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
}
.info-box {
    background: #0d1520; border: 1px solid #1e293b;
    border-radius: 10px; padding: 1rem 1.5rem; margin-bottom: 1rem;
}
.precaution-item {
    background: rgba(0,212,170,0.07); border-left: 2px solid #00d4aa;
    border-radius: 6px; padding: 0.5rem 1rem; margin-bottom: 0.4rem;
    color: #e2e8f0; font-size: 0.9rem;
}
div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; color: #00d4aa; }
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099ff);
    color: #080c14; font-weight: 700; border: none; border-radius: 10px;
    padding: 0.6rem 2rem; font-family: 'Syne', sans-serif; font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────
conn = sqlite3.connect("biopredict.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, age INTEGER, gender TEXT,
    symptoms TEXT, predicted_disease TEXT,
    confidence REAL, risk_level TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
conn.commit()

# ─────────────────────────────────────────────
#  RISK CLASSIFICATION
# ─────────────────────────────────────────────
HIGH_RISK = [
    "AIDS", "Tuberculosis", "Malaria", "Dengue", "Typhoid",
    "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
    "Hepatitis A", "Chronic cholestasis", "Alcoholic hepatitis",
    "Jaundice", "Dimorphic hemmorhoids(piles)", "Heart attack",
    "Varicose veins", "Hypertension"
]
LOW_RISK = [
    "Common Cold", "Chicken pox", "Fungal infection",
    "Allergy", "Drug Reaction", "Acne", "Urinary tract infection"
]

def get_risk(disease):
    if disease in HIGH_RISK:   return "High"
    if disease in LOW_RISK:    return "Low"
    return "Medium"

# ─────────────────────────────────────────────
#  LOAD KAGGLE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    dataset    = pd.read_csv("dataset.csv")
    desc       = pd.read_csv("symptom_Description.csv")
    precaution = pd.read_csv("symptom_precaution.csv")
    severity   = pd.read_csv("Symptom-severity.csv")

    for df in [dataset, desc, precaution, severity]:
        df.columns = df.columns.str.strip()

    dataset = dataset.fillna("")
    sym_cols = [col for col in dataset.columns if "Symptom" in col]
    for col in sym_cols:
        dataset[col] = dataset[col].str.strip().str.replace(" ", "_").str.lower()

    all_syms = set()
    for col in sym_cols:
        all_syms.update(dataset[col].unique())
    all_syms.discard("")
    all_syms = sorted(list(all_syms))

    X, y = [], []
    for _, row in dataset.iterrows():
        disease = row["Disease"].strip()
        vec = [0] * len(all_syms)
        for col in sym_cols:
            s = row[col].strip()
            if s and s in all_syms:
                vec[all_syms.index(s)] = 1
        X.append(vec)
        y.append(disease)

    X = np.array(X)
    y = np.array(y)
    diseases = sorted(list(set(y)))
    return X, y, all_syms, diseases, dataset, desc, precaution, severity, sym_cols

@st.cache_resource
def train_model(_X, _y, _syms, _diseases):
    X_tr, X_te, y_tr, y_te = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_split=4, random_state=42, class_weight="balanced"
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True)
    cv     = cross_val_score(clf, _X, _y, cv=5)
    cm_mat = confusion_matrix(y_te, y_pred, labels=_diseases)
    fi     = dict(zip(_syms, clf.feature_importances_))
    return clf, acc, report, cv, cm_mat, fi, X_te, y_te, y_pred

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:2.5rem'>🧬</div>
        <div style='font-family:Syne,sans-serif;font-weight:800;color:#00d4aa;font-size:1.2rem;'>BioPredict</div>
        <div style='color:#64748b;font-size:0.75rem;'>Computational Disease Analysis</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Home", "🔬 Disease Predictor",
        "📊 EDA Dashboard", "🤖 ML Model Insights", "📋 Patient Records",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b;font-size:0.78rem;line-height:1.8;'>
    <b style='color:#94a3b8;'>Tech Stack</b><br>
    Python · Streamlit · Scikit-learn<br>
    Pandas · NumPy · Plotly<br>
    SQLite · Matplotlib<br><br>
    <b style='color:#94a3b8;'>Dataset</b><br>
    Kaggle Real Medical Data<br>
    4920 Records · 41 Diseases<br>
    131 Real Symptoms<br><br>
    <b style='color:#94a3b8;'>Developer</b><br>
    Shubham Kumar<br>
    B.Tech Biotechnology
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD + TRAIN
# ─────────────────────────────────────────────
with st.spinner("🧬 Loading real Kaggle medical data..."):
    X, y, all_symptoms, disease_list, dataset, desc_df, prec_df, sev_df, sym_cols = load_and_prepare()

with st.spinner("🤖 Training Random Forest on real data..."):
    model, accuracy, report, cv_scores, cm, feat_imp, Xt, yt, yp = train_model(X, y, all_symptoms, disease_list)

SYM_DISPLAY = {s: s.replace("_", " ").title() for s in all_symptoms}

# ══════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="big-title">BioPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Computational Disease Prediction & Healthcare Analytics System</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="badge">🧬 Biotechnology</span>
    <span class="badge">🤖 Machine Learning</span>
    <span class="badge">📊 Data Analytics</span>
    <span class="badge">🌐 Web Application</span>
    <span class="badge">🔬 Computational Biology</span>
    """, unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-box"><div class="stat-number">{len(disease_list)}</div><div class="stat-label">Diseases Covered</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-box"><div class="stat-number">{len(all_symptoms)}</div><div class="stat-label">Symptoms Tracked</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-box"><div class="stat-number">{round(accuracy*100,1)}%</div><div class="stat-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="stat-box"><div class="stat-number">{len(dataset)}</div><div class="stat-label">Real Records</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🏥 Diseases Covered</div>', unsafe_allow_html=True)

    leg1, leg2, leg3, leg4 = st.columns(4)
    with leg2: st.markdown('<span style="color:#fca5a5;">🔴 High Risk</span>', unsafe_allow_html=True)
    with leg3: st.markdown('<span style="color:#fcd34d;">🟡 Medium Risk</span>', unsafe_allow_html=True)
    with leg4: st.markdown('<span style="color:#6ee7b7;">🟢 Low Risk</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    d_cols = st.columns(4)
    for i, disease in enumerate(disease_list):
        risk = get_risk(disease)
        card_class = "disease-card-high" if risk=="High" else "disease-card-low" if risk=="Low" else "disease-card-medium"
        risk_color = "#fca5a5" if risk=="High" else "#6ee7b7" if risk=="Low" else "#fcd34d"
        risk_icon  = "🔴" if risk=="High" else "🟢" if risk=="Low" else "🟡"
        with d_cols[i % 4]:
            st.markdown(f"""
            <div class="{card_class}">
                <div style='color:#e2e8f0;font-weight:600;font-size:0.85rem;'>{disease}</div>
                <div style='color:{risk_color};font-size:0.75rem;margin-top:2px;'>{risk_icon} {risk} Risk</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📋 How It Works</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    steps = [
        ("🩺","Step 1","Select symptoms from 131 real medical symptoms"),
        ("🤖","Step 2","AI model analyzes your symptom pattern"),
        ("🎯","Step 3","Get top 3 disease predictions with confidence %"),
        ("💊","Step 4","Read disease description and medical precautions"),
    ]
    for col, (icon, title, desc) in zip([h1,h2,h3,h4], steps):
        with col:
            st.markdown(f"""
            <div class="info-box" style="text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;color:#00d4aa;font-weight:700;">{title}</div>
                <div style="color:#94a3b8;font-size:0.85rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  DISEASE PREDICTOR
# ══════════════════════════════════════════════
elif page == "🔬 Disease Predictor":
    st.markdown('<div class="big-title" style="font-size:2rem;">🔬 Disease Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">ML-powered prediction using real Kaggle medical dataset — 41 diseases · 131 symptoms</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1: patient_name   = st.text_input("Patient Name", placeholder="Enter name...")
    with p2: patient_age    = st.number_input("Age", min_value=1, max_value=120, value=25)
    with p3: patient_gender = st.selectbox("Gender", ["Male","Female","Other"])

    st.markdown("---")
    st.markdown('<div class="section-title">🩺 Select Symptoms</div>', unsafe_allow_html=True)
    st.info(f"💡 {len(all_symptoms)} real medical symptoms available. Use the search box to find symptoms faster!")

    search = st.text_input("🔍 Search symptoms...", placeholder="e.g. fever, cough, headache...")
    filtered = [s for s in all_symptoms if search.lower().replace(" ","_") in s or search.lower() in s.replace("_"," ")] if search else all_symptoms

    selected = []
    cols_n = 4
    rows   = [filtered[i:i+cols_n] for i in range(0, len(filtered), cols_n)]
    for ri, row_syms in enumerate(rows):
        cols = st.columns(cols_n)
        for ci, sym in enumerate(row_syms):
            with cols[ci]:
                if st.checkbox(SYM_DISPLAY.get(sym, sym), key=f"sym_{ri}_{ci}"):
                    selected.append(sym)

    st.markdown("---")
    if selected:
        st.markdown(f"**✅ {len(selected)} symptom(s) selected:** " +
                    " · ".join([f"`{SYM_DISPLAY.get(s,s)}`" for s in selected]))

    if st.button("🔬 Predict Disease", use_container_width=True):
        if len(selected) < 2:
            st.error("⚠️ Please select at least 2 symptoms for accurate prediction!")
        else:
            with st.spinner("🧠 Analyzing symptoms with Random Forest model..."):
                vec = [0]*len(all_symptoms)
                for s in selected:
                    if s in all_symptoms:
                        vec[all_symptoms.index(s)] = 1
                proba    = model.predict_proba([vec])[0]
                top3_idx = np.argsort(proba)[::-1][:3]
                top3     = [(model.classes_[i], round(proba[i]*100,1)) for i in top3_idx]

            top_disease, top_conf = top3[0]
            risk       = get_risk(top_disease)
            risk_class = "risk-high" if risk=="High" else "risk-medium" if risk=="Medium" else "risk-low"
            risk_icon  = "🔴" if risk=="High" else "🟡" if risk=="Medium" else "🟢"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 Prediction Results</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="{risk_class}">
                🎯 &nbsp;<strong>{top_disease}</strong> &nbsp;|&nbsp;
                Confidence: <strong>{top_conf}%</strong> &nbsp;|&nbsp;
                Risk Level: {risk_icon} <strong>{risk}</strong>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                try:
                    d_row = desc_df[desc_df["Disease"].str.strip()==top_disease]
                    if not d_row.empty:
                        st.markdown('<div class="section-title" style="font-size:1rem;">📌 About this Disease</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="info-box"><div style="color:#94a3b8;font-size:0.9rem;line-height:1.6;">{d_row.iloc[0]["Description"]}</div></div>', unsafe_allow_html=True)
                except: pass
                try:
                    p_row = prec_df[prec_df["Disease"].str.strip()==top_disease]
                    if not p_row.empty:
                        st.markdown('<div class="section-title" style="font-size:1rem;">🚨 Medical Precautions</div>', unsafe_allow_html=True)
                        for pc in [col for col in prec_df.columns if "Precaution" in col]:
                            val = p_row.iloc[0][pc]
                            if pd.notna(val) and str(val).strip():
                                st.markdown(f'<div class="precaution-item">✅ {str(val).strip().capitalize()}</div>', unsafe_allow_html=True)
                except: pass

            with col2:
                fig = go.Figure(go.Bar(
                    x=[t[1] for t in top3], y=[t[0] for t in top3],
                    orientation="h",
                    marker_color=["#00d4aa","#0099ff","#7c3aed"],
                    text=[f"{t[1]}%" for t in top3], textposition="outside"
                ))
                fig.update_layout(
                    title="🏆 Top 3 Predicted Diseases",
                    xaxis_title="Confidence (%)",
                    plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                    font_color="#e2e8f0", height=280,
                    margin=dict(l=10,r=70,t=50,b=30),
                    xaxis=dict(range=[0,120], gridcolor="#1e293b"),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)

                # Severity of selected symptoms
                try:
                    sev_copy = sev_df.copy()
                    sev_copy["Symptom"] = sev_copy["Symptom"].str.strip().str.replace(" ","_").str.lower()
                    matched_sev = sev_copy[sev_copy["Symptom"].isin(selected)]
                    if not matched_sev.empty:
                        matched_sev = matched_sev.sort_values("weight", ascending=False)
                        matched_sev["Symptom"] = matched_sev["Symptom"].str.replace("_"," ").str.title()
                        fig_sev = px.bar(matched_sev, x="weight", y="Symptom",
                                         orientation="h", color="weight",
                                         color_continuous_scale="Reds", template="plotly_dark",
                                         title="⚠️ Selected Symptom Severity")
                        fig_sev.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                                              height=280, coloraxis_showscale=False,
                                              margin=dict(l=10,r=30,t=50,b=30),
                                              yaxis=dict(categoryorder="total ascending"))
                        st.plotly_chart(fig_sev, use_container_width=True)
                except: pass

            st.markdown("---")
            st.markdown('<div class="section-title" style="font-size:1rem;">🔍 Symptom Match Analysis</div>', unsafe_allow_html=True)
            disease_rows = dataset[dataset["Disease"].str.strip()==top_disease]
            if not disease_rows.empty:
                disease_syms = set()
                for col in sym_cols:
                    disease_syms.update(disease_rows[col].str.strip().tolist())
                disease_syms.discard("")
                matched_s   = [s for s in selected if s in disease_syms]
                unmatched_s = [s for s in selected if s not in disease_syms]
                ma1, ma2 = st.columns(2)
                with ma1:
                    st.markdown(f"**✅ Matching symptoms ({len(matched_s)}):**")
                    for s in matched_s:
                        st.markdown(f'<div class="precaution-item">✔️ {SYM_DISPLAY.get(s,s)}</div>', unsafe_allow_html=True)
                with ma2:
                    if unmatched_s:
                        st.markdown(f"**⚠️ Non-specific symptoms ({len(unmatched_s)}):**")
                        for s in unmatched_s:
                            st.markdown(f"<div style='color:#64748b;padding:4px 0;'>• {SYM_DISPLAY.get(s,s)}</div>", unsafe_allow_html=True)

            if patient_name.strip():
                c.execute(
                    "INSERT INTO predictions (name,age,gender,symptoms,predicted_disease,confidence,risk_level) VALUES (?,?,?,?,?,?,?)",
                    (patient_name, patient_age, patient_gender,
                     ", ".join(selected), top_disease, top_conf, risk)
                )
                conn.commit()
                st.success(f"✅ Prediction saved for **{patient_name}**!")

            st.warning("⚕️ **Disclaimer:** This tool is for educational purposes only. Always consult a qualified medical professional.")

# ══════════════════════════════════════════════
#  EDA DASHBOARD
# ══════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.markdown('<div class="big-title" style="font-size:2rem;">📊 EDA Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Exploratory Data Analysis on Real Kaggle Medical Dataset — 4920 Records</div>', unsafe_allow_html=True)
    st.markdown("---")

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Records",   len(dataset))
    k2.metric("Unique Diseases", dataset["Disease"].nunique())
    k3.metric("Total Symptoms",  len(all_symptoms))
    k4.metric("Model Accuracy",  f"{round(accuracy*100,1)}%")
    st.markdown("---")

    # Disease distribution
    st.markdown('<div class="section-title">🦠 Disease Distribution</div>', unsafe_allow_html=True)
    dc = dataset["Disease"].str.strip().value_counts().reset_index()
    dc.columns = ["Disease","Count"]
    fig1 = px.bar(dc, x="Count", y="Disease", orientation="h",
                  color="Count", color_continuous_scale="teal", template="plotly_dark",
                  title="Number of Records per Disease")
    fig1.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                       height=750, coloraxis_showscale=False,
                       yaxis=dict(categoryorder="total ascending"),
                       margin=dict(l=10,r=30,t=50,b=30))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("---")

    # Top 20 symptoms
    st.markdown('<div class="section-title">🔑 Top 20 Most Common Symptoms</div>', unsafe_allow_html=True)
    sym_list = []
    for col in sym_cols:
        sym_list.extend(dataset[col].dropna().str.strip().tolist())
    sym_list  = [s for s in sym_list if s]
    sym_counts = Counter(sym_list).most_common(20)
    sym_df = pd.DataFrame(sym_counts, columns=["Symptom","Count"])
    sym_df["Symptom"] = sym_df["Symptom"].str.replace("_"," ").str.title()
    fig2 = px.bar(sym_df, x="Count", y="Symptom", orientation="h",
                  color="Count", color_continuous_scale="Blues", template="plotly_dark",
                  title="Top 20 Symptoms by Frequency")
    fig2.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                       height=550, coloraxis_showscale=False,
                       yaxis=dict(categoryorder="total ascending"),
                       margin=dict(l=10,r=30,t=50,b=30))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

    # Symptom severity
    st.markdown('<div class="section-title">⚠️ Symptom Severity Weights (Top 30)</div>', unsafe_allow_html=True)
    sev = sev_df.copy()
    sev.columns = sev.columns.str.strip()
    sev["Symptom"] = sev["Symptom"].str.strip().str.replace("_"," ").str.title()
    sev = sev.sort_values("weight", ascending=False).head(30)
    fig3 = px.bar(sev, x="weight", y="Symptom", orientation="h",
                  color="weight", color_continuous_scale="Reds", template="plotly_dark",
                  title="Medical Severity Weight of Each Symptom")
    fig3.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                       height=650, coloraxis_showscale=False,
                       yaxis=dict(categoryorder="total ascending"),
                       xaxis_title="Severity Weight",
                       margin=dict(l=10,r=30,t=50,b=30))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")

    # Avg symptoms per disease
    st.markdown('<div class="section-title">📊 Average Symptoms per Disease</div>', unsafe_allow_html=True)
    spd = []
    for d in dataset["Disease"].unique():
        sub = dataset[dataset["Disease"]==d]
        avg = sub[sym_cols].apply(lambda x: x.str.strip().ne("").sum(), axis=1).mean()
        spd.append({"Disease": d.strip(), "Avg Symptoms": round(avg,1)})
    spd_df = pd.DataFrame(spd).sort_values("Avg Symptoms", ascending=False)
    fig4 = px.bar(spd_df, x="Avg Symptoms", y="Disease", orientation="h",
                  color="Avg Symptoms", color_continuous_scale="Viridis", template="plotly_dark",
                  title="Average Number of Symptoms Reported per Disease")
    fig4.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                       height=750, coloraxis_showscale=False,
                       yaxis=dict(categoryorder="total ascending"),
                       margin=dict(l=10,r=30,t=50,b=30))
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")

    # Risk level pie
    st.markdown('<div class="section-title">🎯 Risk Level Distribution</div>', unsafe_allow_html=True)
    risk_counts = {"High":0, "Medium":0, "Low":0}
    for d in disease_list:
        risk_counts[get_risk(d)] += 1
    risk_df = pd.DataFrame(list(risk_counts.items()), columns=["Risk","Count"])
    fig5 = px.pie(risk_df, values="Count", names="Risk",
                  color="Risk",
                  color_discrete_map={"High":"#dc2626","Medium":"#d97706","Low":"#059669"},
                  template="plotly_dark", title="Disease Distribution by Risk Level", hole=0.4)
    fig5.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14", height=400)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("---")

    st.markdown('<div class="section-title">📋 Raw Dataset Preview (First 50 rows)</div>', unsafe_allow_html=True)
    st.dataframe(dataset.head(50), use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataset.to_excel(writer,  index=False, sheet_name="Raw Data")
        dc.to_excel(writer,       index=False, sheet_name="Disease Counts")
        sym_df.to_excel(writer,   index=False, sheet_name="Symptom Frequency")
        spd_df.to_excel(writer,   index=False, sheet_name="Symptoms Per Disease")
        risk_df.to_excel(writer,  index=False, sheet_name="Risk Distribution")
    output.seek(0)
    st.download_button("⬇️ Download Full EDA Report as Excel", data=output,
        file_name="biopredict_eda_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)

# ══════════════════════════════════════════════
#  ML MODEL INSIGHTS
# ══════════════════════════════════════════════
elif page == "🤖 ML Model Insights":
    st.markdown('<div class="big-title" style="font-size:2rem;">🤖 ML Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Random Forest Classifier trained on Real Kaggle Medical Dataset</div>', unsafe_allow_html=True)
    st.markdown("---")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🎯 Test Accuracy",   f"{round(accuracy*100,1)}%")
    m2.metric("📊 CV Mean",         f"{round(cv_scores.mean()*100,1)}%")
    m3.metric("📉 CV Std Dev",      f"±{round(cv_scores.std()*100,1)}%")
    m4.metric("🌳 Trees in Forest", "200")
    st.markdown("---")

    # Feature importance
    st.markdown('<div class="section-title">🔬 Feature Importance (Top 20 Symptoms)</div>', unsafe_allow_html=True)
    fi_df = pd.DataFrame(list(feat_imp.items()), columns=["Symptom","Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)
    fi_df["Symptom"] = fi_df["Symptom"].apply(lambda x: x.replace("_"," ").title())
    fig_fi = px.bar(fi_df, x="Importance", y="Symptom", orientation="h",
                    color="Importance", color_continuous_scale="teal", template="plotly_dark",
                    title="Which Symptoms Matter Most for Diagnosis?")
    fig_fi.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                         height=600, coloraxis_showscale=False,
                         yaxis=dict(categoryorder="total ascending"),
                         margin=dict(l=10,r=30,t=50,b=30))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown("---")

    # Confusion matrix
    st.markdown('<div class="section-title">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
    st.caption("Diagonal = correct predictions. Off-diagonal = misclassifications.")
    fig_cm = px.imshow(cm, x=disease_list, y=disease_list,
                       color_continuous_scale="Blues", text_auto=True,
                       aspect="auto", template="plotly_dark",
                       title="Confusion Matrix — Predicted vs Actual")
    fig_cm.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                         height=700, xaxis=dict(tickangle=35),
                         xaxis_title="Predicted Label", yaxis_title="Actual Label",
                         margin=dict(l=10,r=30,t=60,b=30))
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown("---")

    # Classification report
    st.markdown('<div class="section-title">📋 Classification Report</div>', unsafe_allow_html=True)
    st.caption("Precision, Recall, F1-Score per disease.")
    rep_df = pd.DataFrame(report).T
    rep_df = rep_df.drop([r for r in ["accuracy","macro avg","weighted avg"] if r in rep_df.index], errors="ignore")
    rep_df = rep_df[["precision","recall","f1-score","support"]].round(3)
    st.dataframe(rep_df, use_container_width=True)
    st.markdown("---")

    # Cross validation
    st.markdown('<div class="section-title">📈 5-Fold Cross Validation</div>', unsafe_allow_html=True)
    st.caption("Model tested on 5 different groups of patients — proves consistent performance.")
    cv_df = pd.DataFrame({
        "Fold":     [f"Fold {i+1}" for i in range(5)],
        "Accuracy": [round(s*100,2) for s in cv_scores]
    })
    fig_cv = px.bar(cv_df, x="Fold", y="Accuracy", color="Accuracy",
                    color_continuous_scale="teal", template="plotly_dark", text="Accuracy",
                    title=f"CV Mean: {cv_scores.mean()*100:.1f}%  |  Std: ±{cv_scores.std()*100:.1f}%")
    fig_cv.add_hline(y=cv_scores.mean()*100, line_dash="dash", line_color="#f39c12",
                     annotation_text=f"Mean: {cv_scores.mean()*100:.1f}%",
                     annotation_font_color="#f39c12")
    fig_cv.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                         height=380, yaxis=dict(range=[0,110]),
                         coloraxis_showscale=False,
                         margin=dict(l=10,r=30,t=60,b=30))
    fig_cv.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_cv, use_container_width=True)
    st.markdown("---")

    # Model architecture
    st.markdown('<div class="section-title">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| n_estimators | 200 |
| max_depth | 15 |
| min_samples_split | 4 |
| class_weight | balanced |
| random_state | 42 |
| Validation | 5-Fold Cross Validation |
| Test Accuracy | {round(accuracy*100,1)}% |
        """)
    with col2:
        st.markdown(f"""
| Dataset Info | Value |
|---|---|
| Source | Kaggle Real Medical Dataset |
| Total Records | {len(dataset)} |
| Unique Diseases | {len(disease_list)} |
| Unique Symptoms | {len(all_symptoms)} |
| Train / Test Split | 80% / 20% |
| Input Type | Binary Symptom Vector |
| Output | Disease + Confidence % |
| CV Mean Accuracy | {round(cv_scores.mean()*100,1)}% |
        """)

# ══════════════════════════════════════════════
#  PATIENT RECORDS
# ══════════════════════════════════════════════
elif page == "📋 Patient Records":
    st.markdown('<div class="big-title" style="font-size:2rem;">📋 Patient Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">All predictions saved in SQLite database</div>', unsafe_allow_html=True)
    st.markdown("---")

    records = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", conn)

    if records.empty:
        st.info("📂 No records yet. Go to **Disease Predictor** and make a prediction!")
    else:
        k1,k2,k3 = st.columns(3)
        k1.metric("Total Records",       len(records))
        k2.metric("Unique Patients",     records["name"].nunique())
        k3.metric("Most Common Disease", records["predicted_disease"].mode()[0])
        st.markdown("---")

        st.markdown('<div class="section-title">📊 Prediction Summary</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        with rc1:
            dc2 = records["predicted_disease"].value_counts().head(10).reset_index()
            dc2.columns = ["Disease","Count"]
            fig_r1 = px.bar(dc2, x="Count", y="Disease", orientation="h",
                            color="Count", color_continuous_scale="teal", template="plotly_dark",
                            title="Top Predicted Diseases")
            fig_r1.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14",
                                 height=350, coloraxis_showscale=False,
                                 yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_r1, use_container_width=True)
        with rc2:
            rk2 = records["risk_level"].value_counts().reset_index()
            rk2.columns = ["Risk","Count"]
            fig_r2 = px.pie(rk2, values="Count", names="Risk",
                            color="Risk",
                            color_discrete_map={"High":"#dc2626","Medium":"#d97706","Low":"#059669"},
                            template="plotly_dark", title="Risk Level Distribution", hole=0.4)
            fig_r2.update_layout(plot_bgcolor="#0d1520", paper_bgcolor="#080c14", height=350)
            st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📋 All Records</div>', unsafe_allow_html=True)
        st.dataframe(records.drop("id",axis=1,errors="ignore"),
                     use_container_width=True, hide_index=True)
        st.markdown("---")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            records.to_excel(writer, index=False, sheet_name="Patient Records")
        output.seek(0)
        st.download_button("⬇️ Download Patient Records as Excel", data=output,
            file_name="patient_records.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Records"):
            c.execute("DELETE FROM predictions")
            conn.commit()
            st.rerun()
