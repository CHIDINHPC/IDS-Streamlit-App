from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)
st.markdown("---")
st.caption("🚀 Developed by IDS AI System | 2026")
dataset_option = st.sidebar.selectbox(
    "Chọn Dataset",
    ("NSL-KDD", "UNSW-NB15")
)
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="IDS Security Dashboard",
    layout="wide",
    page_icon="🛡"
)
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
html, body, [class*="css"] {
    background-color: #0b0f19;
    color: white;
}

/* ===== MAIN TITLE ===== */
h1 {
    text-align: center;
    color: #00FFD1;
    font-weight: 800;
}

/* ===== CARD UI ===== */
.card {
    background: linear-gradient(145deg, #111827, #1f2937);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 15px rgba(0,255,200,0.2);
}

/* ===== METRIC ===== */
[data-testid="metric-container"] {
    background: #111827;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #00FFD1;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* ===== BUTTON ===== */
.stButton button {
    background: linear-gradient(90deg,#00FFD1,#00A3FF);
    color: black;
    font-weight: bold;
    border-radius: 10px;
}

/* ===== TABLE ===== */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
<h1 style='text-align:center;
background: linear-gradient(90deg,#00FFD1,#00A3FF);
-webkit-background-clip: text;
color: transparent;'>
🛡 IDS SECURITY DASHBOARD
</h1>
</style>
""", unsafe_allow_html=True)
st.title("🛡 Network Intrusion Detection System")
# =========================
# LOAD DATA
@st.cache_data
def load_data(dataset_name):

    if dataset_name == "NSL-KDD":
        df = pd.read_csv("nsl_kdd_dataset.csv")

        if "difficulty" in df.columns:
            df = df.drop(columns=["difficulty"])

        df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    else:  # UNSW-NB15

        df = pd.read_csv("UNSW_NB15_training-set.csv")

    df["label"] = df["label"].astype(int)

    if "attack_cat" in df.columns:
        df = df.drop(columns=["attack_cat"])
    return df


data = load_data(dataset_option)

X = data.drop("label", axis=1)
y = data["label"]

X = pd.get_dummies(X)

# =========================
# TRAIN TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧭 IDS Menu")
st.info(f"Dataset đang sử dụng: {dataset_option}")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Dashboard",
        "Model Performance",
        "Model Comparison",   # 👈 thêm dòng này
        "Single Detection",
        "Batch CSV Detection",
        "Realtime Monitor"
    ]
)

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":

    st.header("📊 Network Traffic Overview")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Packets", len(data))
    col2.metric("Attack Packets", sum(y))
    col3.metric("Normal Packets", len(y)-sum(y))
    col4.metric("Attack Rate", f"{y.mean()*100:.2f}%")

    st.divider()

    fig = px.pie(
        names=["Normal","Attack"],
        values=[len(y)-sum(y),sum(y)],
        title="Traffic Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("📈 Attack Trend")

    trend = pd.DataFrame({
        "packet":range(len(y)),
        "attack":y
    })

    fig = px.line(trend,x="packet",y="attack")
    st.plotly_chart(fig,use_container_width=True)

# =========================
# MODEL PERFORMANCE
# =========================
elif page == "Model Performance":

    st.header("🎯 Model Evaluation")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Accuracy",f"{acc:.2%}")
    col2.metric("Precision",f"{precision:.2%}")
    col3.metric("Recall",f"{recall:.2%}")
    col4.metric("F1 Score",f"{f1:.2%}")

    cm = confusion_matrix(y_test,y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp/(fp+tn)

    st.metric("False Positive Rate",f"{fpr:.2%}")

    # ✅ Confusion Matrix
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="reds",
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, key="cm1")

    # =====================
    # PR Curve
    # =====================
    y_prob = model.predict_proba(X_test)[:,1]

    precision_curve, recall_curve, _ = precision_recall_curve(y_test,y_prob)
    pr_auc = auc(recall_curve,precision_curve)

    st.metric("PR-AUC",f"{pr_auc:.3f}")

    st.subheader("Precision Recall Curve")

    fig_pr = px.line(
        x=recall_curve,
        y=precision_curve,
        labels={"x":"Recall","y":"Precision"},
        title="PR Curve"
    )
    st.plotly_chart(fig_pr, key="pr_curve")

    # =====================
    # Feature Importance
    # =====================
    st.subheader("Feature Importance")

    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]

    fig_fi = px.bar(
        x=importances[indices],
        y=[X.columns[i] for i in indices],
        orientation='h',
        title="Top 15 Features"
    )

    st.plotly_chart(fig_fi, use_container_width=True, key="feature_importance")

# =========================
# MODEL COMPARISON
# =========================
elif page == "Model Comparison":

    st.header("⚖️ Model Comparison")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    results = []

    for name, m in models.items():

        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    df_results = pd.DataFrame(results)

    st.subheader("📊 Comparison Table")
    st.dataframe(df_results)

    # =========================
    # BAR CHART
    # =========================
    st.subheader("📈 Model Performance Comparison")

    fig = px.bar(
        df_results,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        barmode="group",
        title="Model Comparison Metrics"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # BEST MODEL
    # =========================
    best_model = df_results.sort_values(by="F1 Score", ascending=False).iloc[0]

    st.success(f"🏆 Best Model: {best_model['Model']} (F1 = {best_model['F1 Score']:.2f})")
# =========================
# SINGLE DETECTION
# =========================
elif page == "Single Detection":

    st.header("🔍 Detect Network Packet")

    duration = st.number_input("Duration",0,100000,10)
    src_bytes = st.number_input("Source Bytes",0,100000,100)
    dst_bytes = st.number_input("Destination Bytes",0,100000,50)
    count = st.number_input("Count",0,1000,5)
    srv_count = st.number_input("Srv Count",0,1000,5)

    input_df = pd.DataFrame([{
        "duration":duration,
        "src_bytes":src_bytes,
        "dst_bytes":dst_bytes,
        "count":count,
        "srv_count":srv_count
    }])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=X.columns,fill_value=0)

    input_scaled = scaler.transform(input_df)

    if st.button("🚀 Detect Attack"):

        with st.spinner("Analyzing Packet..."):
            time.sleep(1)

        pred = model.predict(input_scaled)[0]

        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 0:
            st.success(f"✅ Normal Traffic\nAttack Probability: {prob:.2%}")
        else:
            st.error(f"🚨 Attack Detected!\nProbability: {prob:.2%}")

# =========================
# CSV DETECTION
# =========================
elif page == "Batch CSV Detection":

    st.header("📂 Upload CSV for Detection")

    uploaded_file = st.file_uploader("Upload CSV",type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.dataframe(df.head())

        df = pd.get_dummies(df)

        df = df.reindex(columns=X.columns,fill_value=0)

        df_scaled = scaler.transform(df)

        preds = model.predict(df_scaled)

        df["Prediction"] = preds

        st.subheader("Detection Result")

        st.dataframe(df)

        st.metric("Attack Count",sum(preds))

# =========================
# REALTIME MONITOR
# =========================
elif page == "Realtime Monitor":
    st.header("⚡ Real-time Network Monitoring")

    packet_placeholder = st.empty()
    chart_placeholder = st.empty()

    normal_history = []
    attack_history = []

    for i in range(200):

        sample = data.sample(1)

        row = sample.drop("label", axis=1)

        row = pd.get_dummies(row)

        row = row.reindex(columns=X.columns, fill_value=0)

        row_scaled = scaler.transform(row)

        pred = model.predict(row_scaled)[0]

        # =========================
        # UPDATE HISTORY
        # =========================
        if pred == 0:
            normal_history.append(1)
            attack_history.append(0)
        else:
            normal_history.append(0)
            attack_history.append(1)

        total_packets = i + 1
        total_attacks = sum(attack_history)
        total_normal = sum(normal_history)

        attack_rate = total_attacks / total_packets

        # =========================
        # PACKET INFO
        # =========================
        with packet_placeholder.container():

            col1, col2 = st.columns([2,1])

            with col1:

            

                st.dataframe(sample)

                if pred == 0:
                    st.success("Normal Traffic")
                else:
                    st.error("🚨 Intrusion Detected")

            with col2:

                st.metric("Total Packets", total_packets)
                st.metric("Attacks", total_attacks)
                st.metric("Normal", total_normal)
                st.metric("Attack Rate", f"{attack_rate*100:.2f}%")

        # =========================
        # CREATE CHART DATA
        # =========================
        chart_df = pd.DataFrame({
            "Normal": normal_history,
            "Attack": attack_history
        })

        # =========================
        # LINE CHART
        # =========================
        fig_line = px.line(
            chart_df,
            title="Realtime Attack Detection"
        )

        # =========================
        # PIE CHART
        # =========================
        fig_pie = px.pie(
            names=["Normal","Attack"],
            values=[total_normal,total_attacks],
            title="Traffic Ratio"
        )

        # =========================
        # GAUGE CHART
        # =========================
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attack_rate*100,
            title={'text':"Threat Level (%)"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"red"},
                'steps':[
                    {'range':[0,30],'color':"green"},
                    {'range':[30,60],'color':"yellow"},
                    {'range':[60,100],'color':"red"}
                ]
            }
        ))

        # =========================
        # SHOW CHARTS
        # =========================
        with chart_placeholder.container():

            col1, col2, col3 = st.columns(3)

            col1.plotly_chart(fig_line, use_container_width=True, key=f"line_{i}")
            col2.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{i}")
            col3.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{i}")

        time.sleep(1)