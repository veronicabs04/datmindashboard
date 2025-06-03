import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("naive_bayes_diabetes_model.pkl")

def plot_roc_curve(model, df):
    X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
    y = df["Outcome"]

    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig

def show_missing_values(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    num_columns = len(df.columns)
    colors = plt.cm.viridis(np.linspace(0, 1, num_columns))
    msno.bar(df, color=colors, ax=ax)
    st.pyplot(fig)

def show_clustering(df):
    #preprocess
    features = df.drop("Outcome", axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    #k-means
    kmeans= KMeans(n_clusters=3, random_state=42, n_init="auto")
    clusters =kmeans.fit_predict(scaled_features)

    #PCA
    pca=PCA(n_components=2)
    pca_result= pca.fit_transform(scaled_features)

    #Transform user input
    user_scaled = scaler.transform(input_df)
    user_pca = pca.transform(user_scaled)
    user_cluster = kmeans.predict(user_scaled)[0]

        # Hitung Silhouette Score
    sil_score = silhouette_score(scaled_features, clusters)

    st.markdown(f"📈 **Silhouette Score untuk KMeans (k=3):** `{sil_score:.4f}`")
    if sil_score < 0.5:
        st.info("🔍 Skor ini menunjukkan bahwa clustering masih bisa diperbaiki. Mungkin k terlalu kecil atau data overlap.")
    elif sil_score < 0.7:
        st.success("👍 Clustering cukup baik, tapi masih ada sedikit overlap.")
    else:
        st.success("🚀 Clustering sangat baik! Cluster saling terpisah jelas.")


    # DataFrame untuk visualisasi
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    pca_df["Cluster"] = clusters.astype(str)

    # Plot
    st.subheader("📊 Hasil Clustering (berdasarkan input Anda)")

    fig = go.Figure()

    # Tambahkan data semua cluster
    for clust in sorted(pca_df["Cluster"].unique()):
        subset = pca_df[pca_df["Cluster"] == clust]
        fig.add_trace(go.Scatter(
            x=subset["PCA1"], y=subset["PCA2"],
            mode='markers',
            name=f'Cluster {clust}',
            marker=dict(size=7),
            opacity=0.6
        ))

    # Tambahkan titik user
    fig.add_trace(go.Scatter(
        x=[user_pca[0][0]], y=[user_pca[0][1]],
        mode='markers+text',
        name="Anda",
        marker=dict(size=12, color='black', symbol='x'),
        text=["Anda"],
        textposition="top center"
    ))

    fig.update_layout(title="Visualisasi PCA Clustering (Dengan Input Anda)", 
                      xaxis_title="PCA1", yaxis_title="PCA2")
    st.plotly_chart(fig)

    st.markdown(f"📌 **Input Anda termasuk dalam Cluster {user_cluster}**")
# Load data & model
df = load_data()
model = load_model()

st.title("Dashboard Prediksi Diabetes 🩺")
st.markdown("Prediksi kemungkinan diabetes berdasarkan data medis menggunakan model Naive Bayes.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.slider("Pregnancies", int(df.Pregnancies.min()), int(df.Pregnancies.max()), int(df.Pregnancies.mean()))
    glucose = st.slider("Glucose", int(df.Glucose.min()), int(df.Glucose.max()), int(df.Glucose.mean()))
    bp = st.slider("BloodPressure", int(df.BloodPressure.min()), int(df.BloodPressure.max()), int(df.BloodPressure.mean()))
    skin = st.slider("SkinThickness", int(df.SkinThickness.min()), int(df.SkinThickness.max()), int(df.SkinThickness.mean()))

with col2:
    insulin = st.slider("Insulin", int(df.Insulin.min()), int(df.Insulin.max()), int(df.Insulin.mean()))
    bmi = st.slider("BMI", float(df.BMI.min()), float(df.BMI.max()), float(df.BMI.mean()))
    dpf = st.slider("DiabetesPedigreeFunction", float(df.DiabetesPedigreeFunction.min()), float(df.DiabetesPedigreeFunction.max()), float(df.DiabetesPedigreeFunction.mean()))
    age = st.slider("Age", int(df.Age.min()), int(df.Age.max()), int(df.Age.mean()))

# Pastikan input_df sesuai urutan & nama kolom saat training
feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
input_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=feature_names)

if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("🔍 Hasil Prediksi")
    if prediction == 1:
        st.error(f"🩸 Diabetes dengan probabilitas {probability[1]:.2%}")
    else:
        st.success(f"✅ Tidak Diabetes dengan probabilitas {probability[0]:.2%}")

    st.plotly_chart(
        go.Figure(data=[go.Bar(x=["Tidak Diabetes", "Diabetes"], y=probability, marker_color=['blue', 'red'])])
        .update_layout(title="Probabilitas Kelas", yaxis=dict(range=[0,1]))
    )

    st.pyplot(plot_roc_curve(model, df))

    st.markdown("### 💡 Rekomendasi")
    if prediction == 1:
        st.markdown("- Konsultasi dengan dokter segera.\n- Pantau kadar gula darah secara rutin.")
    else:
        st.markdown("- Jaga pola makan sehat.\n- Lanjutkan gaya hidup aktif.")

# Tambahan: Visualisasi Missing Value
st.markdown("---")
st.subheader("📉 Visualisasi Missing Value")
show_missing_values(df)

# Tambahan: Clustering
st.markdown("---")
show_clustering(df)
