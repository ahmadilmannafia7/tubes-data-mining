import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Konfigurasi halaman
st.set_page_config(page_title="Data Mining App", layout="wide")
st.title("📊 Data Mining Project: Clustering & Classification")

# Sidebar untuk upload file
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset (.xlsx)", type=["xlsx"])

# Logika utama aplikasi
if uploaded_file:
    # Baca dua sheet: sebelum dan sesudah seleksi
    df_raw = pd.read_excel(uploaded_file, sheet_name='Before_Selection')
    df = pd.read_excel(uploaded_file, sheet_name='After_Selection')

    # Tabs untuk memisahkan tampilan
    tab1, tab2, tab3 = st.tabs(["📁 Data", "🔍 Clustering", "✅ Classification"])

    with tab1:
        st.subheader("📁 Data Mentah (Before Selection)")
        st.dataframe(df_raw.head(), use_container_width=True)

        st.subheader("📁 Data Setelah Seleksi (After Selection)")
        st.dataframe(df.head(), use_container_width=True)

    with tab2:
        st.subheader("🔍 Clustering dengan KMeans")

        X = df.drop(columns=['Purchased'])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters

        st.success("✅ Clustering berhasil dilakukan.")
        st.write("Distribusi Cluster:")
        st.bar_chart(df['Cluster'].value_counts())

        # Visualisasi scatter 2D dari 2 fitur pertama
        st.write("Visualisasi Cluster (2 Fitur Pertama)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
        ax.set_xlabel("Fitur 1")
        ax.set_ylabel("Fitur 2")
        ax.set_title("Visualisasi Cluster")
        st.pyplot(fig)

    with tab3:
        st.subheader("✅ Classification dengan Random Forest")

        # Gunakan data tanpa kolom Cluster dan target Purchased
        X = df.drop(columns=['Purchased', 'Cluster'])
        y = df['Purchased']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric(label="🎯 Akurasi Model", value=f"{acc:.2%}")
        st.text("📄 Classification Report:")
        st.text(classification_report(y_test, y_pred))
else:
    st.info("Silakan unggah file Excel dengan sheet 'Before_Selection' dan 'After_Selection'.")
