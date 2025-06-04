import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.title("Data Mining Project: Clustering & Classification")

uploaded_file = st.file_uploader(r"E:\SMT 6\DATA MINING\tubes data mining\file dataset\data_mining_project_dataset.xlsx", type=["xlsx"])
if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, sheet_name='Before_Selection')
    df = pd.read_excel(uploaded_file, sheet_name='After_Selection')

    st.subheader("Data Mentah")
    st.dataframe(df_raw.head())

    st.subheader("Data Setelah Seleksi & Encoding")
    st.dataframe(df.head())

    # Pisahkan fitur dan target
    X = df.drop(columns=['Purchased'])
    y = df['Purchased']

    # Standardisasi fitur untuk clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering dengan KMeans
    st.subheader("Clustering (KMeans)")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    st.write("Distribusi Cluster:")
    st.bar_chart(df['Cluster'].value_counts())

    # Classification dengan Random Forest
    st.subheader("Classification (Random Forest)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.write("Akurasi:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
