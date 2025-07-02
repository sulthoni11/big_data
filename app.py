import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Konfigurasi dasar halaman
st.set_page_config(layout="wide")
st.title("🧥 Analisis E-Commerce Fashion Wanita")

# Sidebar pilihan dataset
dataset_choice = st.sidebar.selectbox(
    "Pilih Dataset:",
    ["Ulasan Pakaian Wanita", "Fashion Dataset"]
)

# Fungsi untuk membersihkan dataset ulasan pakaian wanita
@st.cache_data
def load_clean_reviews_data():
    data = pd.read_csv("clean_Womens_Clothing_Reviews.csv")
    return data

# Fungsi untuk membersihkan dataset fashion
@st.cache_data
def load_clean_fashion_data():
    data = pd.read_csv("clean_FashionDataset.csv")
    return data

# ==============================
# 📌 Dataset: Ulasan Pakaian Wanita
# ==============================
if dataset_choice == "Ulasan Pakaian Wanita":
    data = load_clean_reviews_data()
    st.subheader("🔍 Ulasan Pakaian Wanita")
    st.write(data.head())

    # Barplot jumlah review per class
    st.markdown("### 📊 Jumlah Ulasan per Class Name")
    fig, ax = plt.subplots(figsize=(12, 6))
    class_reviews = data['Class Name'].value_counts()
    sns.barplot(x=class_reviews.index, y=class_reviews.values, palette='magma', ax=ax)
    ax.set_title("Jumlah Ulasan per Class Name")
    ax.set_ylabel("Jumlah Ulasan")
    ax.set_xlabel("Class Name")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Pie chart department
    st.markdown("### 🧾 Distribusi Ulasan Berdasarkan Department")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    department_reviews = data['Department Name'].value_counts()
    ax2.pie(department_reviews.values, labels=department_reviews.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette('coolwarm', len(department_reviews)))
    ax2.set_title("Distribusi Ulasan Berdasarkan Department Name")
    ax2.axis('equal')
    st.pyplot(fig2)

    # Wordcloud Review Text
    st.markdown("### ☁️ WordCloud Review Pelanggan")
    text = " ".join(data['Review Text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis("off")
    st.pyplot(fig3)

# ==============================
# 📌 Dataset: Fashion
# ==============================
else:
    data = load_clean_fashion_data()
    st.subheader("👗 Fashion Dataset")
    st.write(data.head())

    st.markdown("### 🔠 Kategori Unik")
    st.write(data["Category"].unique())

    # Filter untuk "Westernwear"
    Category_fashion = data[data["Category"] == "Westernwear"]

    # WordCloud dari kolom Deatils
    st.markdown("### ☁️ WordCloud Produk Westernwear")
    text = " ".join(i for i in Category_fashion["Deatils"])
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    fig4, ax4 = plt.subplots(figsize=(15, 10))
    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.axis("off")
    st.pyplot(fig4)

    # Brand paling populer (opsional: bisa difilter dulu misalnya LingerieNightwear)
    st.markdown("### 📈 Top 10 Brand")
    fig5, ax5 = plt.subplots(figsize=(20, 8))
    data["BrandName"].value_counts().head(10).plot.bar(ax=ax5)
    ax5.set_title("10 Brand Paling Sering Muncul")
    st.pyplot(fig5)
