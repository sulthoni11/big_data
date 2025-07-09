import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import nltk
from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load data
@st.cache_data
def load_data():
    url = 'clean_data_tokopedia.csv'
    df = pd.read_csv(url)
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigasi")
option = st.sidebar.radio("Pilih menu:", ['Informasi Dataset', 'Visualisasi Data', 'Analisis Teks'])

# Preprocessing
def preprocess_text(df):
    df = df.copy()
    df['Customer Review'] = df['Customer Review'].str.lower()

    word_mapping = {
        "yg": "yang","tp": "tapi","bgt": "begitu","jg": "juga","dgn": "dengan","pake": "pakai","jd": "jadi","klo": "kalo",
        "lg": "lagi","dr": "dari","utk": "untuk","gk": "tidak","sdh": "sudah","ngga": "tidak","brg": "barang","ga": "tidak",
        "gak": "tidak","rapi": "rapih","cpt": "cepat","krn": "karena","sy": "saya","tdk": "tidak","nggak": "tidak",
        "kalo": "kalau","cepet": "cepat","gitu": "begitu","udh": "udah","d": "di","g": "tidak","tgl": "tanggal",
        "sampe": "sampai","mantab": "mantap", "mantaapp": "mantap"
    }

    def normalize_word(text):
        pattern = r'\b(' + '|'.join(re.escape(word) for word in word_mapping.keys()) + r')\b'
        return re.sub(pattern, lambda match: word_mapping[match.group(0)], text)

    df['Customer Review'] = df['Customer Review'].apply(normalize_word)

    stopwords_list = stopwords.words('indonesian')
    keep = ['baik', 'biasa', 'bukan', 'amat', 'baru', 'cukup', 'kurang', 'lama', 'bagus', 'sangat', 'sedikit',
            'tak', 'tidak', 'kali', 'sukses']
    new_stopwords_list = [word for word in stopwords_list if word not in keep]
    extra_stopwords = ['gan', 'nya', 'aja', 'sih', 'deh', 'n', 'dah', 'ya', 'gitu', 'pa', 'kalo', 'udah', 'y']
    new_stopwords_list.extend(extra_stopwords)
    stop_words = set(new_stopwords_list)

    def remove_stopwords(text):
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word.lower() not in stop_words])

    def remove_special_chars(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    df['Customer Review'] = df['Customer Review'].apply(remove_stopwords)
    df['Customer Review'] = df['Customer Review'].apply(remove_special_chars)

    return df

# Menu 1: Informasi Dataset
# Menu 1: Informasi Dataset
if option == 'Informasi Dataset':
    st.title("Informasi Dataset")
    st.write("Dataset Review Produk Indonesia (PRDECT-ID)")

    # Tampilkan kolom yang tersedia untuk debugging
    # st.write("Kolom tersedia:", df.columns.tolist())

    # --- Filter Sidebar ---
    st.sidebar.markdown("### Filter Dataset")

    # Filter Kategori Produk
    categories = df['Category'].dropna().unique()
    selected_categories = st.sidebar.multiselect("Pilih Kategori Produk:", options=categories, default=categories)

    # Filter Harga
    min_price = int(df['Price'].min())
    max_price = int(df['Price'].max())
    selected_price = st.sidebar.slider("Range Harga Produk (Rp):", min_price, max_price, (min_price, max_price))

    # Filter Sentimen
    sentiments = df['Sentiment'].dropna().unique()
    selected_sentiments = st.sidebar.multiselect("Pilih Sentimen:", options=sentiments, default=sentiments)

    # --- Terapkan Filter ---
    filtered_df = df[
        (df['Category'].isin(selected_categories)) &
        (df['Price'].between(*selected_price)) &
        (df['Sentiment'].isin(selected_sentiments))
    ]

    # --- Tampilkan Data ---
    st.markdown("### Dataset Setelah Diterapkan Filter")
    st.write(filtered_df.head())

    st.markdown("### Statistik Dataset (Setelah Filter)")
    st.write(f"Jumlah baris setelah filter: {len(filtered_df)}")
    st.write(f"Data kosong:\n{filtered_df.isnull().sum()}")
    st.write(f"Duplikat: {filtered_df.duplicated().sum()}")


# if option == 'Informasi Dataset':
#     st.title("Informasi Dataset")
#     st.write("Dataset Review Produk Indonesia (PRDECT-ID)")

#     st.write(df.head())
#     st.markdown("### Informasi Umum")
#     st.write(df.info())

#     st.markdown("### Statistik Dataset")
#     st.write(f"Jumlah baris: {len(df)}")
#     st.write(f"Data kosong:\n{df.isnull().sum()}")
#     st.write(f"Duplikat: {df.duplicated().sum()}")

# Menu 2: Visualisasi Data
elif option == 'Visualisasi Data':
    st.title("Visualisasi Data")

    # Bar plot sentimen
    st.subheader("Distribusi Sentimen")
    fig, ax = plt.subplots()
    sns.countplot(x='Sentiment', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)

    # Bar plot kategori vs sentimen
    st.subheader("Distribusi Kategori per Sentimen")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.countplot(y="Category", hue="Sentiment", palette="bright", data=df, ax=ax2)
    ax2.legend(bbox_to_anchor=(1, 1))
    st.pyplot(fig2)

    # Korelasi heatmap
    st.subheader("Korelasi Spearman antar Fitur Numerik")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr(method='spearman')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax3)
    st.pyplot(fig3)

    # Scatter plot
    st.subheader("Scatter Plot Harga vs Jumlah Terjual")
    fig4 = px.scatter(df, x='Price', y='Number Sold', color='Sentiment',
                      title="Hubungan Jumlah Terjual & Harga",
                      labels={'Price': 'Harga', 'Number Sold': 'Jumlah Terjual'})
    st.plotly_chart(fig4)

# Menu 3: Analisis Teks
elif option == 'Analisis Teks':
    st.title("Analisis Teks")

    # Bersihkan data
    # st.info("Proses pembersihan teks sedang berlangsung...")
    clean_df = preprocess_text(df)

    # Kata paling umum
    st.subheader("20 Kata Paling Umum")
    clean_df['temp_list'] = clean_df['Customer Review'].apply(lambda x: x.split())
    top = Counter([word for sublist in clean_df['temp_list'] for word in sublist])
    top_df = pd.DataFrame(top.most_common(20), columns=['Kata', 'Frekuensi'])
    st.dataframe(top_df.style.background_gradient(cmap='Blues'))

    # Wordcloud Positif
    st.subheader("Wordcloud Review Positif")
    txt_pos = ' '.join(clean_df[clean_df['Sentiment'] == "Positive"]['Customer Review'])
    wc_pos = WordCloud(background_color='black', max_words=100, width=800, height=600).generate(txt_pos)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.imshow(wc_pos, interpolation='bilinear')
    ax5.axis("off")
    st.pyplot(fig5)

    # Wordcloud Negatif
    st.subheader("Wordcloud Review Negatif")
    txt_neg = ' '.join(clean_df[clean_df['Sentiment'] == "Negative"]['Customer Review'])
    wc_neg = WordCloud(background_color='black', max_words=100, width=800, height=600).generate(txt_neg)
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.imshow(wc_neg, interpolation='bilinear')
    ax6.axis("off")
    st.pyplot(fig6)
