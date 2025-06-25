import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("women_reviews_cleaned.csv")
    df.rename(columns=lambda x: x.strip().replace(' ', '_').lower(), inplace=True)
    df['review_text'] = df['review_text'].fillna("")
    return df

df = load_data()

# Judul Aplikasi
st.title("Dashboard Analisis Ulasan E-Commerce")
st.subheader("Women's Clothing E-Commerce Reviews")

# Sidebar Filter
st.sidebar.title("Filter")
min_age, max_age = st.sidebar.slider("Usia", int(df.age.min()), int(df.age.max()), (30, 50))
rating_filter = st.sidebar.multiselect("Pilih Rating", df.rating.unique(), default=df.rating.unique())

filtered_df = df[(df['age'] >= min_age) & (df['age'] <= max_age)]
filtered_df = filtered_df[filtered_df['rating'].isin(rating_filter)]

# Visualisasi Rating
st.markdown("### Distribusi Rating")
fig, ax = plt.subplots()
sns.countplot(x='rating', data=filtered_df, ax=ax)
st.pyplot(fig)

# WordCloud
st.markdown("### WordCloud Ulasan")
text = ' '.join(filtered_df['review_text'].dropna().astype(str))
wc = WordCloud(width=800, height=400, background_color='white').generate(text)
fig_wc, ax_wc = plt.subplots(figsize=(10,5))
ax_wc.imshow(wc, interpolation='bilinear')
ax_wc.axis('off')
st.pyplot(fig_wc)

# Tampilkan Data Mentah
if st.checkbox("Tampilkan Data"):
    st.dataframe(filtered_df)
