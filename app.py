# import streamlit as st
# import pickle

# with open('svc_model.pkl', 'rb') as file:
#     svc_model = pickle.load(file)

# with open('tfidf_vectorizer.pkl', 'rb') as file:
#     tfidf_vectorizer = pickle.load(file)


# st.title("IMDB Movie Review Sentiment Analysis")
# st.write("This app classifies IMDB movie reviews as positive or negative.")

# review_text = st.text_area("Enter a movie review")

# if st.button("Predict Sentiment"):
#     transformed_text = tfidf_vectorizer.transform([review_text])

#     prediction = svc_model.predict(transformed_text)

#     if prediction == 'positive':
#         st.success("The review is Positive! 🎉")
#     else:
#         st.error("The review is Negative! 😞")


import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from tqdm import tqdm
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)

with open('svc_model.pkl', 'rb') as file:
    svc_model = pickle.load(file)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

df_review = pd.read_csv('df_review.csv')

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Sentiment Prediction", "EDA"])
    
    if page == "Sentiment Prediction":
        sentiment_prediction_page()
    elif page == "EDA":
        eda_page()

def sentiment_prediction_page():
    st.title("IMDB Movie Review Sentiment Analysis")
    st.write("This app classifies IMDB movie reviews as positive or negative.")
    
    review_text = st.text_area("Enter a movie review")
    
    if st.button("Predict Sentiment"):
        transformed_text = tfidf_vectorizer.transform([review_text])
        prediction = svc_model.predict(transformed_text)
        
        if prediction == 'positive':
            st.success("The review is Positive! 🎉")
        else:
            st.error("The review is Negative! 😞")

def eda_page():
    st.title("Exploratory Data Analysis")
    
    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='sentiment', data=df_review, ax=ax)
    st.pyplot(fig)
    
    # Word Frequency
    st.subheader("Word Frequency")
    sentiment = st.selectbox("Choose sentiment", ["positive", "negative"])
    
    df_sentiment = df_review[df_review['sentiment'] == sentiment]
    words_count = pd.DataFrame(count_words(df_sentiment['review']), columns=['Words', 'Count'])
    
    fig = px.bar(data_frame=words_count[:10], x='Words', y='Count', title=f'{sentiment.capitalize()} Word Frequency', color='Words')
    st.plotly_chart(fig)
    
    # Word Cloud
    st.subheader("Word Cloud")
    reviews = " ".join(df_sentiment['review'].values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'WordCloud - {sentiment.capitalize()} Reviews')
    st.pyplot(fig)

def count_words(data):
    words_counts = []
    for text in data:
        words = text.lower().split()
        words_counts.extend(words)
    return Counter(words_counts).most_common()

if __name__ == "__main__":
    main()