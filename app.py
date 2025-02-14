import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px

openai_api_key = st.text_input("Enter API Key")

st.title("🥗 Customer Review Sentiment Analyzer ")
st.markdown("This app analyzes the sentiment of customer reviews to gain insights into their opinions.")

#import pandas as pd
#df=pd.read_csv("reviews.csv")
#st.write(df)

def classify_sentiment_openai(review_text):
    client = OpenAI(api_key=openai_api_key)
    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 

    return completion.choices[0].message.content

csv_input = st.file_uploader(
    "Upload a CSV file with customer reviews", type=["csv"]
)

if csv_input is not None:
    reviews_df = pd.read_csv(csv_input)
    text_columns = reviews_df.select_dtypes(include="object").columns
    if len(text_columns) == 0:
        st.error("No text column found")

    review_columns = st.selectbox("Select thje columns with customer reviews",text_columns)

    reviews_df["sentiment"] = reviews_df[review_columns].apply(classify_sentiment_openai)
    #st.write(reviews_df)
    reviews_df["sentiment"] = reviews_df["sentiment"].str.title()
    sentiment_counts = reviews_df["sentiment"].value_counts()
    #massage data
    #st.title("Sentiment")
    #st.write(sentiment_counts)
    #st.write(reviews_df)
    col1, col2, col3 = st.columns(3)
    with col1:
        positive_count = sentiment_counts.get("Positive",0)
        st.metric("Positive",positive_count, f"{positive_count / len(reviews_df) * 100:.2f}%")

    with col2:
        negative_count = sentiment_counts.get("Negative",0)
        st.metric("Negative",negative_count, f"{negative_count / len(reviews_df) * 100:.2f}%")
    
    with col3:
        neutral_count = sentiment_counts.get("Neutral",0)
        st.metric("Neutral",neutral_count, f"{neutral_count / len(reviews_df) * 100:.2f}%")
     
    fig = px.pie(
        values = sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution")

    st.plotly_chart(fig)


# Example usage

#st.write(classify_sentiment_openai(user_input))
