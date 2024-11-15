import requests
import re
import emoji
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess
from collections import Counter
import streamlit as st
import matplotlib.pyplot as plt
def beforefetch(link):
    return link.split('/')[-1]
bearer_token = "Enter your bearer token"
def fetchtweet(bearer_token,tweet_id,no_of_tweets):
    url = f"https://api.twitter.com/2/tweets/search/recent?query=conversation_id:{tweet_id}"
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    params = {
        "max_results": no_of_tweets,
        "expansions": "author_id",
    }
    response = requests.get(url,headers=headers,params=params)
    if response.status_code == 200:
        print("Request successful!")
        replies=response.json()
    else:
        print("Request failed with status code", response.status_code)
        replies=None
    return replies

def postfetch(replies):
    res = []
    for i in replies['data']:
        res.append(i['text'])
    res1 = [i.split() for i in res]
    res1 = [word for i in res1 for word in i]
    # return Counter(res1).most_common()
    out=[]
    for i in [i[0] for i in Counter(res1).most_common()]:
        if '@' in list(i):
            out.append(i)
    author=out[0]
    out=[]
    for i in res:
        out.append(" ".join(emoji.replace_emoji(re.sub(r'http[s]?://\S+', '', i)).split()[1:]))
    cleaned_comments= [name for name in out if name.strip() != '']
    return cleaned_comments
def SentimentAnalysis(comment):
    preprocessed_comment = preprocess(comment)
    new_comment = token_mod.transform([preprocessed_comment]) 
    return new_comment
with open("token_model.pkl","rb") as f:
    token_mod=pickle.load(f)
with open("model.pkl","rb") as f:
    model=pickle.load(f)
class_labels = ['Positive', 'Neutral', 'Irrelevant', 'Negative']

# output = {'A': 10, 'B': 20, 'C': 30, 'D': 40}
st.header("Tweets response classification and analysis")
link=st.text_input("Enter the Link of the tweet")
no_of_tweets=st.text_input("Enter the number of tweets to fetch")
if st.button("Predict",type="primary"):
    if int(no_of_tweets) >=25:
        st.error('Number pf tweets mustbe less than 25', icon="🚨")
    else:
        tweet_id=beforefetch(link)
        replies=fetchtweet(tweet_id=tweet_id,no_of_tweets=no_of_tweets,bearer_token=bearer_token)
        res=[]
        comments=postfetch(replies)
        for i in comments:
            prediction = model.predict(SentimentAnalysis(i))
            res.append(class_labels[prediction[0]-1])
        output=Counter(res)
        fig, ax = plt.subplots(2, 1) 
        ax[0].bar(output.keys(), output.values())
        ax[1].pie(output.values(), labels=output.keys())
        st.dataframe(pd.DataFrame([output]))
        st.pyplot(fig)