import pymongo
import pandas as pd
import preprocessor as p
import re
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import warnings
import streamlit as st

apostrophe = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}


# Emotion detection by different symbols
emoji = {
":)": "happy",
":‑)": "happy",
":-]": "happy",
":-3": "happy",
":->": "happy",
"8-)": "happy",
":-}": "happy",
":o)": "happy",
":c)": "happy",
":^)": "happy",
"=]": "happy",
"=)": "happy",
"<3": "happy",
":-(": "sad",
":(": "sad",
":c": "sad",
":<": "sad",
":[": "sad",
">:[": "sad",
":{": "sad",
">:(": "sad",
":-c": "sad",
":-< ": "sad",
":-[": "sad",
":-||": "sad"
}
def emotion_check(text):
    for key in emoji:
        value = emoji[key]
        text = text.replace(key, value)
        return text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt.lower()
def apostrophe_check(text):
    for key in apostrophe:
        value = apostrophe[key]
        text = text.replace(key, value)
        return text
def clean(input_str):
    p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.MENTION, p.OPT.HASHTAG)
    input_str = p.clean(input_str)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", input_str).split())

analyser = SentimentIntensityAnalyzer()
def calculate_sentiment_scores(text):
    sentiment_scores = analyser.polarity_scores(text)
    return pd.Series([sentiment_scores['neg'], sentiment_scores['neu'],
                      sentiment_scores['pos'], sentiment_scores['compound']])

@st.cache_data(show_spinner = False)
def load_data(path):
    df = pd.read_csv(path)
    return df
@st.cache_data(show_spinner = False)
def retrieve_data(price, news):
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = news.copy()
    df1 = price.copy()
    df = df.dropna()
    df["cleaned"] = df["translated_text"].apply(clean)
    mask = df["cleaned"].apply(lambda x: x in (None, '') or pd.isna(x))
    tweets = df[-mask]
    tweets['clean_tweet'] = np.vectorize(remove_pattern)(tweets['cleaned'], "@[\w]*")
    tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x: apostrophe_check(x))
    tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x: emotion_check(x))
    tweets['clean_tweet'] = tweets['clean_tweet'].str.replace("[^a-zA-Z]", " ")
    tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
    tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x: re.sub(r"\s+", " ", x))
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    # Tokenize the tweets (replace this with your tokenization logic)
    tokenized_tweet = tweets['clean_tweet'].apply(lambda sentence: sentence.split())
    tokenized_tweet
    cleaned_tweets = []
    for sentence_tokens in tokenized_tweet:
        cleaned_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in sentence_tokens]
        cleaned_tweets.append(' '.join(cleaned_tokens))
    cleaned_df = pd.DataFrame({'stemmed': cleaned_tweets})
    tweets.reset_index(drop=True, inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)
    tweets = pd.concat([tweets, cleaned_df], axis=1)
    sentiment_scores_df = tweets['stemmed'].apply(calculate_sentiment_scores)
    tweets[['p_neg', 'p_neu', 'p_pos', 'p_comp']] = sentiment_scores_df
    tweets['date'] = pd.to_datetime(tweets['date'], format='%b %d, %Y %H:%M:%S')
    df1['time'] = pd.to_datetime(df1['time'])
    df1['time'] = df1['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    tweets['date'] = tweets['date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    tweets['date'] = pd.to_datetime(tweets['date'])
    tweets['count'] = tweets.groupby(tweets['date'].dt.date)['content'].transform('count')
    extracted = tweets[['date', 'p_neg','p_neu','p_pos','p_comp', 'count']]
    extracted['date_only'] = extracted['date'].dt.date
    grouped_df = extracted.groupby('date_only').agg({
        'p_neg': 'mean',
        'p_neu': 'mean',
        'p_pos': 'mean',
        'p_comp': 'mean',
        'date': 'count'
    }).reset_index()
    grouped_df.rename(columns={'date_only': 'date', 'date': 'count'}, inplace=True)
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])
    date_to_remove = pd.to_datetime("2021-07-03").date()
    grouped_df = grouped_df[grouped_df['date'].dt.date != date_to_remove]
    start_date = pd.to_datetime("2021-08-01").date()
    end_date = pd.to_datetime("2023-08-13").date()
    df1['time'] = pd.to_datetime(df1['time'])
    df_filtered = df1[(df1['time'].dt.date >= start_date) & (df1['time'].dt.date <= end_date)]
    df_filtered['date'] = df_filtered['time'].dt.date
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    merged_df = pd.merge(df_filtered,grouped_df, on='date', how='inner')
    dataset = merged_df
    dataset = dataset.drop(['time'], axis=1)
    dataset['date'] = pd.to_datetime(dataset['date'])
    warnings.resetwarnings()
    return dataset
