import re
import warnings
import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocessor as p
import seaborn as sns
import streamlit as st
from h2o.automl import H2OAutoML
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('wordnet')

def convert_values(*args):
    converted_values = []

    for arg in args:
        try:
            if isinstance(arg, str):
                value = float(arg) if '.' in arg else int(arg)
                converted_values.append(value)
            else:
                converted_values.append(arg)
        except ValueError:
            print(f"Invalid input: {arg}. Please provide a valid numeric value.")

    if len(converted_values) == 1:
        return converted_values[0]
    else:
        return converted_values

@st.cache_resource(show_spinner = False)
def build_model(selected_model, dataset, *params):
    model_functions = {
        "H2O (AutoML)": buildH2OAutoML,
        "Linear Regression": buildLinearRegression,
        "Gradient Boosting": buildGradientBoosting,
        "Random Forest": buildRandomForest,
        "LSTM": buildLSTMModel
    }
    default_params = {
        "H2O (AutoML)": (0.8,600),
        "Linear Regression": (0.8,),
        "Gradient Boosting": (0.8, 42, 100, 0.1),
        "Random Forest": (100, 0.8),
        "LSTM": (0.8, 3, 200, 32, 0.1)
    }

    if selected_model in model_functions:
        model_func = model_functions[selected_model]
        default_param_values = default_params[selected_model]
        filled_params = [default_value if param in (None, '') else param for param, default_value in
                         zip(params, default_param_values)]
        result_df = model_func(dataset, *filled_params)
        return result_df
    else:
        raise ValueError("Selected model not recognized")


def buildH2OAutoML (dataset, split_ratio = 0.8, max_run_time_h2o = 600):
    split_ratio = convert_values(split_ratio)
    h2o.init(nthreads=-1)
    df_train = dataset.loc[:int(dataset.shape[0] * split_ratio), :]
    df_test = dataset.loc[int(dataset.shape[0] * split_ratio):, :]
    hf_train = h2o.H2OFrame(df_train)
    hf_test = h2o.H2OFrame(df_test)
    y = 'close'
    X = hf_train.columns
    X.remove(y)
    aml = H2OAutoML(max_runtime_secs=max_run_time_h2o,
                    seed=42)
    aml.train(x=X,
              y=y,
              training_frame=hf_train,
              leaderboard_frame=hf_test)
    leader_model = aml.leader
    hf_test_predict = leader_model.predict(hf_test)
    df_results = pd.DataFrame()
    df_results['ground_truth'] = df_test['close'].reset_index(drop=True)
    df_results['date'] = df_test['date'].reset_index(drop=True)
    df_results['predictions'] = h2o.as_list(hf_test_predict, use_pandas=True)
    return df_results, leader_model

def buildLinearRegression (dataset, split_ratio=0.8):
    split_ratio = convert_values(split_ratio)
    dataset['date'] = pd.to_datetime(dataset['date'])
    split_index = int(len(dataset) * split_ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    X_train = train_data.drop(columns=['close', 'date'])
    X_test = test_data.drop(columns=['close', 'date'])
    y_train = train_data['close']
    y_test = test_data['close']
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_df = pd.DataFrame({
        'date': test_data['date'],
        'ground_truth': y_test,
        'predictions': y_pred
    })

    return results_df, model

def buildGradientBoosting(dataset, split_ratio=0.8, random_state=42, n_estimators=100, learning_rate=0.1):
    split_ratio, random_state, n_estimators, learning_rate = convert_values(split_ratio, random_state, n_estimators, learning_rate)
    dataset['date'] = pd.to_datetime(dataset['date'])
    split_index = int(len(dataset) * split_ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    X_train = train_data.drop(columns=['close', 'date'])
    X_test = test_data.drop(columns=['close', 'date'])
    y_train = train_data['close']
    y_test = test_data['close']
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_df = pd.DataFrame({
        'date': test_data['date'],
        'ground_truth': y_test,
        'predictions': y_pred
    })
    return results_df, model


def buildRandomForest(dataset, n_estimators=100, split_ratio=0.8):
    n_estimators, split_ratio = convert_values(n_estimators, split_ratio)
    df = dataset.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    train_size = int(len(df) * split_ratio)
    train_data, test_data = df[:train_size], df[train_size:]
    X_train, y_train = train_data.drop(columns=['close', 'date']), train_data['close']
    X_test, y_test = test_data.drop(columns=['close', 'date']), test_data['close']
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    test_dates = test_data['date'].values
    results_df = pd.DataFrame({'date': test_dates, 'ground_truth': y_test.values, 'predictions': predictions})
    return results_df, model

def buildLSTMModel(dataset, split_ratio=0.8, sequence=3,
                   epoch=200, batch_size=32, validation_split=0.1):
    split_ratio, sequence, epoch, batch_size, validation_split = convert_values(split_ratio, sequence, epoch,
                                                                                batch_size, validation_split)
    feature_columns = dataset.columns.tolist()
    feature_columns.remove('close')
    feature_columns.remove('date')
    X = dataset[feature_columns].values
    y = dataset['close'].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - sequence + 1):
        X_sequences.append(X[i: i + sequence])
        y_sequences.append(y[i + sequence - 1])
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    split_index = int(split_ratio * len(X_sequences))
    X_train = X_sequences[:split_index]
    y_train = y_sequences[:split_index]
    X_test = X_sequences[split_index:]
    y_test = y_sequences[split_index:]
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence, X.shape[1])))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=validation_split)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    dates = pd.to_datetime(dataset['date'].iloc[split_index + sequence - 1:]).values
    results_df = pd.DataFrame({'ground_truth': y_test_actual.flatten(), 'date': dates, 'predictions': y_pred.flatten()})
    return results_df, model

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
    tokenized_tweet = tweets['clean_tweet'].apply(lambda sentence: sentence.split())
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


st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('CryptoEval Model Evaluation Dashboard')
st.markdown('Please refer to the tutorial in the following Github link')
st.markdown('https://github.com/Dwwy/FYP')
build_disabled = False
df = None
col1, col2 = st.columns(2)
with col1:
    news_csv = st.file_uploader("Upload a Bitcoin Tweet CSV file", type=["csv"])
if news_csv is not None:
    with st.spinner('Retrieving News Data...'):
        news = load_data(news_csv)
with col2:
    price_csv = st.file_uploader("Upload a Bitcoin Price CSV file", type=["csv"])

if price_csv is not None:
    with st.spinner('Retrieving Price Data...'):
        price = load_data(price_csv)

selected_model = st.selectbox(
    "Select Algorithm",
    ["Select an Algorithm", "H2O (AutoML)", "LSTM", "Linear Regression", "Gradient Boosting", "Random Forest"]
)

defaults = {
    "split_ratio_h2o": 0.8,
    "max_run_time_h2o": 600,
    "split_ratio_lstm": 0.8,
    "sequence_lstm": 3,
    "epoch_lstm": 200,
    "batch_size_lstm": 32,
    "validation_split_lstm": 0.1,
    "split_ratio_lr": 0.8,
    "split_ratio_gb": 0.8,
    "random_state_gb": 42,
    "n_estimator_gb": 100,
    "learning_rate_gb": 0.1,
    "split_ratio_rf": 0.8,
    "n_estimator_rf": 100
}

missing_defaults = [param for param, default_value in defaults.items() if param not in st.session_state]

for param in missing_defaults:
    st.session_state[param] = defaults[param]

if 'app_state' not in st.session_state:
    st.session_state.app_state = 'initial'
def reset_number_input():
    for param, default_value in defaults.items():
        st.session_state[param] = default_value

if selected_model != "Select an Algorithm":
    st.write(f"Model: {selected_model}")
    button = st.button("Default", on_click=reset_number_input)
split_ratio_description = "Percentage for train data (For algorithm to learn the pattern)"
if selected_model == "H2O (AutoML)":
    col1, col2 = st.columns(2)
    with col1:
        split_ratio_h2o = st.number_input("Split Ratio:", key="split_ratio_h2o", min_value=0.1, max_value=0.9, step=0.1)
        st.markdown(split_ratio_description)
    with col2:
        max_run_time_h2o = st.slider("Maximum Runtime (seconds):", key="max_run_time_h2o", min_value=10, max_value=1000, step=10)
        st.markdown("Time for the algorithm to run and look for the best performance")

elif selected_model == "LSTM":
    col1, col2, col3 = st.columns(3)
    with col1:
        split_ratio_lstm = st.number_input("Split Ratio:", key="split_ratio_lstm", min_value=0.1, max_value=0.9, step=0.1)
        st.markdown(split_ratio_description)
    with col2:
        sequence_lstm = st.number_input("Sequence:", key="sequence_lstm", min_value=2, max_value=10, step=1)
        st.markdown("Number for sliding window (For example, combine 3 rows of data to predict the 4th row)")
    with col3:
        epoch_lstm = st.number_input("Epoch:", key="epoch_lstm", min_value=10, max_value=300, step=10)
        st.markdown("Number of times the algorithm process the data")
    col1, col2 = st.columns(2)
    with col1:
        batch_size_lstm = st.slider("Batch Size:", key="batch_size_lstm", min_value=1, max_value=128, step=1)
        st.markdown("Number of samples processed together in each epoch (Often divisible by 2)")
    with col2:
        validation_split_lstm = st.number_input("Validation Split:", key="validation_split_lstm", min_value=0.1, max_value=1.0, step=0.1)
        st.markdown("Percentage of train data set aside to serve as a secondary validation set to evaluate the model performance")


elif selected_model == "Linear Regression":
    split_ratio_lr = st.number_input("Split Ratio:", key="split_ratio_lr", min_value=0.1, max_value=0.9, step=0.1)
    st.markdown(split_ratio_description)
elif selected_model == "Gradient Boosting":
    col1, col2 = st.columns(2)
    with col1:
        n_estimator_gb = st.slider("n_estimator:", key="n_estimator_gb", min_value=1, max_value=1000, step=10)
        st.markdown("Number of learning trees to be included while building the model")
    with col2:
        random_state_gb = st.slider("Random State:", key="random_state_gb", min_value=0, max_value=1000, step=2)
        st.markdown("Controls the randomness of subsampling data during training. (Value 42 is often used)")
    col1, col2 = st.columns(2)
    with col1:
        split_ratio_gb = st.number_input("Split Ratio:", key="split_ratio_gb", min_value=0.1, max_value=0.9, step=0.1)
        st.markdown(split_ratio_description)
    with col2:
        learning_rate_gb = st.slider("Learning Rate:", key="learning_rate_gb", min_value=0.01, max_value=1.0, step=0.01)
        st.markdown("Controls the step size at which the algorithm updates the model. (The smaller the more accurate and more time taken)")
elif selected_model == "Random Forest":
    col1, col2 = st.columns(2)
    with col1:
        split_ratio_rf = st.number_input("Split Ratio:", key="split_ratio_rf", min_value=0.1, max_value=0.9, step=0.1)
        st.markdown(split_ratio_description)
    with col2:
        n_estimator_rf = st.slider("n_estimator:", key="n_estimator_rf", min_value=1, max_value=1000, step=10)
        st.markdown("Number of learning trees to be included while building the model")


@st.cache_data(show_spinner=False)
def showStatistics(df_results):
    mae = mean_absolute_error(df_results['ground_truth'], df_results['predictions'])
    mse = mean_squared_error(df_results['ground_truth'], df_results['predictions'])
    rmse = mse ** 0.5
    r2 = r2_score(df_results['ground_truth'], df_results['predictions'])
    st.header('Evaluation Metrics')
    st.write(f'MAE: {mae:.2f} (Average deviation of predicted value from actual value)')
    st.write(f'MSE: {mse:.2f} (Average squared difference between predicted and actual value)')
    st.write(f'RMSE: {rmse:.2f} (Square root of MSE, providing an average magnitude of errors)')
    st.write(f'R2 Score: {r2:.2f} (The degree to which the input variable explains the target variable)')
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    with row1_col1:
        st.header('Q-Q Plot')
        st.markdown('Describes the distribution of the predicted value')
        st.markdown('Optimally the points should gather at the diagonal line')
        fig, ax = plt.subplots()
        stats.probplot(df_results['predictions'], plot=ax)
        st.pyplot(fig)
    with row1_col2:
        st.header('Scatter Plot: Actual vs. Predicted')
        st.markdown('Plots the actual value and the predicted value together')
        st.markdown('Optimally the points should gather to form a diagonal line')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='ground_truth', y='predictions', data=df_results)
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        st.pyplot()
    with row2_col1:
        st.header('Residual Plot')
        st.markdown('Plots the difference of actual value and the predicted value')
        st.markdown('Optimally the points should gather at y=0 to show a low difference')
        residuals = df_results['predictions'] - df_results['ground_truth']
        plt.figure(figsize=(8, 6))
        sns.residplot(x=df_results['ground_truth'], y=residuals, lowess=True)
        plt.xlabel('Actual Value')
        plt.ylabel('Residuals')
        st.pyplot()
    with row2_col2:
        st.header('Histogram of Residuals')
        st.markdown('Plots the difference of actual value and the predicted value')
        st.markdown('Optimally the bar at 0 should be the highest, showing a that the occurrence of 0 in difference is the highest')
        residuals = df_results['predictions'] - df_results['ground_truth']
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=20, edgecolor='k')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        st.pyplot()
    if 'date' in df_results.columns:
        st.header('Time Series Plot: Actual vs. Predicted')
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['date'], df_results['ground_truth'], label='Actual')
        plt.plot(df_results['date'], df_results['predictions'], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        st.pyplot()
    st.header('Data Table')
    st.dataframe(df_results)
build = st.button('Build', key="build_button", disabled=build_disabled)
if build or (st.session_state.app_state != 'initial' and st.session_state.app_state == selected_model):
    if price_csv is None or news_csv is None:
        st.error("Please upload both csv file...")
    else:
        if selected_model == "Select an Algorithm":
            st.error("Please choose a model and configure the parameters")
        else:
            with st.spinner('Building Model...'):
                df = retrieve_data(price, news)
                if selected_model == "H2O (AutoML)":
                    df_results, model = build_model(selected_model, df, split_ratio_h2o, max_run_time_h2o)
                elif selected_model == "LSTM":
                    feature_columns = ['open', 'p_neg', 'p_neu', 'p_pos', 'p_comp', 'count', 'Volume', 'Volume MA']
                    df_results, model = build_model(selected_model, df, split_ratio_lstm, sequence_lstm, epoch_lstm, batch_size_lstm, validation_split_lstm)
                elif selected_model == "Linear Regression":
                    df_results, model = build_model(selected_model, df, split_ratio_lr)
                elif selected_model == "Gradient Boosting":
                    df_results, model = build_model(selected_model, df, split_ratio_gb, random_state_gb, n_estimator_gb, learning_rate_gb)
                elif selected_model == "Random Forest":
                    df_results, model = build_model(selected_model, df, n_estimator_rf, split_ratio_rf)
                st.session_state.app_state = selected_model
                st.sidebar.title("Mock Prediction")
                st.sidebar.header("Enter Mock Input Data")
                open_price_text = st.sidebar.number_input("Open Price", value=0)
                volume_text = st.sidebar.number_input("Volume", value=0)
                volume_ma_text = st.sidebar.number_input("Volume Moving Average", value=0)
                st.sidebar.markdown("Highest price reached on that day")
                high_text = st.sidebar.number_input("High", value=0)
                st.sidebar.markdown("Lowest price reached on that day")
                low_text = st.sidebar.number_input("Low", value=0)
                st.sidebar.markdown("Negative, positive and neutral are decimals that should sums up to 1")
                p_neg_text = st.sidebar.number_input("Negative Sentiment", value=0.0, step=0.01)
                p_pos_text = st.sidebar.number_input("Positive Sentiment", value=0.0, step=0.01)
                p_neu_text = st.sidebar.number_input("Neutral Sentiment", value=0.0, step=0.01)
                open_price = float(open_price_text)
                volume = float(volume_text)
                volume_ma = float(volume_ma_text)
                high = float(high_text)
                low = float(low_text)
                p_neg = float(p_neg_text)
                p_pos = float(p_pos_text)
                p_neu = float(p_neu_text)
                p_comp = (p_pos-p_neg) + p_neu
                if st.sidebar.button("Predict"):
                    mock_input = {
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "Volume": volume,
                        "Volume MA": volume_ma,
                        "p_neg": p_neg,
                        "p_neu": p_neu,
                        "p_pos": p_pos,
                        "p_comp": p_comp,
                        "count": 250
                    }
                    mock_input_df = pd.DataFrame([mock_input])
                    if selected_model == "H2O (AutoML)":
                        latest_date = df['date'].max()
                        latest_date = pd.to_datetime(latest_date)
                        next_day = latest_date + pd.DateOffset(days=1)
                        next_day_formatted = next_day.strftime('%Y-%m-%d')
                        mock_input_df['date'] = next_day_formatted
                        order = ["open", "high", "low", "Volume", "Volume MA", "date", "p_neg", "p_neu", "p_pos", "p_comp", "count"]
                        mock_input_df = mock_input_df[order]
                        mock_input_df = h2o.H2OFrame(mock_input_df)
                    elif selected_model == "LSTM":
                        scaler = MinMaxScaler()
                        mock_input_data_scaled = scaler.fit_transform(mock_input_df.values)
                        mock_input_df = np.array([mock_input_data_scaled] * sequence_lstm)
                    predicted_output = model.predict(mock_input_df)
                    st.sidebar.subheader("Prediction:")
                    if selected_model == "H2O (AutoML)":
                        predicted_value = predicted_output.as_data_frame().to_numpy()[0][0]
                        st.sidebar.write(f"Predicted Output: {predicted_value:.2f}")
                    elif selected_model == "LSTM":
                        scaler.fit(df[:int(split_ratio_lstm*len(df.values))]['close'].values.reshape(-1, 1))
                        predicted_value = scaler.inverse_transform(predicted_output.reshape(-1, 1))
                        st.sidebar.write(f"Predicted Output: {predicted_value[0, 0]:.2f}")
                    else:
                        st.sidebar.write(f"Predicted Output: {predicted_output[0]:.2f}")
                showStatistics(df_results)

