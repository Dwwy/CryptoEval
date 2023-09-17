import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from utils import retrieveData, model

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('CryptoEval Model Evaluation Dashboard')
build_disabled = False
df = None
col1, col2 = st.columns(2)
with col1:
    news_csv = st.file_uploader("Upload a Bitcoin Tweet CSV file", type=["csv"])
if news_csv is not None:
    with st.spinner('Retrieving News Data...'):
        news = retrieveData.load_data(news_csv)
with col2:
    price_csv = st.file_uploader("Upload a Bitcoin Price CSV file", type=["csv"])

if price_csv is not None:
    with st.spinner('Retrieving Price Data...'):
        price = retrieveData.load_data(price_csv)

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
                df = retrieveData.retrieve_data(price, news)
                if selected_model == "H2O (AutoML)":
                    df_results, model = model.build_model(selected_model, df, split_ratio_h2o, max_run_time_h2o)
                elif selected_model == "LSTM":
                    feature_columns = ['open', 'p_neg', 'p_neu', 'p_pos', 'p_comp', 'count', 'Volume', 'Volume MA']
                    df_results, model = model.build_model(selected_model, df, split_ratio_lstm, sequence_lstm, epoch_lstm, batch_size_lstm, validation_split_lstm)
                elif selected_model == "Linear Regression":
                    df_results, model = model.build_model(selected_model, df, split_ratio_lr)
                elif selected_model == "Gradient Boosting":
                    df_results, model = model.build_model(selected_model, df, split_ratio_gb, random_state_gb, n_estimator_gb, learning_rate_gb)
                elif selected_model == "Random Forest":
                    df_results, model = model.build_model(selected_model, df, n_estimator_rf, split_ratio_rf)
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

