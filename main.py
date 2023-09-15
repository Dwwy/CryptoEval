import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from streamlit.runtime.state import SessionState
from streamlit import session_state
import retrieveData
import time
import model
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Model Evaluation Dashboard')
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
        # df = pd.read_csv('/Users/danielwong/Downloads/preprocessed.csv')
# if st.button('Process', key="process_button"):
#     if price_csv is not None and news_csv is not None:
#         with st.spinner('Processing Data...'):
#             df = retrieveData.retrieve_data(price, news)
#             build_disabled = False
#     else:
#         st.error("Please upload both csv file...")

selected_model = st.selectbox(
    "Select Model",
    ["Select a model", "H2O (AutoML)", "LSTM", "Linear Regression", "Gradient Boosting", "Random Forest", "ARIMA"]
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
    "n_estimator_rf": 100,
    "split_ratio_arima": 0.9,
    "forecast_periods_arima": 10,
    "p_arima": 1,
    "d_arima": 1,
    "q_arima": 1,
}

# Initialize session state variables for each parameter
missing_defaults = [param for param, default_value in defaults.items() if param not in st.session_state]

# Initialize missing session state variables
for param in missing_defaults:
    st.session_state[param] = defaults[param]
def reset_number_input():
    for param, default_value in defaults.items():
        st.session_state[param] = default_value



# Display text input fields based on the selected model
if selected_model != "Select a model":
    st.write(f"Model: {selected_model}")
    button = st.button("Default", on_click=reset_number_input)

if selected_model == "H2O (AutoML)":
    col1, col2 = st.columns(2)
    with col1:
        split_ratio_h2o = st.number_input("Split Ratio:", key="split_ratio_h2o", min_value=0.1, max_value=0.9, step=0.1)
    with col2:
        max_run_time_h2o = st.slider("Maximum Runtime (seconds):", key="max_run_time_h2o", min_value=10, max_value=1000, step=10)

elif selected_model == "LSTM":
    col1, col2, col3 = st.columns(3)
    with col1:
        split_ratio_lstm = st.number_input("Split Ratio:", key="split_ratio_lstm", min_value=0.1, max_value=0.9, step=0.1)
    with col2:
        sequence_lstm = st.number_input("Sequence:", key="sequence_lstm", min_value=2, max_value=10, step=1)
    with col3:
        epoch_lstm = st.number_input("Epoch:", key="epoch_lstm", min_value=10, max_value=300, step=10)
    col1, col2 = st.columns(2)
    with col1:
        batch_size_lstm = st.slider("Batch Size:", key="batch_size_lstm", min_value=1, max_value=128, step=1)
    with col2:
        validation_split_lstm = st.number_input("Validation Split:", key="validation_split_lstm", min_value=0.0, max_value=1.0, step=0.1)

    # Add more input fields for LSTM parameters if needed
elif selected_model == "Linear Regression":
    split_ratio_lr = st.number_input("Split Ratio:", key="split_ratio_lr", min_value=0.1, max_value=0.9, step=0.1)
elif selected_model == "Gradient Boosting":
    col1, col2 = st.columns(2)
    with col1:
        n_estimator_gb = st.slider("n_estimator:", key="n_estimator_gb", min_value=1, max_value=1000, step=10)
    with col2:
        random_state_gb = st.slider("Random State:", key="random_state_gb", min_value=0, max_value=1000, step=2)
    col1, col2 = st.columns(2)
    with col1:
        split_ratio_gb = st.number_input("Split Ratio:", key="split_ratio_gb", min_value=0.1, max_value=0.9, step=0.1)
    with col2:
        learning_rate_gb = st.number_input("Learning Rate:", key="learning_rate_gb", min_value=0.01, max_value=1.0, step=0.01)
elif selected_model == "Random Forest":
    col1, col2 = st.columns(2)
    with col1:
        split_ratio_rf = st.number_input("Split Ratio:", key="split_ratio_rf", min_value=0.1, max_value=0.9, step=0.1)
    with col2:
        n_estimator_rf = st.slider("n_estimator:", key="n_estimator_rf", min_value=1, max_value=1000, step=10)
elif selected_model == "ARIMA":
    col1, col2 = st.columns(2)
    with col1:
        split_ratio_arima = st.number_input("Split Ratio:", key="split_ratio_arima", min_value=0.1, max_value=0.9, step=0.1)
    with col2:
        forecast_periods_arima = st.number_input("Forecast Period:", key="forecast_periods_arima", min_value=1, max_value=20, step=1)
    col1, col2, col3 = st.columns(3)
    with col1:
        p_arima = st.number_input("P value:", key="p_arima", min_value=0, max_value=10, step=1)
    with col2:
        d_arima = st.number_input("D value:", key="d_arima", min_value=0, max_value=10, step=1)
    with col3:
        q_arima = st.number_input("Q value:", key="q_arima", min_value=0, max_value=10, step=1)

build = st.button('Build', key="build_button", disabled= build_disabled)

if build:
    if price_csv is None or news_csv is None:
        st.error("Please upload both csv file...")
    else:
        with st.spinner('Building Model...'):
            df = retrieveData.retrieve_data(price, news)
            # Linear Regression Best
            # Load your data
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
            elif selected_model == "ARIMA":
                df_results = model.build_model(selected_model, df, p_arima, d_arima, q_arima, forecast_periods_arima, split_ratio_arima)

            # Calculate evaluation metrics
            mae = mean_absolute_error(df_results['ground_truth'], df_results['predictions'])
            mse = mean_squared_error(df_results['ground_truth'], df_results['predictions'])
            rmse = mse ** 0.5
            r2 = r2_score(df_results['ground_truth'], df_results['predictions'])

            # Create a Streamlit app

            # Display evaluation metrics
            st.header('Evaluation Metrics')
            st.write(f'MAE: {mae:.2f}')
            st.write(f'MSE: {mse:.2f}')
            st.write(f'RMSE: {rmse:.2f}')
            st.write(f'R2 Score: {r2:.2f}')

            # Create columns for rows
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            # Q-Q Plot
            with row1_col1:
                st.header('Q-Q Plot')
                fig, ax = plt.subplots()
                stats.probplot(df_results['predictions'], plot=ax)
                st.pyplot(fig)

            # Scatter Plot
            with row1_col2:
                st.header('Scatter Plot: Actual vs. Predicted')
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x='ground_truth', y='predictions', data=df_results)
                plt.xlabel('Actual Value')
                plt.ylabel('Predicted Value')
                st.pyplot()

            # Residual Plot
            with row2_col1:
                st.header('Residual Plot')
                residuals = df_results['predictions'] - df_results['ground_truth']
                plt.figure(figsize=(8, 6))
                sns.residplot(x=df_results['ground_truth'], y=residuals, lowess=True)
                plt.xlabel('Actual Value')
                plt.ylabel('Residuals')
                st.pyplot()

            # Histogram of Residuals
            with row2_col2:
                st.header('Histogram of Residuals')
                residuals = df_results['predictions'] - df_results['ground_truth']
                plt.figure(figsize=(8, 6))
                plt.hist(residuals, bins=20, edgecolor='k')
                plt.xlabel('Residuals')
                plt.ylabel('Frequency')
                st.pyplot()

            # Time Series Plot (if applicable)
            if 'date' in df_results.columns:
                st.header('Time Series Plot: Actual vs. Predicted')
                plt.figure(figsize=(10, 6))
                plt.plot(df_results['date'], df_results['ground_truth'], label='Actual')
                plt.plot(df_results['date'], df_results['predictions'], label='Predicted')
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot()

            # Display the data table
            st.header('Data Table')
            st.dataframe(df_results)
