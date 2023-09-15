import streamlit as st

if 'number_input' not in st.session_state:
    st.session_state.number_input = 0.1
defaults = {
    "split_ratio_h2o": 0.8,
    "split_ratio_lstm": 0.8,
    "sequence_lstm": 3,
    "epoch_lstm": 200,
    "batch_size_lstm": 32,
    "validation_split_lstm": 0.1,
    "split_ratio_lr": 0.8,
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


split_ratio_h2o = st.number_input("Split Ratio:", min_value=0.1, max_value=0.9, step=0.1, key= "split_ratio_h2o")
button = st.button("Button", on_click=reset_number_input)