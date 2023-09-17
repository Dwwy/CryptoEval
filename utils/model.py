import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import h2o
from h2o.automl import H2OAutoML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import streamlit as st
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
