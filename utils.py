import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import tensorflow as tf
from datetime import datetime, timedelta

# Directory managment 
import os

def convert_to_daily(data, date_list, df_columns):
    date_list = [a[0] for a in date_list] # get the first future date we can 
    data = data[:,0, :] # just get the first future value
    df = pd.DataFrame(data, index=date_list, columns=df_columns)
    df = df.groupby(df.index.strftime("%Y%m%d")).mean()
    df.index = pd.to_datetime(df.index)
    return df


def compute_squared_error(gt_data_df, test_data_df):
    return np.square(np.subtract(gt_data_df, test_data_df))
    
    

def load_csv(csv_path):
    """
    Args:
        csv_path : input csv path to load and format the data frame
    Return:
        df: loaded data frame
    """
    try:
        # read data
        df = pd.read_csv(csv_path)
        
        # format datetime and load as datetime object
        df['DATETIME'] = [
            datetime.strptime(x, '%d/%m/%Y %H:%M') for x in df['DATETIME']
        ]
        # Making sure there are no duplicated data
        # If there are some duplicates we average the data during those duplicated days
        df = df.groupby('DATETIME',
                        as_index=True)['TEMPERATURE',
                                        'FORECASTDEMAND', 'DAYLIGHT',
                                        'SGU_Rated_Output_In_kW',
                                        'Cumulative Installations', 'RAINFALL',
                                        'TOTALDEMAND'].mean()
        # Sorting the values
        df.sort_values('DATETIME', inplace=True)
    except Exception as e:
        print("Exception in loading data Error - {}".format(e))
    return df

def scale_data(df, scalers = {}):
    """
    Args:
        df: dataframe to scale
        scalers: scalers use to normalise the input (if not given computed from the input df)
    Return:
        scaled_df: scaled data frame
        scalers: scalers used/ computed for the scaling
    """
    scaled_df = df.copy()
    for i in df.columns:
        scaler_name = 'scaler_' + i
        if scaler_name in scalers: # test data
            s_s = scalers[scaler_name].transform(df[i].values.reshape(-1,1))
        else: # train data
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scalers[scaler_name] = scaler
            s_s = scaler.fit_transform(df[i].values.reshape(-1, 1))
        s_s = np.reshape(s_s, len(s_s))
        scaled_df[i] = s_s
    return scaled_df.values, scalers, scaled_df.index.values
    

def split_series(series, n_past, n_ahead, time_list=None):
    """
    Args:
        series: input df data values
        n_past : number of past data to use for the sliding window
        n_ahead : number of feature data to use 
    Return:
        splitted slided time series past and ahead data
    """
    X, y = list(), list()
    splitted_time_list = []
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_ahead
        if future_end > len(series):
            break
        if time_list is not None:
            splitted_time_list.append(time_list[past_end:future_end])
            
            
        # TODO need to check the time difference and skip the gap (data is not consecutive)
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[
            past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y), splitted_time_list


def inverse_scaled_data(scaled_df, scalers):
    for index, i in enumerate(scalers["columns"]):
        scaler = scalers['scaler_'+ i]
        scaled_df[:,:,index] = scaler.inverse_transform(scaled_df[:,:,index])
    return scaled_df

def TSmodel(n_past, n_ahead, n_features, n_lstmLayers):
    # encoder network
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    encoder_l1 = tf.keras.layers.LSTM(n_lstmLayers, return_sequences = True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = tf.keras.layers.LSTM(n_lstmLayers, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]
    # decoder network
    decoder_inputs = tf.keras.layers.RepeatVector(n_ahead)(encoder_outputs2[0])
    decoder_l1 = tf.keras.layers.LSTM(n_lstmLayers, return_sequences=True)(decoder_inputs, initial_state = encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(n_lstmLayers, return_sequences=True)(decoder_l1, initial_state = encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
    
    # setup the model with input and outpu
    model = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
    # print the summary of the model
    model.summary()
    
    return model