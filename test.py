import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import tensorflow as tf
from datetime import datetime, timedelta

# Directory managment 
import os
import conf
from utils import load_csv, scale_data, split_series, TSmodel

def load_and_preprocess_data():
    # load train_data
    train_df  = load_csv(conf.train_csv)
    val_df  = load_csv(conf.val_csv)
    test_df  = load_csv(conf.test_csv)
    scaled_train_df, scalers = scale_data(train_df)
    scaled_test_df, _ = scale_data(train_df, scalers)
    scaled_val_df, _ = scale_data(train_df, scalers)
    train_x, train_y = split_series(scaled_train_df, conf.n_past, conf.n_ahead)
    val_x, val_y = split_series(scaled_val_df, conf.n_past, conf.n_ahead)
    test_x, test_y = split_series(scaled_test_df, conf.n_past, conf.n_ahead)
    return train_x, train_y, val_x, val_y, test_x, test_y, scalers
    
def main():
    # set a new log sub dir for the current training
    curr_log_dir = os.path.join(conf.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(curr_log_dir)
    
    train_x, train_y, val_x, val_y, test_x, test_y, scalers = load_and_preprocess_data()
    ts_model = TSmodel(conf.n_past, conf.n_ahead, conf.n_features, conf.n_lstmLayers)
    ts_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    model_path = os.path.join(conf.model_dir, "tsmodel-{epoch:03d}-{val_loss:03f}.h5")
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    cb_tensorboad = tf.keras.callbacks.TensorBoard(log_dir=curr_log_dir)
    cb_reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
    call_backs = [cb_checkpoint,cb_tensorboad, cb_reduce_lr]
    
    history = ts_model.fit(train_x, train_y, epochs=conf.n_train_epochs, validation_data=(test_x, test_y), batch_size=conf.train_batchSize, verbose=2, callbacks=call_backs)
    # now saving the history data to plot some stats in future
    np.save(os.path.join(curr_log_dir, "train_history.npy", history.history))

if __name__ == "__main__":
    main()
