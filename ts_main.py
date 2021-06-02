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
import pickle
from utils import load_csv, scale_data, split_series, TSmodel, inverse_scaled_data
register_matplotlib_converters()

def load_train_val_data():
    # load training and validation data
    train_df  = load_csv(conf.train_csv)
    val_df  = load_csv(conf.val_csv)
    scaled_train_df, scalers = scale_data(train_df)
    scaled_val_df, _ = scale_data(val_df, scalers)
    train_x, train_y = split_series(scaled_train_df, conf.n_past, conf.n_ahead)
    val_x, val_y = split_series(scaled_val_df, conf.n_past, conf.n_ahead)
    scalers["columns"] = train_df.columns
    with open(os.path.join(conf.model_dir, "scalers.pkl"), 'wb') as fp:
        pickle.dump(scalers, fp)

    return train_x, train_y, val_x, val_y
    
def train():
    # set a new log sub dir for the current training
    curr_log_dir = os.path.join(conf.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(curr_log_dir)
    
    train_x, train_y, val_x, val_y = load_train_val_data()
    ts_model = TSmodel(conf.n_past, conf.n_ahead, conf.n_features, conf.n_lstmLayers)
    ts_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    # model_path = os.path.join(conf.model_dir, "tsmodel-{epoch:03d}-{val_loss:03f}.h5")
    model_path = os.path.join(conf.model_dir, "tsmodel.h5") # it's good to overide the existing one
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    cb_tensorboad = tf.keras.callbacks.TensorBoard(log_dir=curr_log_dir)
    cb_reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
    call_backs = [cb_checkpoint,cb_tensorboad, cb_reduce_lr]
    
    history = ts_model.fit(train_x, train_y, epochs=conf.n_train_epochs, validation_data=(val_x, val_y), batch_size=conf.train_batchSize, verbose=2, callbacks=call_backs)
    # now saving the history data to plot some stats in future
    np.save(os.path.join(curr_log_dir, "train_history.npy", history.history))

def evaluate():
    with open(os.path.join(conf.model_dir, "scalers.pkl"), 'rb') as fp:
        scalers = pickle.load(fp)
    model_path = os.path.join(conf.model_dir, "tsmodel.h5")
    ts_model = tf.keras.models.load_model(model_path)
    test_df  = load_csv(conf.test_csv)
    scaled_test_df, _ = scale_data(test_df, scalers)
    test_x, test_y = split_series(scaled_test_df, conf.n_past, conf.n_ahead)
    pred_y = ts_model.predict(test_x)
    # unscaled_test_y = inverse_scaled_data(test_y, scalers)
    unscaled_pred_y = inverse_scaled_data(pred_y, scalers)
    fc = test_df.tail(len(test_y)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = unscaled_pred_y[:,1,6] # we just need the power demand column
    # Ploting the forecasts
    plt.figure(figsize=(12, 8))
    for dtype in ['TOTALDEMAND', 'forecast']:
        plt.plot(
            'DATETIME',
            dtype,
            data=fc,
            label=dtype,
            alpha=0.8
        )
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('plot1.png')
    # TODO MSE error evaluation

def predict_n_ahead(model, PROIR_X, n_ahead: int):
    """
    Current model is trained to predict pre-determined n_ahead step, but if we need to predict some other n_ahead numbers we can use this function
    #TODO we use only the single prediction from the model here, we could use the first n_ahead trained or use that to get average etc.
    THIS FUNCTION IS NOT COMPLETE YET. 
    """    
    # Making the prediction list 
    yhat = []
    for _ in range(n_ahead):
        # Making the prediction
        fc = model.predict(PROIR_X)[:,1,:] # just get the single out for now.
        yhat.append(fc)

        # Creating a new input matrix for forecasting
        X = np.append(X, fc)

        # Ommiting the first variable
        X = np.delete(X, 0)

        # Reshaping for the next iteration
        X = np.reshape(X, (1, len(X), 1))
    
    return yhat

if __name__ == "__main__":
    # train()
    predict()
