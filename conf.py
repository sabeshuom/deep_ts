train_csv = "data/Train.csv"
val_csv = "data/Val.csv"
test_csv = "data/Test.csv"
n_past = 20 # number of past data to use 
n_ahead = 1 # number of ahead data to predict
n_features = 7 # number of features in the dataset
n_lstmLayers = 100 # number of layers in the lstm
n_train_epochs = 200
train_batchSize = 64
model_dir = "models"
log_dir = "logs"
