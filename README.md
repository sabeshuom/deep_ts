# Create and activating virtual environment
```
virtualenv -p python3.7 deep_ts
source deep_ts/bin/activate
```
# install required python packages
```
pip install -r requirements.txt
```
# training 
set the model config and paths in conf.py and run train.py
```
python train.py
```
# directories
./data - contains data for Train.csv, Val.csv, Test.csv 
./models - all the models .h5
./logs - all the tensorboard event data and history

