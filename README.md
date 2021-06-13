# Create and activating virtual environment
```
virtualenv -p python3.7 deep_ts
source deep_ts/bin/activate
```
# clone the repo
```
git clone https://github.com/sabeshuom/deep_ts
```
# Install required python packages
```
pip install -r requirements.txt
```
# Training 
Set the model config and paths in conf.py and run train.py
```
python ts_main.py --train
```
# Testing

python ts_main.py --test

# Directory Structure
./data - contains data for Train.csv, Val.csv, Test.csv 
./models - all the models .h5
./logs - all the tensorboard event data and history

