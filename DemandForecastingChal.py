import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

######################################################################################################
# Functions
######################################################################################################
def create_dataset(X, y, time_steps=1,pred_steps=0):
    Xs, ys = [], []
    for i in range(len(X) - time_steps-pred_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps+pred_steps])
    return np.array(Xs), np.array(ys)

def create_sub_dataset(X, y, time_steps=1):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

def custom_smape(x, x_): # From the Public Kernel https://www.kaggle.com/rezas26/simple-keras-starter
    return keras.backend.mean(2*keras.backend.abs(x-x_)/(keras.backend.abs(x)+keras.backend.abs(x_)))

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

######################################################################################################
# Constants
######################################################################################################
DATA_FOLDER_PATH = "/home/rana/Software&Data/Data/Upal/demand_forecas/"
TIME_STEPS = 90
PRED_STEP = 90

######################################################################################################
# Date read & Analysis
######################################################################################################

# DF_Feature = pd.read_csv(DATA_FOLDER_PATH+'features.csv',parse_dates=['Date'])
# DF_Stores = pd.read_csv(DATA_FOLDER_PATH+'stores.csv')
DF_train = pd.read_csv(DATA_FOLDER_PATH+'train.csv',parse_dates=['date'])
DF_test = pd.read_csv(DATA_FOLDER_PATH+'test.csv',parse_dates=['date'])
# DF_train= DF_train.sort_values('date')
# DF_test= DF_test.sort_values('date')

print(len(DF_train))
DF_train['day'] = DF_train['date'].dt.day
DF_train['month'] = DF_train['date'].dt.month
DF_train['year'] = DF_train['date'].dt.year

# we are taking training data for according to the prediction and timestep to predict actual test set
DF_sub_test = pd.DataFrame(columns=DF_train.columns)
count = 0
ind = 0
for index, row in DF_test.iterrows():
    # find test set start date and corresponding training date for predicting submission test set
    sub_test_start_date = row['date']  - pd.to_timedelta(TIME_STEPS+PRED_STEP, unit='d')

    DF_sub_test_start = DF_train[(DF_train['date']==sub_test_start_date) & (DF_train['store']==row['store']) & (DF_train['item']==row['item'])]
    ind = DF_sub_test_start.index.values[0]
    # print(DF_sub_test_start)
    DF_sub_test = DF_sub_test.append(DF_sub_test_start)
    # count = count+1
    # if count >= 100:
    #     break

# print(ind)
# print(len(DF_train.iloc[ind:(ind+TIME_STEPS+PRED_STEP),]))

# Append time_steps + pred_step data for last sequence
DF_sub_test= DF_sub_test.append(DF_train.iloc[ind:(ind+TIME_STEPS+PRED_STEP-1),])

# take all rows from that index to the end of training set
DF_sub_test.reset_index()
print(len(DF_sub_test))

######################################################################################################
# Preprocessing
######################################################################################################
# train test splitting
train_size = int(len(DF_train) * 0.9)
train, test = DF_train.iloc[0:train_size], DF_train.iloc[train_size:len(DF_train)]
# print('test: ', test)
features = ['store','item','day','month','year']
features_sales = ['store','item','day','month','year','sales']

# feature scaling
feature_scaler = MinMaxScaler()
sale_scaler = MinMaxScaler()

train.loc[:,'sales'] = sale_scaler.fit_transform(train[['sales']])
test.loc[:,'sales'] = sale_scaler.fit_transform(test[['sales']])
DF_sub_test.loc[:,'sales'] = sale_scaler.fit_transform(DF_sub_test[['sales']])

train[features] = feature_scaler.fit_transform(train[features])
test[features] = feature_scaler.fit_transform(test[features])
DF_sub_test[features] = feature_scaler.fit_transform(DF_sub_test[features])


# print('train: ',train)
# print('test: ',test)
# print('DF_sub_test: ',DF_sub_test)

# lets test with a sequence size of 4(which means we'll use 4 weeks of data to predict the 5th week
X_train,y_train = create_dataset(train[features_sales],train[['sales']],TIME_STEPS,PRED_STEP)
X_test, y_test = create_dataset(test[features_sales], test[['sales']], TIME_STEPS,PRED_STEP)
X_sub_test = create_sub_dataset(DF_sub_test[features_sales], DF_sub_test[['sales']], TIME_STEPS)
print('train shape: ',X_train.shape, y_train.shape)
print('test shape: ',X_test.shape, y_test.shape)
print('sub_test shape: ',X_sub_test.shape)




######################################################################################################
# Model building
######################################################################################################

model = keras.Sequential()
model.add(keras.layers.LSTM(100,input_shape=(TIME_STEPS,X_train.shape[2])))
model.add(keras.layers.Dropout(rate=0.1))
model.add(keras.layers.Dense(units=1))
model.compile(loss=custom_smape, optimizer='adam')

history = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=5, batch_size=365,shuffle = False,callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# predictions
test_pred = model.predict(X_test)
sub_test_pred = model.predict(X_sub_test)

# inverse transform
test_pred = sale_scaler.inverse_transform(test_pred)
test_y = sale_scaler.inverse_transform(y_test)
sub_test_pred = sale_scaler.inverse_transform(sub_test_pred)

print('test_y:',test_y)
print('test_pred:',test_pred)
print('sub_test_pred:',sub_test_pred)

# calculate SMAPE
SMAPE = smape(test_pred,test_y);
print('validation SMAPE:',SMAPE)

# write to submission file
sub_df = pd.DataFrame(np.array(sub_test_pred).flatten(),columns=['sales'])
sub_df.to_csv(DATA_FOLDER_PATH+'sample_submission.csv',header=False)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot prediction
# plt.plot(np.arange(0, len(y_train)), train_y.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), test_y.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), test_pred.flatten(), 'r', label="prediction")
plt.ylabel('Sales')
plt.xlabel('Time Step')
plt.legend()
plt.show()
