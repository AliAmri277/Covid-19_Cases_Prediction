#%%
# Import 
from tensorflow import keras
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import SimpleRNN, Dense, LSTM, Dropout
from keras import Sequential, Input
from keras.callbacks import TensorBoard, EarlyStopping
import datetime
from keras.utils import plot_model

#%%
# 1. Data Loading

# 1.1. Load Data for Train File
TRAIN_CSV_PATH= os.path.join(os.getcwd(), 'dataset', "cases_malaysia_train.csv")
df_train=pd.read_csv(TRAIN_CSV_PATH)

# 1.2. Load Data for Test File
TEST_CSV_PATH=os.path.join(os.getcwd(), 'dataset', "cases_malaysia_test.csv")
df_test=pd.read_csv(TEST_CSV_PATH)

#%%
# 2. Data Inspection
df_train.info()

# Check for missing values 
print(df_train['cases_new'].isna().sum())

#%%
df_test.info()

# Check for missing values 
print(df_test['cases_new'].isna().sum())

#%%
# 3. Data Cleaning Train Data
# 3.1 Cleaning Train Data

# Change character values to NaN
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'],errors='coerce')

# Fill NaN with number using interpolate 
df_train['cases_new']=df_train['cases_new'].interpolate(method='polynomial',order=2)

df_train.info()

# Plot graph to see missing values
plt.figure()
plt.plot(df_train['cases_new'].values)
plt.show()

#%%
# 3.2 Cleaning Test Data

# Check for missing values 
print(df_test['cases_new'].isna().sum())

# Fill NaN with number using interpolate 
df_test['cases_new']=df_test['cases_new'].interpolate(method='polynomial',order=2)

df_test.info()

# Plot graph to see missing values
plt.figure()
plt.plot(df_test['cases_new'].values)
plt.show()

#%%
# 4. Features Selection

#%%
# 5. Data preprocessing

# Reshape cases into 2D
data=df_train['cases_new'].values
data=data.reshape(-1,1)

# Normalize feature
mm=MinMaxScaler()
mm.fit(data)
data=mm.transform(data)

# Creating 30 days window size
win_size=30
X_train=[]
y_train=[]

# Loop 30 days window and append the list
for i in range(win_size, len(data)): 
    X_train.append(data[i-win_size:i]) 
    y_train.append(data[i]) 

#to convert into numpy array
X_train=np.array(X_train)
y_train=np.array(y_train)

#train test split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,random_state=123)

#%%
#6. Model Development
model = Sequential()
model.add(LSTM(64, input_shape=X_train.shape[1:], return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
plot_model(model,show_shapes=True)

model.summary()

#Compile model
model.compile(optimizer='adam',loss='mse',metrics=['mse', 'mape'])

#callbacks
#early stopping and tensorboard
LOGS_PATH=os.path.join(
    os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%M%d-%H%M%S'))

tensorbaord_callback=TensorBoard(log_dir=LOGS_PATH)

hist=model.fit(X_train, y_train, epochs=20, callbacks=[tensorbaord_callback], validation_data=(X_test,y_test))

# %%
#7. Model Evaluation

# Concatinating train data with test data 
concat= pd.concat((df_train['cases_new'], df_test['cases_new']),axis=0)

# Slice concat
concat= concat[len(concat)-win_size-len(df_test):]

# Normalize the concat
concat=mm.transform(concat[::,None])

# Create empty list
X_testtest=[]
y_testtest=[]

# loop window and append list
for i in range(win_size,len(concat)):
    X_testtest.append(concat[i-win_size:i])
    y_testtest.append(concat[i])

# convert to array
X_testtest=np.array(X_testtest)
y_testtest=np.array(y_testtest)

# %%
#to predict the new cases based on the testing dataset
cases_new=model.predict(X_testtest)

#%%
#To visualize the predicted new cases vs actual new cases
cases_new = mm.inverse_transform(cases_new)
y_testtest = mm.inverse_transform(y_testtest)

plt.figure()
plt.plot(cases_new,color='red')
plt.plot(y_testtest,color='blue')
plt.legend(['Predicted Cases','Actual Cases'])
plt.xlabel('time')
plt.ylabel('Covid-19 Cases')
plt.show()

#%%
# Metrics to evaluate the performance
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
print('MAPE: ', mean_absolute_percentage_error(y_testtest,cases_new))

# %% 
# Model saving
# Save scaler
import pickle
with open('mm.pkl', 'wb') as f:
    pickle.dump(mm, f)

# Save entire model
model.save('model.h5')
