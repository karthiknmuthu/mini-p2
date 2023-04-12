import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from datetime import datetime

st.title('Stock Price Prediction')
start = '2022-01-01'
endd = '2023-03-03'
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
#df = data.DataReader (user_input, 'yahoo', start, end)
yf.pdr_override()
y_symbols = user_input
startdate = datetime(2015,1,1)
enddate = datetime(2023,3,3)
data = pdr.get_data_yahoo(y_symbols, startdate, enddate)
data.head()

st.subheader('Data from 2015 - 2023')
st.write(data.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Close)
st.pyplot(fig)

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



#Trained model
#model = tf.keras.saving.load_model('keras_model.h5')
#model = keras.models.load_model('keras_model.h5')
#model = load_model('C:\Apps\StockPricePrediction\keras_model.h5', compile=True)
model = load_model("C:\\Users\\ASUS\\AppData\\Local\\Programs\\Python\\Python311\\miniproject\\keras_models.h5",compile=False)
model.compile(optimizer='adam',loss='mean_squared_error')

#Testing
past_100_days=data_training.tail(100)
final_df=pd.concat([past_100_days,data_testing],ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

#making prediction
y_predicted=model.predict(x_test)
scaler = scaler.scale_
scale_factor=1/0.00682769
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Graph
st.subheader('Prediction vs Real Stock Prices')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
