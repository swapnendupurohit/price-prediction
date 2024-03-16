import streamlit as st 
from datetime import date
import pandas as pd 
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
from keras.models import load_model

start='2000-01-01'
end=date.today().strftime("%Y-%m-%d")

st.title('Stock Trend Prediction')


# Create an empty placeholder
result_placeholder = st.empty()
n_years=st.slider('Years of Prediction: ',0,4)
period = n_years*365

# Function 1: Input Function
def input_function():
    st.header("Find your stock")
    # Add input fields or widgets for user input
    ticker_symbol = st.text_input("Enter the stock ticker symbol (e.g., AAPL):")
    if st.button("Next"):
        return ticker_symbol  # Return ticker symbol to pass to next function

#@st.cache_resource
def load_data(ticker_symbol):
    # Check if input is provided
    if ticker_symbol:
        try:
            # Download stock data from Yahoo Finance
            data_load_state=st.text('Loading data...')
            data = yf.download(ticker_symbol.upper(), start, end)
            data_load_state.text('Loading data.....done!')
        except Exception:
            result_placeholder.error(f"Error downloading stock data. Please try it again.")
    else:
        # Display initial message when no input is provided
        result_placeholder.write("Enter a ticker symbol to see stock data.")
    data.reset_index(inplace=True)
    return data

# Function to preprocess stock data


def preprocess_data(data):
    # Reset index
    data.reset_index(inplace=True)
    # Drop NaN values
    data.dropna(inplace=True)
    return data

# Function 3: Display Data Function
def display_data_function(data):
    # Display the loaded stock data (if available)
    if data is not None:
        #Describing the Data
        st.subheader('Recent Stock Prices')
        st.write(data.head())  # Display the first few rows of the data
    else:
        st.warning("No stock data available.")


#Ploting Raw Data
def plot_raw_data(data):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='Stock Close'))
    fig.update_layout(title='Time Series Data',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


#Forecasting
def create_model(data):
    df_train=data[['Date','Close']]
    df_train=df_train.rename(columns={'Date':'ds','Close':'y'})
    df_train['ds']=pd.to_datetime(df_train['ds'])

    model=Prophet()
    model.fit(df_train)
    return model

def Forecast(model):
   #Forecast Data
   st.subheader('Forecast Data (Using PROPHET)')
   future=model.make_future_dataframe(periods=period)
   forecast=model.predict(future)
   forecast.set_index('ds',inplace=True)
   st.write(forecast.head())
   st.subheader('Enter the date of prediction')
   input_date=st.text_input('please enter in yyyy-mm-dd format','2024-02-25')
   future_date=pd.DataFrame({'ds':[input_date]})
   prediction=model.predict(future_date)
   predicted_price=prediction.loc[0,'yhat']
   st.write('Predicted price: ',predicted_price)
   return forecast
   
#Plot Forecast data
def plot_forecast_data(data,forecast,model):
  forecast.reset_index(inplace=True)
  st.write('Forecast data (Using PROPHET)')
  fig1=go.Figure()
  fig1.add_trace(go.Scatter(x=forecast['ds'],y=forecast['yhat'],name='Predicted'))
  fig1.add_trace(go.Scatter(x=forecast['ds'],y=data['Close'],name='Actual'))
  fig1.update_layout(xaxis_rangeslider_visible=True)
  st.plotly_chart(fig1)
  st.write('Forecast Components')
  fig2=model.plot_components(forecast)
  st.write(fig2)

# Function to calculate moving averages and plot


#Visualisation
def visualisation(data):
   st.subheader('Closing Price vs Time Graph (Using LSTM)')
   fig=plt.figure(figsize=(12,6))
   plt.plot(data.Close)
   plt.legend()
   st.pyplot(fig)
   
   st.subheader('Closing Price vs Time Graph with 100MA (Using LSTM)')
   ma_100_days = data.Close.rolling(100).mean()
   fig=plt.figure(figsize=(12,6))
   plt.plot(data.Close,'b',label='Close Price')
   plt.plot(ma_100_days, 'r', label='MA 100 days')
   plt.legend()
   st.pyplot(fig)

   st.subheader('Closing Price vs Time Graph with 100MA and 200MA (Using LSTM)')
   ma_200_days = data.Close.rolling(200).mean()
   fig=plt.figure(figsize=(12,6))
   plt.plot(ma_100_days, 'r', label='MA 100 days')
   plt.plot(ma_200_days, 'b', label='MA 200 days')
   plt.plot(data.Close, 'g', label='Close Price')
   plt.legend()
   st.pyplot(fig)


#Splitting Data into Training and Testing

def train_test_data(data):
   data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
   data_test = pd.DataFrame(data.Close[int(len(data)*0.80): int(len(data))])

   scaler = MinMaxScaler(feature_range=(0,1))
   data_train_array = scaler.fit_transform(data_train)

   #Splitting Data into x_train and Y
   x_train = []
   y_train = []

   for i in range(100, data_train_array.shape[0]):
       x_train.append(data_train_array[i-100:i])
       y_train.append(data_train_array[i,0])
   x_train,y_train=np.array(x_train),np.array(y_train)

   model=load_model('model.keras')
   #Testing Data
   past_days = data_train.tail(100)
   data_test = pd.concat([past_days, data_test], ignore_index=True)
   data_test_scale  =  scaler.fit_transform(data_test)

   x_test = []
   y_test = []

   for i in range(100, data_test_scale.shape[0]):
       x_test.append(data_test_scale[i-100:i])
       y_test.append(data_test_scale[i,0])
   x_test, y_test = np.array(x_test), np.array(y_test)

   #Predict Data
   y_predict = model.predict(x_test)

   scaler=scaler.scale_
   scale_factor=1/scaler[0]
   y_predict=y_predict*scale_factor
   y_test=y_test*scale_factor
   return y_test,y_predict

#Final Graph
def final_graph(y_test,y_predict):
   st.subheader('Predictions vs Originals (Using LSTM)')
   fig2=plt.figure(figsize=(12,6))
   plt.plot(y_test,'b',label='Actual')
   plt.plot(y_predict,'r',label='Prediction')
   #plt.plot(y_predict_1,'g',label='Prediction( PROPHET )')
   plt.xlabel('Time')
   plt.ylabel('Price')
   plt.legend()
   st.pyplot(fig2)


# Main Streamlit App
def main():
    st.title("Loading Data from Yahoo Finance")
    # Execute functions sequentially
    ticker_symbol = input_function()
    if ticker_symbol is not None:
        stock_data = load_data(ticker_symbol)
        display_data_function(stock_data)
        plot_raw_data(stock_data)
        model_1=create_model(stock_data)
        forecast=Forecast(model_1)
        y_predict_1=forecast['yhat']
        plot_forecast_data(stock_data,forecast,model_1)
        model_2=load_model('model.keras')
        visualisation(stock_data)
        y_test,y_predict=train_test_data(stock_data)
        final_graph(y_test,y_predict)
    
    
if "_main_":
    main()