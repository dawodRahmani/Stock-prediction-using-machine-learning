import yfinance as yf
import pandas as pd
import plotly.express as px
import streamlit as st
import pandas_ta as ta
import plotly.graph_objects as go

layout="wide"
st.set_page_config(page_title="MSFT PP", page_icon=":bar_chart:",layout="wide"
)

def retrieve_data(interval,period):
    ticker = 'MSFT'
    data = yf.download(ticker, interval=interval, period=period)
    return data

def process_data(data):
    df = pd.DataFrame(data)
    
    df['ATR'] = df.ta.atr(length=20)
    df['RSI'] = df.ta.rsi()
    df['Average'] = df.ta.midprice(length=1)
    df['MA40'] = df.ta.sma(length=40)
    df['MA80'] = df.ta.sma(length=80)
    df['MA160'] = df.ta.sma(length=160)
    return df

# Define the function to be called when the button is clicked
def do_some_work():
    st.write("Button clicked! Doing some work...")

# svm
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
import joblib

def predict_function(mod):
  # Assuming you have already retrieved the data and calculated the technical indicators
  data = retrieve_data("1d", "2y")
  df = process_data(data)

  # Extract the last row
  last_row = df.tail(5)

  # Load the pre-trained model
  loaded_model = joblib.load('../Model/' + mod)

  # Select features for prediction
  features = ['ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160']

  # Scale the features from the new data
  scaler = StandardScaler()
  last_row_scaled = scaler.fit_transform(last_row[features])

  # Make prediction
  prediction = loaded_model.predict(last_row_scaled)

  return prediction



# sidebar
st.sidebar.header("Please Filter Here:")
interval = st.sidebar.selectbox("Select Interval", ["1d", "5m", "15m", "30m", "1h", "1d"])
period = st.sidebar.selectbox("Select Period", ["2y", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])

# Create a button
if st.sidebar.button("Click me"):
    # Call the function when the button is clicked
    pred = predict_function('XgboostModel.sav')
    st.sidebar.write("Xgboost: " + str(pred))
    
    pred1 = predict_function('KnnModel.sav')
    st.sidebar.write("KNN: " + str(pred1))

    pred2 = predict_function('svm.sav')
    st.sidebar.write("SVM: " + str(pred2))

    dataa = retrieve_data("1d", "2y")
    dfa = process_data(dataa)

  # Extract the last row
    last_rowa = dfa.tail(1)
    st.sidebar.write(last_rowa)

if st.sidebar.button("last day price"):


  data12 = retrieve_data("1d", "2y")
  df12 = process_data(data12)

  # Extract the last row
  last_row12 = df12.tail(5)
  st.sidebar.write(last_row12)


    # data1 = retrieve_data("1d", "400d")
    # df = process_data(data1)
    # df1 = pd.DataFrame(data1)
    # last_row = df1.tail(1)
    # st.sidebar.write(data2)

# main page
st.title(":bar_chart: MSFT HOURLY CHARTS")
st.markdown("##")

data = retrieve_data(interval, period)
df = process_data(data)

st.write(df)

# Plot Candlestick Chart
fig = go.Figure(data=[go.Candlestick(x=df.index,
                       open=df['Open'], high=df['High'],
                       low=df['Low'], close=df['Close'])])

st.plotly_chart(fig)
