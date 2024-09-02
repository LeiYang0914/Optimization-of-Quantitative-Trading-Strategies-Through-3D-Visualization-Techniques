from datetime import datetime
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import plotly.graph_objects as go
import streamlit as st

from bayes_opt import BayesianOptimization
from skopt import gp_minimize
from skopt.space import Integer,Real
from skopt.utils import use_named_args
from plotly.subplots import make_subplots
from sklearn.model_selection import ParameterGrid
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Custom CSS for enhancing the UI
st.markdown("""
    <style>
        /* Background color */
        body {
            background-color: #f0f2f6;
        }

        /* Increase font sizes */
        .big-font {
            font-size:35px !important;
            color: #FF6347;
            font-weight: bold;
        }
        .header-font {
            font-size:32px !important;
            font-weight: bold;
            color: #FF4500;
        }
        .subheader-font {
            font-size:26px !important;
            font-weight: bold;
            color: #FF6347;
        }
        .metric-value {
            font-size:24px !important;
            color: #333;
        }
        
        /* Custom buttons and sidebar styles */
        .stButton>button {
            background-color: #FF4500;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            height: 50px;
            width: 100%;
        }
        .stSidebar > div {
            background-color: #fff;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stSidebar h1 {
            font-size: 24px;
            color: #FF4500;
            font-weight: bold;
        }
        
        /* Expander header */
        .streamlit-expanderHeader {
            font-size: 20px !important;
            font-weight: bold;
            background-color: #FF6347;
            color: white !important;
            border-radius: 5px;
            padding: 10px;
        }

        /* Center content */
        .center-content {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        /* Add some margin to the content */
        .stApp {
            padding: 30px;
        }

        /* Table styling */
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 18px;
            text-align: left;
        }
        .metrics-table th, .metrics-table td {
            padding: 12px 15px;
        }
        .metrics-table th {
            background-color: #FF6347;
            color: white;
            font-weight: bold;
        }
        .metrics-table tr {
            border-bottom: 1px solid #dddddd;
        }
        .metrics-table tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .metrics-table tr:last-of-type {
            border-bottom: 2px solid #FF6347;
        }
    </style>
    """, unsafe_allow_html=True)

# Define the strategies
def macd_strategy(data, fast_length=12, slow_length=26):
    # Define a fixed macd_length
    macd_length = 5

    # Calculate the MACD and its signal line
    data['MACD'] = ta.trend.ema_indicator(data['Close'], window=fast_length) - ta.trend.ema_indicator(data['Close'], window=slow_length)
    data['Signal'] = ta.trend.ema_indicator(data['MACD'], window=macd_length)
    data['Delta'] = data['MACD'] - data['Signal']

    # Generate signals
    data['Signal_Long'] = np.where((data['Delta'].shift(1) < 0) & (data['Delta'] > 0), 1, 0)
    data['Signal_Short'] = np.where((data['Delta'].shift(1) > 0) & (data['Delta'] < 0), -1, 0)

    data['Position (Long Short)'] = data['Signal_Long'] + data['Signal_Short']

    # Define trading fees
    taker_fee = 0.00055  # 0.055%

    # Calculate PnL with fees, Cumulative PnL, and Drawdown
    data['PnL (Long Short)'] = data['Position (Long Short)'].shift(1) * (data['Close'] / data['Close'].shift(1) - 1)
    
    # Determine if the trade was a maker or taker (for simplicity, we'll assume all trades are taker trades)
    data['PnL (Long Short) with Fees'] = data['PnL (Long Short)'] - (data['Position (Long Short)'].shift(1) * taker_fee)
    
    data['Cumulative PnL (Long Short)'] = data['PnL (Long Short) with Fees'].cumsum()
    data['Cumulative Max'] = data['Cumulative PnL (Long Short)'].cummax()
    data['Drawdown (Long Short)'] = data['Cumulative Max'] - data['Cumulative PnL (Long Short)']

    # Calculate Number of Trades
    data['Number of Trades'] = ((data['Position (Long Short)'] != 0) & (data['Position (Long Short)'] != data['Position (Long Short)'].shift(1))).cumsum()

    return data

def moving_avg_cross_strategy(data, fast_length=9, slow_length=18):
    """
    Apply the Moving Average Cross strategy on a given DataFrame.
    
    Parameters:
    - data (pd.DataFrame): DataFrame with a 'Close' price column.
    - fast_length (int): Fast moving average window length.
    - slow_length (int): Slow moving average window length.
    
    Returns:
    - data (pd.DataFrame): DataFrame with MA strategy columns and cumulative returns.
    """
    # Calculate fast and slow moving averages
    data['fast_ma'] = ta.trend.SMAIndicator(close=data['Close'], window=fast_length).sma_indicator()
    data['slow_ma'] = ta.trend.SMAIndicator(close=data['Close'], window=slow_length).sma_indicator()
    
    # Generate trading signals
    data['long_signal'] = (data['fast_ma'] > data['slow_ma']) & (data['fast_ma'].shift(1) <= data['slow_ma'].shift(1))
    data['short_signal'] = (data['fast_ma'] < data['slow_ma']) & (data['fast_ma'].shift(1) >= data['slow_ma'].shift(1))
    
    # Initialize positions
    data['Position (Long Short)'] = 0
    data.loc[data['long_signal'], 'Position (Long Short)'] = 1
    data.loc[data['short_signal'], 'Position (Long Short)'] = -1
    
    # Forward fill positions
    data['Position (Long Short)'] = data['Position (Long Short)'].replace(0, pd.NA).ffill().fillna(0)
    
    # Define trading fees
    taker_fee = 0.00055  # 0.055%

    # Calculate strategy returns with fees
    data['PnL (Long Short)'] = data['Position (Long Short)'].shift(1) * data['Close'].pct_change()
    data['PnL (Long Short) with Fees'] = data['PnL (Long Short)'] - (abs(data['Position (Long Short)'].shift(1) - data['Position (Long Short)']) * taker_fee)

    # Calculate cumulative returns
    data['Cumulative PnL (Long Short)'] = data['PnL (Long Short) with Fees'].cumsum()
    
    # Calculate drawdown
    data['Cumulative Max'] = data['Cumulative PnL (Long Short)'].cummax()
    data['Drawdown (Long Short)'] = data['Cumulative Max'] - data['Cumulative PnL (Long Short)']
    
    # Calculate number of trades
    data['Trade Entry'] = (data['Position (Long Short)'] != 0) & (data['Position (Long Short)'] != data['Position (Long Short)'].shift(1))
    data['Number of Trades'] = data['Trade Entry'].cumsum()
    
    return data

def adx_di_crossover_strategy(data, adx_length=14, adx_threshold=25):
    """
    Apply ADX and DI Crossover Strategy on a given DataFrame.
    
    Parameters:
    - data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' price columns.
    - adx_length (int): Length for ADX calculation.
    - adx_threshold (int): Threshold for ADX to consider a strong trend.
    
    Returns:
    - data (pd.DataFrame): DataFrame with strategy results.
    """
    # Calculate ADX, +DI, -DI
    adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=adx_length)
    data['adx'] = adx.adx()
    data['+DI'] = adx.adx_pos()
    data['-DI'] = adx.adx_neg()
    
    # Generate signals
    data['long_signal'] = (data['adx'] > adx_threshold) & (data['+DI'] > data['-DI']) & (data['+DI'].shift(1) <= data['-DI'].shift(1))
    data['short_signal'] = (data['adx'] > adx_threshold) & (data['-DI'] > data['+DI']) & (data['-DI'].shift(1) <= data['+DI'].shift(1))
    
    # Initialize positions
    data['Position (Long Short)'] = 0
    data.loc[data['long_signal'], 'Position (Long Short)'] = 1
    data.loc[data['short_signal'], 'Position (Long Short)'] = -1
    data['Position (Long Short)'] = data['Position (Long Short)'].replace(0, pd.NA).ffill().fillna(0)
    
    # Define trading fees
    taker_fee = 0.00055  # 0.055%

    # Calculate strategy returns with fees
    data['PnL (Long Short)'] = data['Position (Long Short)'].shift(1) * data['Close'].pct_change()
    data['PnL (Long Short) with Fees'] = data['PnL (Long Short)'] - (abs(data['Position (Long Short)'].shift(1) - data['Position (Long Short)']) * taker_fee)

    # Calculate cumulative returns
    data['Cumulative PnL (Long Short)'] = data['PnL (Long Short) with Fees'].cumsum()
    
    # Calculate drawdown
    data['Cumulative Max'] = data['Cumulative PnL (Long Short)'].cummax()
    data['Drawdown (Long Short)'] = data['Cumulative Max'] - data['Cumulative PnL (Long Short)']
    
    # Calculate number of trades
    data['Trade Entry'] = (data['Position (Long Short)'] != 0) & (data['Position (Long Short)'] != data['Position (Long Short)'].shift(1))
    data['Number of Trades'] = data['Trade Entry'].cumsum()
    
    return data

def mean_reversion_strategy(data, move_limit=10, ma_length=50):
    """
    Apply the Mean Reversion Strategy on a given DataFrame.
    
    Parameters:
    - data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', and 'Close' columns.
    - move_limit (int): Percentage limit to identify a big move.
    - ma_length (int): Moving average period length.
    
    Returns:
    - data (pd.DataFrame): DataFrame with strategy columns and cumulative returns.
    """
    # Moving Average
    data['ma'] = ta.trend.SMAIndicator(data['Close'], window=ma_length).sma_indicator()
    
    # Identify Down and Up Bars
    data['down_bar'] = data['Open'] > data['Close']
    data['up_bar'] = ~data['down_bar']
    
    # Identify 2 consecutive down/up bars
    data['is_two_down'] = data['down_bar'] & data['down_bar'].shift(1)
    data['is_two_up'] = data['up_bar'] & data['up_bar'].shift(1)
    
    # Identify Big Moves
    data['big_move_down'] = ((data['Open'] - data['Close']) / (0.001 + data['High'] - data['Low'])) > move_limit / 100.0
    data['big_move_up'] = ((data['Close'] - data['Open']) / (0.001 + data['High'] - data['Low'])) > move_limit / 100.0
    
    # Long Entry and Exit Signals
    data['is_long_buy'] = data['is_two_down'] & data['big_move_down']
    data['is_long_exit'] = data['Close'] > data['High'].shift(1)
    
    # Short Entry and Exit Signals
    data['is_short_buy'] = data['is_two_up'] & data['big_move_up'] & (data['Close'] < data['ma'])
    data['is_short_exit'] = data['Close'] < data['Low'].shift(1)
    
    # Initialize Positions
    data['Position (Long Short)'] = 0
    data.loc[data['is_long_buy'], 'Position (Long Short)'] = 1
    data.loc[data['is_long_exit'], 'Position (Long Short)'] = 0
    data.loc[data['is_short_buy'], 'Position (Long Short)'] = -1
    data.loc[data['is_short_exit'], 'Position (Long Short)'] = 0
    data['Position (Long Short)'] = data['Position (Long Short)'].replace(0, pd.NA).ffill().fillna(0).infer_objects(copy=False)
    
    # Define trading fees
    taker_fee = 0.00055  # 0.055%

    # Calculate PnL with fees
    data['PnL (Long Short)'] = data['Position (Long Short)'].shift(1) * (data['Close'] / data['Close'].shift(1) - 1)
    data['PnL (Long Short) with Fees'] = data['PnL (Long Short)'] - (abs(data['Position (Long Short)'].shift(1) - data['Position (Long Short)']) * taker_fee)

    # Calculate Cumulative PnL
    data['Cumulative PnL (Long Short)'] = data['PnL (Long Short) with Fees'].cumsum()
    
    # Calculate Cumulative Max and Drawdown
    data['Cumulative Max'] = data['Cumulative PnL (Long Short)'].cummax()
    data['Drawdown (Long Short)'] = data['Cumulative Max'] - data['Cumulative PnL (Long Short)']

    # Calculate Number of Trades
    data['Number of Trades'] = ((data['Position (Long Short)'] != 0) & (data['Position (Long Short)'] != data['Position (Long Short)'].shift(1))).cumsum()
    
    return data


def bollinger_rsi_strategy(data, rsi_length=14, bb_length=20):
    """
    Apply the Bollinger Band with RSI Strategy on a given DataFrame.
    
    Parameters:
    - data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', and 'Close' columns.
    - rsi_length (int): Length for RSI calculation.
    - bb_length (int): Length for Bollinger Bands calculation.
    
    Returns:
    - data (pd.DataFrame): DataFrame with strategy results.
    """
    # Define fixed parameters
    bb_stddev = 3.0
    long_tp_pct = 15
    long_sl_pct = 8
    short_tp_pct = 12
    short_sl_pct = 25

    # Calculate RSI
    data['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_length).rsi()
    
    # Calculate Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(close=data['Close'], window=bb_length, window_dev=bb_stddev)
    data['bb_upper'] = bb_indicator.bollinger_hband()
    data['bb_lower'] = bb_indicator.bollinger_lband()
    data['bb_middle'] = bb_indicator.bollinger_mavg()
    
    # Calculate take profit and stop loss levels
    data['long_tp_level'] = data['Close'] * (1 + long_tp_pct / 100)
    data['long_sl_level'] = data['Close'] * (1 - long_sl_pct / 100)
    data['short_tp_level'] = data['Close'] * (1 - short_tp_pct / 100)
    data['short_sl_level'] = data['Close'] * (1 + short_sl_pct / 100)
    
    # Entry and Exit Signals
    data['entry_long'] = (data['rsi'] < 30) & (data['Close'] < data['bb_lower']) 
    data['exit_long'] = (data['rsi'] > 70) | (data['Close'] > data['bb_middle'])  
    
    data['entry_short'] = (data['rsi'] > 70) & (data['Close'] > data['bb_upper'])  
    data['exit_short'] = (data['rsi'] < 30) | (data['Close'] < data['bb_middle'])  
    
    # Initialize Positions and Trades
    data['Position (Long Short)'] = 0
    data['In Trade'] = False
    
    for i in range(1, len(data)):
        if not data['In Trade'].iloc[i-1]:
            if data['entry_long'].iloc[i]:
                data.at[data.index[i], 'Position (Long Short)'] = 1
                data.at[data.index[i], 'In Trade'] = 'Long'
                entry_price = data['Close'].iloc[i]
                tp_level = entry_price * (1 + long_tp_pct / 100)
                sl_level = entry_price * (1 - long_sl_pct / 100)
            elif data['entry_short'].iloc[i]:
                data.at[data.index[i], 'Position (Long Short)'] = -1
                data.at[data.index[i], 'In Trade'] = 'Short'
                entry_price = data['Close'].iloc[i]
                tp_level = entry_price * (1 - short_tp_pct / 100)
                sl_level = entry_price * (1 + short_sl_pct / 100)
        elif data['In Trade'].iloc[i-1] == 'Long':
            if data['Close'].iloc[i] >= tp_level or data['Close'].iloc[i] <= sl_level or data['exit_long'].iloc[i]:
                data.at[data.index[i], 'Position (Long Short)'] = 0
                data.at[data.index[i], 'In Trade'] = False
            else:
                data.at[data.index[i], 'Position (Long Short)'] = 1
        elif data['In Trade'].iloc[i-1] == 'Short':
            if data['Close'].iloc[i] <= tp_level or data['Close'].iloc[i] >= sl_level or data['exit_short'].iloc[i]:
                data.at[data.index[i], 'Position (Long Short)'] = 0
                data.at[data.index[i], 'In Trade'] = False
            else:
                data.at[data.index[i], 'Position (Long Short)'] = -1
    
    data['Position (Long Short)'] = data['Position (Long Short)'].replace(0, pd.NA).ffill().fillna(0)
    
    # Define trading fees
    taker_fee = 0.00055  # 0.055%

    # Calculate strategy returns with fees
    data['PnL (Long Short)'] = data['Position (Long Short)'].shift(1) * (data['Close'] / data['Close'].shift(1) - 1)
    data['PnL (Long Short) with Fees'] = data['PnL (Long Short)'] - (abs(data['Position (Long Short)'].shift(1) - data['Position (Long Short)']) * taker_fee)

    # Calculate cumulative returns
    data['Cumulative PnL (Long Short)'] = data['PnL (Long Short) with Fees'].cumsum()
    
    # Calculate drawdown
    data['Cumulative Max'] = data['Cumulative PnL (Long Short)'].cummax()
    data['Drawdown (Long Short)'] = data['Cumulative Max'] - data['Cumulative PnL (Long Short)']
    
    # Calculate Number of Trades
    data['Trade Entry'] = (data['Position (Long Short)'] != 0) & (data['Position (Long Short)'] != data['Position (Long Short)'].shift(1))
    data['Number of Trades'] = data['Trade Entry'].cumsum()
    
    return data


def z_score_sma_strategy(data, sma_length=50, z_score_threshold=0.75):
    """
    Apply the Z-Score SMA strategy on a given DataFrame.
    
    Parameters:
    - data (pd.DataFrame): DataFrame with a 'Close' price column.
    - sma_length (int): Simple moving average window length.
    - z_score_threshold (float): Z-score threshold for signal generation.
    
    Returns:
    - data (pd.DataFrame): DataFrame with strategy columns and cumulative returns.
    """
    # Calculate the SMA
    data['sma'] = ta.trend.SMAIndicator(close=data['Close'], window=sma_length).sma_indicator()
    
    # Calculate the Z-Score
    data['mean'] = data['Close'].rolling(window=sma_length).mean()
    data['std'] = data['Close'].rolling(window=sma_length).std()
    data['z_score'] = (data['Close'] - data['sma']) / data['std']
    
    # Generate trading signals
    data['long_signal'] = (data['z_score'] > z_score_threshold)
    data['short_signal'] = (data['z_score'] < -z_score_threshold)
    
    # Initialize positions
    data['Position (Long Short)'] = 0
    data.loc[data['long_signal'], 'Position (Long Short)'] = 1
    data.loc[data['short_signal'], 'Position (Long Short)'] = -1
    
    # Forward fill positions
    data['Position (Long Short)'] = data['Position (Long Short)'].replace(0, np.nan).ffill().fillna(0)
    
    # Define trading fees
    taker_fee = 0.00055  # 0.055%

    # Calculate strategy returns with fees
    data['PnL (Long Short)'] = data['Position (Long Short)'].shift(1) * data['Close'].pct_change()
    data['PnL (Long Short) with Fees'] = data['PnL (Long Short)'] - (abs(data['Position (Long Short)'].shift(1) - data['Position (Long Short)']) * taker_fee)

    # Calculate cumulative returns
    data['Cumulative PnL (Long Short)'] = data['PnL (Long Short) with Fees'].cumsum()
    
    # Calculate drawdown
    data['Cumulative Max'] = data['Cumulative PnL (Long Short)'].cummax()
    data['Drawdown (Long Short)'] = data['Cumulative Max'] - data['Cumulative PnL (Long Short)']
    
    # Calculate number of trades
    data['Trade Entry'] = (data['Position (Long Short)'] != 0) & (data['Position (Long Short)'] != data['Position (Long Short)'].shift(1))
    data['Number of Trades'] = data['Trade Entry'].cumsum()
    
    return data

# Define the performance metrics function
def performance_metrics(data, timeframe):
    """
    This function calculates the performance metrics for the given strategy for different timeframes.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'PnL (Long Short)', 'Drawdown (Long Short)', and 'Number of Trades'.
    timeframe (str): The timeframe of the data as a string (e.g., "1H", "1Day", "45Min").

    Returns:
    dict: Dictionary containing Sharpe Ratio, Average Return, Maximum Drawdown, Number of Trades, Calmar Ratio, and Profit/Loss Ratio.
    """
    
    # Look up the number of periods per year based on the timeframe
    periods_per_year = timeframe_mapping.get(timeframe, None)

    if periods_per_year is None:
        raise ValueError("Invalid timeframe. Please choose from the following: " + ", ".join(timeframe_mapping.keys()))

    # Sharpe Ratio calculation
    sharpe_ratio = (data['PnL (Long Short)'].mean() / 
                    data['PnL (Long Short)'].std()) * np.sqrt(periods_per_year)

    # Average Return calculation
    average_return = data['PnL (Long Short)'].mean() * periods_per_year

    # Maximum Drawdown calculation
    maximum_drawdown = data['Drawdown (Long Short)'].max()

    # Calmar Ratio calculation
    calmar_ratio = average_return / abs(maximum_drawdown) if maximum_drawdown != 0 else np.nan

    # Number of Trades
    number_of_trades = data['Number of Trades'].max()

    return {
        'Sharpe Ratio': sharpe_ratio,
        'Average Return': average_return * 100,
        'Maximum Drawdown': maximum_drawdown * 100,
        'Calmar Ratio': calmar_ratio,
        'Number of Trades': number_of_trades 
    }

# Define the function to generate the heatmap
def plot_heatmap(results, param1_name, param2_name, metric_name):
    """
    Plot a heatmap of the optimization results.

    Parameters:
    - results (list of tuples): List of tuples containing parameter combinations and their respective metric.
    - param1_name (str): Name of the first parameter.
    - param2_name (str): Name of the second parameter.
    - metric_name (str): Name of the metric to plot.
    """
    # Extract parameter values and metrics
    param1_vals = [params[param1_name] for params, _ in results]
    param2_vals = [params[param2_name] for params, _ in results]
    metrics = [metric for _, metric in results]

    # Create a DataFrame for the heatmap
    data = pd.DataFrame({
        param1_name: param1_vals,
        param2_name: param2_vals,
        metric_name: metrics
    })

    # Pivot the data for the heatmap
    heatmap_data = data.pivot(index=param1_name, columns=param2_name, values=metric_name)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis')
    plt.title(f'Heatmap of {metric_name}')
    plt.xlabel(param2_name)
    plt.ylabel(param1_name)
    st.pyplot(plt)  # Use st.pyplot to display the heatmap in Streamlit

# Define the search spaces for Bayesian Optimization
strategy_params = {
    'MACD': {
        'strategy': macd_strategy,
        'space': [
            Integer(5, 31, name='fast_length'),
            Integer(20, 51, name='slow_length')
        ]
    },
    'Moving Average Cross': {
        'strategy': moving_avg_cross_strategy,
        'space': [
            Integer(5, 21, name='fast_length'),
            Integer(20, 51, name='slow_length')
        ]
    },
    'ADX DI Crossover': {
        'strategy': adx_di_crossover_strategy,
        'space': [
            Integer(5, 51, name='adx_length'),
            Integer(10, 51, name='adx_threshold')]
    },
    'Mean Reversion': {
        'strategy': mean_reversion_strategy,
        'space': [
            Real(1, 21, name='move_limit'),
            Integer(10, 201, name='ma_length')]
    },
    'Bollinger Band + RSI': {
        'strategy': bollinger_rsi_strategy,
        'space': [
            Integer(5, 101, name='rsi_length'),
            Integer(10, 101, name='bb_length')
         ]
    },
    'Z-Score SMA': {
        'strategy': z_score_sma_strategy,
        'space': [
            Integer(10, 201, name='sma_length'),  
            Real(0.1, 3.1, name='z_score_threshold')  
]
    }
}

# Function to generate parameter grid
def generate_param_grid(space):
    param_grid = {}
    for dim in space:
        if isinstance(dim, Integer):
            param_grid[dim.name] = list(range(dim.bounds[0], dim.bounds[1] + 1, 5))
        elif isinstance(dim, Real):
            param_grid[dim.name] = np.linspace(dim.bounds[0], dim.bounds[1], num=10).tolist()
    return param_grid

# Page Title
st.markdown('<p class="big-font">Quantitative Trading Strategy Optimization</p>', unsafe_allow_html=True)

## 1. Upload a CSV file
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# 2. Choose a timeframe
timeframe_mapping = {
    "1Min": 365 * 24 * 60,
    "5Min": 365 * 24 * 60 / 5,
    "15Min": 365 * 24 * 60 / 15,
    "30Min": 365 * 24 * 60 / 30,
    "45Min": 365 * 24 * 60 / 45,
    "1H": 365 * 24,
    "4H": 365 * 24 / 4,
    "8H": 365 * 24 / 8,
    "12H": 365 * 24 / 12,
    "1Day": 365,
    "1Week": 365 / 7,
    "1Month": 12
}

timeframes = list(timeframe_mapping.keys())
selected_timeframe = st.sidebar.selectbox("Select Timeframe", timeframes)

# 3. Choose the strategy
strategy_choice = st.sidebar.selectbox("Select Strategy", list(strategy_params.keys()))

# 4. Choose optimization method
optimization_method = st.sidebar.radio(
    "Optimization Method",
    ("Bayesian Optimization", "Permutation Testing (Grid Search CV)")
)

# Placeholder for results
best_params = {}
sharpe_ratio = None
metrics = {}

# Perform the selected optimization
if st.sidebar.button("Run Optimization"):
    strategy_info = strategy_params[strategy_choice]
    strategy_function = strategy_info['strategy']
    space = strategy_info['space']

    if optimization_method == "Bayesian Optimization":
        @use_named_args(space)
        def objective(**params):
            result = strategy_function(data.copy(), **params)
            local_metrics = performance_metrics(result, selected_timeframe)
            if np.isnan(local_metrics['Sharpe Ratio']):
                return 0
            return -local_metrics['Sharpe Ratio']

        res = gp_minimize(objective, space, n_calls=50, random_state=0)
        best_params = {dim.name: res.x[i] for i, dim in enumerate(space)}
        best_result = strategy_function(data.copy(), **best_params)
        sharpe_ratio = -res.fun
        metrics = performance_metrics(best_result, selected_timeframe)

        # Extract the evaluated points and corresponding negative Sharpe Ratios
        x_vals = [point[0] for point in res.x_iters]
        y_vals = [point[1] for point in res.x_iters]
        z_vals = [-val for val in res.func_vals]

        # Find the maximum Sharpe Ratio
        max_sharpe_ratio = max(z_vals)
        max_index = z_vals.index(max_sharpe_ratio)
        best_x = x_vals[max_index]
        best_y = y_vals[max_index]

        # Display 3D scatter plot for Bayesian Optimization
        with st.expander("Surface Plot"):
            st.markdown(f'<p class="header-font">3D Visualization of Bayesian Optimization for {strategy_choice} Strategy</p>', unsafe_allow_html=True)
            fig = go.Figure(data=[go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=5,
                    color=z_vals,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Sharpe Ratio')
                ),
                text=[f'{space[0].name}: {x}, {space[1].name}: {y}, Sharpe Ratio: {z:.4f}' for x, y, z in zip(x_vals, y_vals, z_vals)],
                hoverinfo='text'
            )])

            # Highlight the highest Sharpe Ratio point
            fig.add_trace(go.Scatter3d(
                x=[best_x],
                y=[best_y],
                z=[max_sharpe_ratio],
                mode='markers+text',
                marker=dict(size=7, color='red'),
                text=[f'Max Sharpe Ratio: {max_sharpe_ratio:.4f}'],
                textposition='top center'
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title=space[0].name,
                    yaxis_title=space[1].name,
                    zaxis_title='Sharpe Ratio'
                ),
                autosize=True
            )
            st.plotly_chart(fig)

    elif optimization_method == "Permutation Testing (Grid Search CV)":
        param_grid = generate_param_grid(space)

        results = []
        x_vals = []
        y_vals = []
        z_vals = []

        for params in ParameterGrid(param_grid):
            result = strategy_function(data.copy(), **params)
            local_metrics = performance_metrics(result, selected_timeframe)
            if not np.isnan(local_metrics['Sharpe Ratio']):
                results.append((params, local_metrics['Sharpe Ratio']))
                x_vals.append(params[space[0].name])
                y_vals.append(params[space[1].name])
                z_vals.append(local_metrics['Sharpe Ratio'])

        if results:
            best_params, sharpe_ratio = max(results, key=lambda x: x[1])
            best_result = strategy_function(data.copy(), **best_params)
            metrics = performance_metrics(best_result, selected_timeframe)

            # Create grid for surface plot
            x_grid, y_grid = np.meshgrid(np.unique(x_vals), np.unique(y_vals))
            z_grid = np.zeros_like(x_grid, dtype=float)

            # Populate the z values based on the unique x and y values
            for i, x in enumerate(np.unique(x_vals)):
                for j, y in enumerate(np.unique(y_vals)):
                    sharpe_values = [z for (xi, yi), z in zip(zip(x_vals, y_vals), z_vals) if xi == x and yi == y]
                    if sharpe_values:
                        z_grid[j, i] = max(sharpe_values)

            # Display 3D surface plot for Grid Search CV
            with st.expander("Surface Plot"):
                st.markdown(f'<p class="header-font">3D Surface Plot for {strategy_choice} Strategy (Grid Search CV)</p>', unsafe_allow_html=True)
                fig = go.Figure(data=[go.Surface(x=x_grid, y=y_grid, z=z_grid, colorscale='Viridis')])

                # Add a scatter point for the maximum Sharpe Ratio
                max_sharpe_idx = np.unravel_index(np.argmax(z_grid, axis=None), z_grid.shape)
                max_sharpe_value = z_grid[max_sharpe_idx]
                best_fast_length = x_grid[max_sharpe_idx]
                best_slow_length = y_grid[max_sharpe_idx]

                fig.add_trace(go.Scatter3d(
                    x=[best_fast_length],
                    y=[best_slow_length],
                    z=[max_sharpe_value],
                    mode='markers+text',
                    marker=dict(size=5, color='red'),
                    text=[f'Max Sharpe Ratio: {max_sharpe_value:.4f}'],
                    textposition='top center'
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title=space[0].name,
                        yaxis_title=space[1].name,
                        zaxis_title='Sharpe Ratio'
                    ),
                    autosize=True
                )
                st.plotly_chart(fig)

            # Display heatmap in the "Heat Map" expander
            with st.expander("Heat Map"):
                st.markdown('<p class="header-font">Heatmap of Sharpe Ratio</p>', unsafe_allow_html=True)
                plot_heatmap(results, space[0].name, space[1].name, 'Sharpe Ratio')
                
    # 5. Show the Best parameter combination in a table format
    with st.expander("Best Parameter Combination"):
        if best_params:
            st.markdown('<p class="subheader-font">Best Parameter Combination</p>', unsafe_allow_html=True)
            
            # Convert the best_params dictionary into a DataFrame for neat display
            params_df = pd.DataFrame(list(best_params.items()), columns=["Parameter", "Value"])
            
            # Display the DataFrame as a table
            st.table(params_df)
        else:
            st.write("Run optimization to see results.")
    
    # 6. Show performance metrics inside the "Performance Metrics" expander
    with st.expander("Performance Metrics"):
        st.markdown('<p class="subheader-font">Performance Metrics</p>', unsafe_allow_html=True)
        st.markdown("""
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{:.4f}</td>
                </tr>
                <tr>
                    <td>Average Return</td>
                    <td>{:.2f}%</td>
                </tr>
                <tr>
                    <td>Maximum Drawdown</td>
                    <td>{:.2f}%</td>
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{:.4f}</td>
                </tr>
                <tr>
                    <td>Number of Trades</td>
                    <td>{}</td>
                </tr>
            </table>
            """.format(sharpe_ratio, metrics.get("Average Return", 0), metrics.get("Maximum Drawdown", 0),
                       metrics.get("Calmar Ratio", 0), metrics.get("Number of Trades", 0)),
            unsafe_allow_html=True)


    with st.expander("Equity Curve"):
        st.markdown(f'<p class="header-font">{strategy_choice} Strategy Cumulative PnL</p>', unsafe_allow_html=True)
        plt.figure(figsize=(12, 6))
        plt.plot(best_result['Cumulative PnL (Long Short)'], label='Cumulative PnL (Long Short)', color='red')
        plt.title(f'{strategy_choice} Strategy Cumulative PnL')
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.legend()
        st.pyplot(plt)


