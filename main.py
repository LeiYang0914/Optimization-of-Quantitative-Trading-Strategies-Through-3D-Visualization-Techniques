import yfinance as yf
from datetime import datetime
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import plotly.graph_objects as go
from bayes_opt import BayesianOptimization

from skopt import gp_minimize
from skopt.space import Integer,Real
from skopt.utils import use_named_args
from plotly.subplots import make_subplots
from sklearn.model_selection import ParameterGrid
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

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
    }
}

# Page Title
st.title("Parameter Optimization UI")

# 1. Upload a CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully!")
    st.write(data.head())
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

# Perform the selected optimization
if st.sidebar.button("Run Optimization"):
    strategy_info = strategy_params[strategy_choice]
    strategy_function = strategy_info['strategy']
    space = strategy_info['space']

    if optimization_method == "Bayesian Optimization":
        @use_named_args(space)
        def objective(**params):
            result = strategy_function(data.copy(), **params)
            metrics = performance_metrics(result, selected_timeframe)
            if np.isnan(metrics['Sharpe Ratio']):
                return 0
            return -metrics['Sharpe Ratio']

        res = gp_minimize(objective, space, n_calls=50, random_state=0)
        best_params = {dim.name: res.x[i] for i, dim in enumerate(space)}
        best_result = strategy_function(data.copy(), **best_params)
        sharpe_ratio = -res.fun

        # Extract the evaluated points and corresponding negative Sharpe Ratios
        x_vals = [point[0] for point in res.x_iters]
        y_vals = [point[1] for point in res.x_iters]
        z_vals = [-val for val in res.func_vals]

    elif optimization_method == "Permutation Testing (Grid Search CV)":
        param_grid = {
            dim.name: list(range(dim.bounds[0], dim.bounds[1] + 1, 5))
            for dim in space
        }

        results = []
        x_vals = []
        y_vals = []
        z_vals = []

        for params in ParameterGrid(param_grid):
            result = strategy_function(data.copy(), **params)
            metrics = performance_metrics(result, selected_timeframe)
            if not np.isnan(metrics['Sharpe Ratio']):
                results.append((params, metrics['Sharpe Ratio']))
                x_vals.append(params[space[0].name])
                y_vals.append(params[space[1].name])
                z_vals.append(metrics['Sharpe Ratio'])

        if results:
            best_params, sharpe_ratio = max(results, key=lambda x: x[1])
            best_result = strategy_function(data.copy(), **best_params)

    # 5. Show the 3D plot
    st.write("### Parameter Optimization 3D Graph")
    if best_params:
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

        fig.update_layout(
            scene=dict(
                xaxis_title=space[0].name,
                yaxis_title=space[1].name,
                zaxis_title='Sharpe Ratio'
            ),
            title=f'3D Visualization of {optimization_method} for {strategy_choice} Strategy',
            autosize=True
        )
        st.plotly_chart(fig)

    # 6. Show the Best parameter combination and Sharpe ratio
    st.write("### Best Parameter Combination:")
    if best_params:
        st.write(f"**Best Parameters:** {best_params}")
        st.write(f"**Sharpe Ratio:** {sharpe_ratio}")
    else:
        st.write("Run optimization to see results.")

    # 7. Show performance metrics
    st.write("### Performance Metrics:")
    metrics = performance_metrics(best_result, selected_timeframe)
    st.write(f"**Average Return:** {metrics.get('Average Return', 'N/A')}")
    st.write(f"**Maximum Drawdown:** {metrics.get('Maximum Drawdown', 'N/A')}")
    st.write(f"**Calmar Ratio:** {metrics.get('Calmar Ratio', 'N/A')}")
    st.write(f"**Number of Trades:** {metrics.get('Number of Trades', 'N/A')}")
