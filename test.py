import numpy as np
import pandas as pd
import yfinance as yf
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Set parameters
symbol = 'BTC-USD'
start_date = '2017-01-01'
end_date = '2022-02-27'
macd_fast = 12
macd_slow = 26
macd_avg = 10
st_atr = 7
st_mul = 3
capital = 100000

# Get data
df = yf.download(symbol, start_date, end_date)

# Calculate MACD
df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['Adj Close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_avg)

def SuperTrend(df, atr_multiplier, atr_period):
    hl2 = (df['High'] + df['Low']) / 2
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period)
    df['Upperband'] = hl2 + (atr_multiplier * df['ATR'])
    df['Lowerband'] = hl2 - (atr_multiplier * df['ATR'])
    df['In Uptrend'] = True
    
    for current in range(1, len(df.index)):
        previous = current - 1
        
        if df['Close'][current] > df['Upperband'][previous]:
            df['Upperband'][current] = df['Upperband'][current] + (atr_multiplier * df['ATR'][current])
            df['Lowerband'][current] = df['Lowerband'][previous] + (atr_multiplier * df['ATR'][current])
        else:
            df['Upperband'][current] = df['Upperband'][previous] - (atr_multiplier * df['ATR'][current])
            df['Lowerband'][current] = df['Lowerband'][current] - (atr_multiplier * df['ATR'][current])
        
        if df['Close'][current] <= df['Upperband'][current] and df['Close'][previous] > df['Upperband'][previous]:
            df['In Uptrend'][current] = False
        elif df['Close'][current] >= df['Lowerband'][current] and df['Close'][previous] < df['Lowerband'][previous]:
            df['In Uptrend'][current] = True
        else:
            df['In Uptrend'][current] = df['In Uptrend'][previous]
    
    return df['Upperband'], df['Lowerband']

# Calculate SuperTrend
df['atr'] = talib.ATR(df['High'], df['Low'], df['Adj Close'], timeperiod=st_atr)
df['upperband'], df['lowerband'] = SuperTrend(df, st_atr, st_mul)
df['trend'] = np.where(df['Adj Close'] > df['upperband'], 1, np.where(df['Adj Close'] < df['lowerband'], -1, df['trend'].shift(1)))

# Generate trading signals
df['prev_macdhist'] = df['macdhist'].shift(1)
df['signal'] = np.where((df['macdhist'] > 0) & (df['prev_macdhist'] <= 0) & (df['trend'] == 1), 1, 
                        np.where((df['macdhist'] < 0) & (df['prev_macdhist'] >= 0) & (df['trend'] == -1), -1, 0))
df['position'] = df['signal'].shift(1)

# Simulate trades
df['pct_change'] = df['Adj Close'].pct_change()
df['strategy_returns'] = df['pct_change'] * df['position']
df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() * capital

# Calculate performance metrics
daily_returns = df['strategy_returns'].dropna()
cumulative_returns = df['cumulative_returns'][-1]
sharpe_ratio = (np.sqrt(252) * daily_returns.mean()) / daily_returns.std()
max_drawdown = (df['cumulative_returns'] / df['cumulative_returns'].cummax() - 1).min()

# Plot performance
fig, ax = plt.subplots(2, 1, figsize=(16,12))

ax[0].plot(df.index, df['cumulative_returns'], label='Strategy')
ax[0].plot(df.index, capital * df['Adj Close'] / df['Adj Close'][0], label='Buy & Hold')
ax[0].set_title('Cumulative Returns')
ax[0].legend(loc='upper left')

sns.lineplot(x=df.index, y=df['macdhist'], ax=ax[1])
ax[1].axhline(y=0, color='r', linestyle='-')
ax[1].set_title('MACD Histogram')
ax[1].legend(loc='upper left')

plt.show()

print(f"Performance Metrics:\nCumulative Returns: {cumulative_returns:.2f}\nSharpe Ratio: {sharpe_ratio:.2f}\nMax Drawdown:{max_drawdown:.2f}")
