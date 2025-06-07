'''
1. **Sell ATM Weekly Calls on VXX**:
   - I sell at-the-money VXX call options every week.
   - Position size is determined by current VIX level:
     - VIX < LOW → allocate X% of capital
     - VIX between LOW and HIGH → allocate Y%
     - VIX > HIGH → allocate Z%
   - Option premium is modeled using the Black-Scholes formula, with volatility estimated from VXX historical returns.

2. **Assignment Handling**:
   - If VXX rises and the call finishes ITM, I am assigned and become **short VXX**.
   - I do **not cover the short immediately**.

3. **Covering the Short**:
   - I wait until VXX falls back to or below my original short price.
   - At that point, I sell an **ATM VXX put** with size equal to my short position.
   - If the put is exercised, it covers my short.

4. **No Forced Covering**:
   - If VXX remains above my short price, I do **nothing** and wait it out.
   - The position is collateralized and I avoid taking realized losses by never covering high.
'''

import argparse
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ------------------------------
# Parse command-line arguments
# ------------------------------
parser = argparse.ArgumentParser(description="VXX Covered Call Strategy")
parser.add_argument('--exposure', type=float, default=0.05, help='Fractional capital exposure per position (e.g. 0.02 for 2%)')
parser.add_argument('--disable_ema_filter', action='store_true', help='Disable SPY EMA 20/80 filter for call selling')
args = parser.parse_args()
exposure_pct = args.exposure
disable_ema_filter = args.disable_ema_filter

# ------------------------------
# Load and prepare data
# ------------------------------
df = pd.read_csv("./daily_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate SPY EMA 20 and EMA 80
df['EMA_20'] = df['SPY'].ewm(span=20, adjust=False).mean()
df['EMA_80'] = df['SPY'].ewm(span=80, adjust=False).mean()
df['EMA_signal'] = df['EMA_20'] > df['EMA_80']

# Aggregate weekly
weekly_df = df[['VIX', 'VXX', 'EMA_signal']].resample('W-FRI').last()
weekly_df['VXX_return'] = weekly_df['VXX'].pct_change()
weekly_df['VXX_vol'] = weekly_df['VXX_return'].rolling(window=4).std() * np.sqrt(52)
if disable_ema_filter:
    weekly_df['EMA_signal'] = True

# ------------------------------
# Black-Scholes pricing function
# ------------------------------
def black_scholes_call_price(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def plot_pnl(capital_track, start_date='2010-01-01'):
    """Plot PnL over time using capital_track values."""
    dates = pd.date_range(start=start_date, periods=len(capital_track), freq='W-FRI')
    plt.figure(figsize=(12, 6))
    plt.plot(dates, capital_track, label='Capital / PnL')
    plt.title('PnL Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# Strategy Execution
# ------------------------------
capital = 350_000
initial_capital = capital
capital_track = []
short_active = False
short_price = 0
short_position = 0
short_pnl = 0
peak_capital = capital
T_weekly = 1 / 52
r = 0.01
t_bill_weekly = (1 + 0.045) ** (1 / 52) - 1

for i in range(len(weekly_df)):
    row = weekly_df.iloc[i]
    vxx_price = row['VXX']
    vxx_vol = row['VXX_vol']
    allow_trade = row['EMA_signal']

    if np.isnan(vxx_price) or np.isnan(vxx_vol) or not allow_trade or vxx_price <= 0 or vxx_vol <= 0:
        capital_track.append(capital)
        continue

    position_value = exposure_pct * capital
    position_size = position_value / vxx_price
    idle_cash = capital - (position_value if not short_active else 0)
    capital += idle_cash * t_bill_weekly

    if not short_active:
        call_premium = black_scholes_call_price(vxx_price, vxx_price, T_weekly, r, vxx_vol)
        capital += call_premium * position_size

        if i + 1 < len(weekly_df):
            next_price = weekly_df.iloc[i + 1]['VXX']
            if next_price > vxx_price:
                short_price = vxx_price
                short_position = position_size
                short_active = True
    else:
        current_price = vxx_price
        short_pnl = (short_price - current_price) * short_position

        if current_price <= short_price:
            put_premium = black_scholes_put_price(current_price, current_price, T_weekly, r, vxx_vol)
            capital += put_premium * short_position
            short_active = False
            short_pnl = 0

    capital_with_pnl = capital + short_pnl
    peak_capital = max(peak_capital, capital_with_pnl)
    capital_track.append(capital_with_pnl)

# ------------------------------
# Performance Metrics
# ------------------------------
returns = pd.Series(np.diff(capital_track, prepend=initial_capital)) / initial_capital
ann_return = (capital_track[-1] / initial_capital) ** (52 / len(capital_track)) - 1
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(52)
max_drawdown = (pd.Series(capital_track).cummax() - pd.Series(capital_track)).div(pd.Series(capital_track).cummax()).max()
expected_pnl = initial_capital * ann_return

plot_pnl(capital_track, start_date=weekly_df.index[0].strftime('%Y-%m-%d'))

summary = pd.DataFrame([{
    "Annual Return (%)": round(ann_return * 100, 2),
    "Sharpe Ratio": round(sharpe_ratio, 2),
    "Max Drawdown (%)": round(max_drawdown * 100, 2),
    "Annualized Expected PnL ($)": round(expected_pnl, 2),
    "Short Active": short_active,
    "Short Entry Price": round(short_price, 2) if short_active else None,
    "Short Position Size": round(short_position, 2) if short_active else None
}])

print(summary.to_string(index=False))
