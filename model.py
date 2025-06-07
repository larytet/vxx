# Reload the uploaded CSV file and prepare the data
import pandas as pd
import numpy as np
from scipy.stats import norm

# Load file
df = pd.read_csv("./daily_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Weekly resample
weekly_df = df[['VIX', 'VXX']].resample('W-FRI').last()
weekly_df['VXX_return'] = weekly_df['VXX'].pct_change()
weekly_df['VXX_vol'] = weekly_df['VXX_return'].rolling(window=4).std() * np.sqrt(52)

# Define Black-Scholes call pricing
def black_scholes_call_price(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Strategy logic
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

    if np.isnan(vxx_price) or np.isnan(vxx_vol) or vxx_price <= 0 or vxx_vol <= 0:
        capital_track.append(capital)
        continue

    position_value = 0.02 * capital
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
            put_premium = black_scholes_call_price(current_price, current_price, T_weekly, r, vxx_vol)
            capital += put_premium * short_position
            short_active = False
            short_pnl = 0

    capital_with_pnl = capital + short_pnl
    peak_capital = max(peak_capital, capital_with_pnl)
    capital_track.append(capital_with_pnl)

# Performance metrics
returns = pd.Series(np.diff(capital_track, prepend=initial_capital)) / initial_capital
ann_return = (capital_track[-1] / initial_capital) ** (52 / len(capital_track)) - 1
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(52)
max_drawdown = (pd.Series(capital_track).cummax() - pd.Series(capital_track)).div(pd.Series(capital_track).cummax()).max()

summary = pd.DataFrame([{
    "Annual Return (%)": round(ann_return * 100, 2),
    "Sharpe Ratio": round(sharpe_ratio, 2),
    "Max Drawdown (%)": round(max_drawdown * 100, 2)
}])

print(summary.to_string(index=False))