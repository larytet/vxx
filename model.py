# Strategy: 2% max exposure, sell ATM VXX calls weekly
# If assigned → short VXX → wait until VXX ≤ entry price → then sell ATM put to cover
# Also add 4.5% T-bill yield on idle cash

# Initialize
capital = 350_000
initial_capital = capital
capital_track_assigned_puts = []
short_active = False
short_price = 0
short_position = 0
short_pnl = 0
peak_capital = capital
T_weekly = 1 / 52
t_bill_weekly = (1 + 0.045) ** (1 / 52) - 1  # Weekly interest on idle cash

for i in range(len(weekly_df)):
    row = weekly_df.iloc[i]
    vxx_price = row['VXX']
    vxx_vol = row['VXX_vol']

    if np.isnan(vxx_price) or np.isnan(vxx_vol) or vxx_price <= 0 or vxx_vol <= 0:
        capital_track_assigned_puts.append(capital)
        continue

    # Determine position size (2% of capital)
    position_value = 0.02 * capital
    position_size = position_value / vxx_price
    idle_cash = capital - (position_value if not short_active else 0)
    capital += idle_cash * t_bill_weekly

    if not short_active:
        # Sell ATM call
        call_premium = black_scholes_call_price(vxx_price, vxx_price, T_weekly, r, vxx_vol)
        capital += call_premium * position_size

        # Check assignment
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
            # VXX has reverted — sell ATM put to cover
            put_premium = black_scholes_call_price(current_price, current_price, T_weekly, r, vxx_vol)
            capital += put_premium * short_position
            short_active = False
            short_pnl = 0

    capital_with_pnl = capital + short_pnl
    peak_capital = max(peak_capital, capital_with_pnl)
    capital_track_assigned_puts.append(capital_with_pnl)

# Performance metrics
returns_assigned_puts = pd.Series(np.diff(capital_track_assigned_puts, prepend=initial_capital)) / initial_capital
ann_return_assigned_puts = (capital_track_assigned_puts[-1] / initial_capital) ** (52 / len(capital_track_assigned_puts)) - 1
sharpe_assigned_puts = (returns_assigned_puts.mean() / returns_assigned_puts.std()) * np.sqrt(52)
max_dd_assigned_puts = (pd.Series(capital_track_assigned_puts).cummax() - pd.Series(capital_track_assigned_puts)).div(pd.Series(capital_track_assigned_puts).cummax()).max()

ann_return_assigned_puts, sharpe_assigned_puts, max_dd_assigned_puts

