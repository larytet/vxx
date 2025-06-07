'''
I have a systematic short volatility strategy using VXX (iPath S&P 500 VIX Short-Term Futures ETN). Here's how it works:

###  Strategy Overview

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

### 🔍 Objectives

1. **Model the full strategy** over 10 years of weekly data using actual VXX and VIX prices.
2. **Optimize the VIX band thresholds** and allocations in each band to maximize Sharpe ratio.
3. **Use actual option premiums** priced with Black-Scholes using VXX historical volatility.
4. **Measure**:
   - Total return
   - Sharpe ratio
   - Max drawdown
   - Time in drawdown
   - Behavior during VXX spikes


### 📥 How to Download Data from Yahoo Finance

1. Go to [Yahoo Finance](https://finance.yahoo.com)
2. Search for:
   - `^VIX` (CBOE Volatility Index)
   - `VXX` (iPath Series B S&P 500 VIX Short-Term Futures ETN)
3. Click **"Historical Data"** tab
4. Choose **Time Period: Max** and **Frequency: Weekly**
5. Click **Download** and save as HTML (or CSV)


Please simulate this strategy, optimize the thresholds and allocations, and report the results. Use realistic assumptions and model all edge cases (e.g., prolonged spikes in VXX, no put sale condition).
'''

###  Python Code to Process Yahoo HTML Files
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

def extract_yahoo_table(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    tables = soup.find_all('table')
    for table in tables:
        try:
            df = pd.read_html(StringIO(str(table)))[0]
        except Exception:
            continue
        if 'Date' in df.columns:
            close_col = next((c for c in df.columns if 'Close' in c and 'Adj' not in c), None)
            if close_col:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.set_index('Date', inplace=True)
                df['Close'] = pd.to_numeric(df[close_col], errors='coerce')
                return df[['Close']].dropna()
    raise ValueError(f"No suitable table with Date and Close columns found in {file_path}")

# Define file-to-column mapping
files = {
    "VIX_file.html": "VIX",
    "VXX_file.html": "VXX",
    "SPY_file.html": "SPY"
}

# Extract and rename
dataframes = {}
for file, col_name in files.items():
    df = extract_yahoo_table(file)
    df.rename(columns={'Close': col_name}, inplace=True)
    dataframes[col_name] = df

# Merge all dataframes on the index (Date)
daily_data = pd.concat(dataframes.values(), axis=1).dropna().sort_index()

# Optional preview and save
print(daily_data)
daily_data.to_csv("daily_data.csv")
print(f"Saved {len(daily_data)} rows to 'daily_data.csv'")
