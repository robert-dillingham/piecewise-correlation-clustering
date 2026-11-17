# import dependencies
import requests
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime

# get data - using stocks as an example but can be any time series with categories (e.g. symbols)
url = "https://en.wikipedia.org/wiki/S%26P_100"
headers = {"User-Agent": "myproject/0.1 (robert.c.dillingham@gmail.com)"}

resp = requests.get(url, headers=headers, timeout=30)
resp.raise_for_status()

tables = pd.read_html(StringIO(resp.text))
sp100 = next(t for t in tables if "Symbol" in t.columns)
symbols = sp100["Symbol"].tolist()

# limit data for testing
symbols = symbols[:10]

# get data
client = StockHistoricalDataClient("PKNLL27KIDVAKW4A4KYGIFJ25Q", "C36MzmRaJ4HK8VT49Rviqc8j6L62TPL3jgNVsL81mTyJ")
req = StockBarsRequest(
    symbol_or_symbols=symbols[:10],  # batch in chunks of ~100-200
    timeframe=TimeFrame.Hour,
    start=datetime(2026, 1, 1),
    feed=DataFeed.IEX,
)
df = client.get_stock_bars(req).df

# assign further groupings & weights
# the 'keys' dict tells the function how the data should be summarized and the weights to apply to each for the weighted average matrix
keys = {'hourly': 4, 'daily': 3, 'weekly': 2, 'monthly': 1}
keys = {j[0]: j[1]/sum([i for i in keys.values()]) for j in keys.items()}

# keys need to map to a specific 'freq' value for the .groupby() function
operations = {
    'hourly': "h"
    , 'daily': "D"
    , 'weekly': "W"
    , 'monthly': "ME"
}

# because we're using US stocks, convert into NYC time zone
data = df["high"].tz_convert("America/New_York", level="timestamp")
dt = {}

# standardize each dataset, using the "High" for each stock
for tick in symbols[:10]:
    dt[tick] = {'high': data.loc[tick], 'stand': 1 + ((data.loc[tick] - max(data.loc[tick])) / max(data.loc[tick]))}

# make a dictionary entry for each summary level under each symbol. Interpolate any NaN values
for k in keys:

    for tick in symbols:
        dt[tick][k] = dt[tick]['stand'].groupby(
            pd.Grouper(level="timestamp", freq=operations[k])
        ).mean()
        dt[tick][k] = dt[tick][k].interpolate()

# make a dictionary entry for each summary level containing a matrix of r^2 values between each symbol
corr_dict = {}
for k in keys:
    m = pd.DataFrame(0.0, index=symbols, columns=symbols)

    for s1 in symbols:

        for s2 in symbols:
            m.loc[s1, s2] = dt[s1][k].corr(dt[s2][k]) ** 2

    corr_dict[k] = m

# create a final matrix of weighted average r^2 values to be used as clustering & predictive values
corr_dict['result'] = sum([m[0][1]*m[1][1] for m in zip(corr_dict.items(), keys.items())])
        