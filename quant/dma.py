import numpy as np
import plotly.express as px
import pandas as pd
from data_download import *

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

def moving_avg(data: np.ndarray, window: int):
    ma = np.zeros_like(data)

    for idx in range(len(data) - (window - 1)):
        end_idx = idx + window
        val = sum(data[idx: end_idx]) / window
        ma[end_idx - 1] = val
    
    return ma


if __name__ == "__main__":
    bars = download_stock_daily_data("000001", Exchange.SSE)
    close_prices = np.array([bar.close_price for bar in bars])

    ma10 = moving_avg(close_prices, 10)
    ma20 = moving_avg(close_prices, 20)

    d = {
        "close": close_prices,
        "ma10": ma10,
        "ma20": ma20,
    }
    df = pd.DataFrame(d)
    fig = px.line(df)
    fig.show()

