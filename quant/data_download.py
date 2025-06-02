from datetime import datetime, timedelta, timezone
import requests
import pytz
import pandas as pd
from pytz import timezone
from time import sleep
import rqdatac

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.trader.database import DB_TZ, get_database

def download_binance_minute_data(symbol: str, start: str, end: str):
    base = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    url = base + endpoint
    CHINA_TZ = timezone("Asia/Shanghai")

    start_dt = datetime.strptime(start, "%Y%m%d")
    start_dt = CHINA_TZ.localize(start_dt)

    end_dt = datetime.strptime(end, "%Y%m%d")
    end_dt = CHINA_TZ.localize(end_dt)

    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }

    bar_data = {}
    finish = False

    while True:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": int(start_dt.timestamp()*1000),
            "limit": 1000
        }

        r = requests.get(url, params=params, proxies=proxies)
        data = r.json()

        if data:
            for l in data:
                dt = datetime.fromtimestamp(l[0]/1000)
                dt = CHINA_TZ.localize(dt)

                if dt > end_dt:
                    finish = True
                    break

                bar = BarData(
                    symbol=symbol,
                    exchange=Exchange.BINANCE,
                    datetime=CHINA_TZ.localize(dt),
                    interval=Interval.MINUTE,
                    open_price=float(l[1]),
                    high_price=float(l[2]),
                    low_price=float(l[3]),
                    close_price=float(l[4]),
                    volume=float(l[5]),
                    gateway_name="BINANCE"
                )
                bar_data[bar.datetime] = bar
            
            print(start_dt, bar.datetime)
        else:
            finish = True
        
        if finish:
            break
    dts = list(bar_data.keys())
    dts.sort()
    
    return [bar_data[dt] for dt in dts]


def download_future_minute_data(symbol: str, start: str, end: str):
    username = ""
    password = ""

    rqdatac.init(username=username, password=password)
    # df = rqdatac.all_instruments(type="Future")
    # df.head()

    df = rqdatac.get_price(
        symbol,
        start_date = start,
        end_date = end,
        frequency = "1m"
    )

    return df


def download_stock_daily_data(symbol: str, exchange: Exchange):
    exchange_map = {
        Exchange.SSE: "sh",
        Exchange.SZSE: "sz"
    }
    exchange_str = exchange_map[exchange]

    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={exchange_str}{symbol},day,2024-03-01,2024-12-01,2000,qfq"
    r = requests.get(url)
    tmp = r.json()["data"][f"{exchange_str}{symbol}"]
    qfqday = tmp["day"]

    bar_data = []
    for rd in qfqday:
        api_time = datetime.strptime(rd[0], "%Y-%m-%d")
        local_tz = pytz.timezone("Asia/Shanghai")
        dt = local_tz.localize(api_time)

        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=Interval.DAILY,
            open_price=float(rd[1]),
            close_price=float(rd[2]),
            high_price=float(rd[3]),
            low_price=float(rd[4]),
            volume=float(rd[5]),
            gateway_name="国泰君安证券"
        )
        bar_data.append(bar)

    return bar_data


def move_df_to_sqlite(imported_data:pd.DataFrame, datetime_format:str, collection_name:str=None):
    database_manager = get_database()
    bars = []
    start = None
    count = 0

    for row in imported_data.itertuples():
        api_time = datetime.strptime(row.datetime, datetime_format)
        local_tz = pytz.timezone("Asia/Shanghai")
        dt = local_tz.localize(api_time)

        bar = BarData(
              symbol=row.symbol,
              exchange=row.exchange,
              datetime=dt,
              interval=row.interval,
              volume=row.volume,
              open_price=row.open,
              high_price=row.high,
              low_price=row.low,
              close_price=row.close,
              open_interest=row.open_interest,
              gateway_name="DB",

        )
        bars.append(bar)

        # do some statistics
        count += 1
        if not start:
            start = bar.datetime
    end = bar.datetime

    # insert into database
    database_manager.save_bar_data(bars)
    print(f"Insert Bar: {count} from {start} - {end}")


def save_csv_data(interval: Interval, exchange: Exchange):
    df = pd.read_csv('/home/bob/dataset/quant/test/if.csv')
    df['exchange'] = exchange
    df['interval'] = interval

    float_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
    for col in float_cols:
        df[col] = df[col].astype('float')

    datetime_format = '%Y-%m-%d %H:%M:%S'
    # df['datetime'] = pd.to_datetime(df['datetime'],format=datetime_format)
    # df['datetime'] = df['datetime'].dt.to_pydatetime()
    # df['datetime'] = df['datetime'].apply(lambda x: datetime.strptime(x, datetime_format))

    # print(df.head(10))
    # print(f"Total number of rows: {len(df)}")
    # print(df.count())

    move_df_to_sqlite(df, datetime_format)


if __name__ == "__main__":
    # rqdatac.init()
    # df = rqdatac.all_instruments(type="Future")
    # df.head()

    # database_manager = get_database()

    # bar_data = download_stock_daily_data("600072", Exchange.SSE)
    # bar_data = download_binance_minute_data("BTCUSDT", "20240917", "20240920")
    # df_data = download_future_minute_data("IF2106", "2024-10-01", "2024-10-10")

    # df = pd.DataFrame.from_records([bar.__dict__ for bar in bar_data])
    # df = pd.DataFrame.from_records([{"close": bar.close_price} for bar in bar_data])

    # df.to_csv("./data/600072.csv")
    # df.to_csv("./data/BTC_close.csv", float_format="%.3f")
    # df_data.to_csv("./data/IF2106.csv")

    save_csv_data(interval=Interval.MINUTE, exchange=Exchange.CFFEX)
