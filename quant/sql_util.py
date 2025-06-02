import pandas as pd
from csv import DictReader
import pytz
from datetime import datetime
from vnpy.trader.setting import SETTINGS
SETTINGS["database.timezone"] = "Asia/Shanghai"
SETTINGS["database.name"] = "mysql"
SETTINGS["database.database"] = "vnpy"
SETTINGS["database.host"] = "localhost"
SETTINGS["database.port"] = 3306
SETTINGS["database.user"] = "root"
SETTINGS["database.password"] = "1p0q2o9w3i8e@"
SETTINGS["database.authentication_sour"] = ""

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.trader.database import get_database
from sqlalchemy import create_engine

database_manager = get_database()

engine = create_engine("mysql+pymysql://root:1p0q2o9w3i8e%40@127.0.0.1:3306/vnpy")
df = pd.read_csv('./data/600072.csv')
# df = pd.read_sql(query, engine)
df.to_sql("stock_600072", con=engine ,schema="vnpy")
