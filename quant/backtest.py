from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.strategies.king_keltner_strategy import KingKeltnerStrategy
from vnpy_ctastrategy.strategies.double_ma_strategy import DoubleMaStrategy
from vnpy_ctastrategy.strategies.multi_timeframe_strategy import MultiTimeframeStrategy
from datetime import datetime

engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="IF888.CFFEX",
    interval="1m",
    start=datetime(2020, 1, 1),
    end=datetime(2020, 9, 30),
    # 手续费，万分之0.3
    rate=0.3/10000,
    #滑点，0.2
    slippage=0.2,
    # 合约数量 300
    size=300,
    #最小变动单位 0.2
    pricetick=0.2,
    # 开始回测资金 100万
    capital=1000000,
)
engine.add_strategy(MultiTimeframeStrategy, {})
engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()


