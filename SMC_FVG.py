import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import ccxt  # 必須先 pip install ccxt
import time
import matplotlib.pyplot as plt # 引入繪圖庫

# ==========================================
# 策略核心：PriceActionSMCStrategy (嚴格版 - 含繪圖數據記錄)
# ==========================================
class PriceActionSMCStrategy(bt.Strategy):
    params = (
        ('fvg_lookback', 3),
        ('retracement_limit', 0.5), # 嚴格 50%
        ('entry_buffer', 0.001),     
    )

    def __init__(self):
        self.orders = None 
        self.trend_dir = 0          
        self.anchor_price = None    
        self.peak_price = None      
        self.retraced_deep = False  
        
        # === 新增：用於繪製資金曲線的列表 ===
        self.equity_curve = []
        self.date_curve = []

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            direction = "買入" if order.isbuy() else "賣出"
            self.log(f'【成交{direction}】價格: {order.executed.price:.2f} | 數量: {order.executed.size:.4f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.orders and order.ref == self.orders[0].ref:
                self.orders = None

    def cancel_all_orders(self):
        if self.orders:
            for o in self.orders:
                if o.status in [bt.Order.Submitted, bt.Order.Accepted]:
                    self.cancel(o)
            self.orders = None

    def next(self):
        # === 新增：記錄每個 K 線結束時的資金與時間 ===
        self.equity_curve.append(self.broker.getvalue())
        self.date_curve.append(self.datas[0].datetime.datetime(0))

        # === 以下為您原本的策略邏輯 (完全未變動) ===
        close = self.datas[0].close[0]
        high = self.datas[0].high[0]
        low = self.datas[0].low[0]
        _open = self.datas[0].open[0]

        # 初始化
        if self.trend_dir == 0:
            if close > _open:
                self.trend_dir = 1
                self.anchor_price = low
                self.peak_price = high
            else:
                self.trend_dir = -1
                self.anchor_price = high
                self.peak_price = low
            return

        if self.position:
            if self.orders: self.orders = None 
            return

        # 1. 結構破壞 (MSB)
        if self.trend_dir == 1:
            if close < self.anchor_price:
                self.log(f'[趨勢反轉] 多 -> 空 (跌破 {self.anchor_price:.2f})')
                self.trend_dir = -1
                self.cancel_all_orders()
                self.anchor_price = self.peak_price 
                self.peak_price = low
                self.retraced_deep = False
                return
        elif self.trend_dir == -1:
            if close > self.anchor_price:
                self.log(f'[趨勢反轉] 空 -> 多 (突破 {self.anchor_price:.2f})')
                self.trend_dir = 1
                self.cancel_all_orders()
                self.anchor_price = self.peak_price
                self.peak_price = high
                self.retraced_deep = False
                return

        # 2. 趨勢延續 & FVG
        if self.trend_dir == 1: # 多頭
            if high > self.peak_price:
                if self.retraced_deep: 
                    self.anchor_price = self.datas[0].low[-1] 
                    self.retraced_deep = False
                    self.cancel_all_orders()
                self.peak_price = high
            
            range_len = self.peak_price - self.anchor_price
            if range_len == 0: return
            discount_limit = self.anchor_price + (1 - self.params.retracement_limit) * range_len
            
            if low < discount_limit:
                self.retraced_deep = True

            bar1_high = self.datas[0].high[-3]
            bar3_low = self.datas[0].low[-1]
            
            if not self.orders and bar3_low > bar1_high: 
                fvg_top = bar1_high
                if fvg_top < discount_limit and fvg_top > self.anchor_price:
                    entry = fvg_top * (1 + self.params.entry_buffer)
                    self.log(f'[訊號] 多頭 FVG {entry:.2f}')
                    self.orders = self.buy_bracket(price=entry, limitprice=self.peak_price, stopprice=self.anchor_price, valid=datetime.timedelta(days=2))

        elif self.trend_dir == -1: # 空頭
            if low < self.peak_price:
                if self.retraced_deep:
                    self.anchor_price = self.datas[0].high[-1]
                    self.retraced_deep = False
                    self.cancel_all_orders()
                self.peak_price = low
            
            range_len = self.anchor_price - self.peak_price
            if range_len == 0: return
            premium_limit = self.peak_price + (1 - self.params.retracement_limit) * range_len
            
            if high > premium_limit:
                self.retraced_deep = True

            bar1_low = self.datas[0].low[-3]
            bar3_high = self.datas[0].high[-1]

            if not self.orders and bar3_high < bar1_low:
                fvg_bot = bar1_low
                if fvg_bot > premium_limit and fvg_bot < self.anchor_price:
                    entry = fvg_bot * (1 - self.params.entry_buffer)
                    self.log(f'[訊號] 空頭 FVG {entry:.2f}')
                    self.orders = self.sell_bracket(price=entry, limitprice=self.peak_price, stopprice=self.anchor_price, valid=datetime.timedelta(days=2))

# ==========================================
# 工具函數：從幣安 (Binance) 下載長歷史數據
# ==========================================
def fetch_binance_data(symbol, timeframe, start_str, end_str):
    exchange = ccxt.binance()
    since = exchange.parse8601(start_str + 'T00:00:00Z')
    end_ts = exchange.parse8601(end_str + 'T00:00:00Z')
    
    all_ohlcv = []
    limit = 1000 

    print(f"正在從 Binance 下載 {timeframe} 數據 (可能需要一點時間)...")
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if len(ohlcv) == 0:
                break
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1 
            time.sleep(0.1) 
        except Exception as e:
            print(f"下載中斷: {e}")
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[df.index <= end_str]
    return df

# ==========================================
# 主程式
# ==========================================
if __name__ == '__main__':
    print("=========================================")
    print("      SMC 策略回測 (含資金曲線圖)        ")
    print("=========================================")
    
    try:
        days_back = int(input("1. 請輸入回測天數 (例如 365): "))
        start_cash = float(input("2. 請輸入初始本金 (例如 1000000): "))
    except:
        days_back = 365
        start_cash = 1000000.0

    tf_input = input("3. 請輸入 K 線週期 (30m, 1h, 4h): ").strip().lower()
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print("\n-----------------------------------------")
    print(f"模式: {tf_input} | 天數: {days_back}")
    print("-----------------------------------------\n")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=90)
    cerebro.addstrategy(PriceActionSMCStrategy)

    data_df = pd.DataFrame()

    # ====== 數據下載邏輯 ======
    if tf_input == '30m' and days_back > 59:
        print("💡 檢測到長週期 30m 需求，切換至 Binance 下載數據...")
        data_df = fetch_binance_data('BTC/USDT', '30m', start_str, end_str)
    else:
        print("💡 使用 Yahoo Finance 下載數據...")
        yf_interval = '1h' if tf_input == '4h' else tf_input
        if tf_input == '30m' and days_back > 59:
            print("⚠️ Yahoo 限制 30m 最多 60 天，已自動修正起始日。")
            real_start = end_date - datetime.timedelta(days=59)
            data_df = yf.download('BTC-USD', start=real_start, end=end_date, interval='30m', progress=False)
        else:
            data_df = yf.download('BTC-USD', start=start_date, end=end_date, interval=yf_interval, progress=False)
        
        if isinstance(data_df.columns, pd.MultiIndex):
            data_df.columns = data_df.columns.get_level_values(0)

    if data_df.empty:
        print("❌ 錯誤：無法下載數據。")
        exit()

    data = bt.feeds.PandasData(dataname=data_df)

    if tf_input == '4h':
        print("模式：重採樣 1h -> 4h")
        cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=60*4)
    else:
        print(f"模式：直接使用 {tf_input} 數據")
        cerebro.adddata(data)

    print("開始回測，請稍候...")
    results = cerebro.run()
    strat = results[0]
    
    print('\n=========================================')
    final_value = cerebro.broker.getvalue()
    roi = ((final_value - start_cash) / start_cash * 100)
    print(f'最終資金: {final_value:.2f}')
    print(f'總回報率: {roi:.2f}%')
    print('=========================================')

    # === 新增：繪製資金曲線圖 ===
    print("正在繪製資金曲線圖...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(strat.date_curve, strat.equity_curve, label='Equity Curve', color='blue')
    
    # 標記初始資金線 (紅色虛線)
    plt.axhline(y=start_cash, color='red', linestyle='--', label='Initial Capital')
    
    plt.title(f'SMC Strategy Performance ({tf_input})', fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
