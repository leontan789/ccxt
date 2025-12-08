import ccxt
import pandas as pd
import numpy as np
import time

# --- 1. 獲取數據 ---
def get_binance_data(symbol='BTC/USDT', timeframe='1h', days=365):
    print(f"正在從 Binance 下載 {symbol} 過去 {days} 天的 {timeframe} 數據...")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_ohlcv = []
    limit = 1000
    
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
        except Exception as e:
            print(f"下載中斷: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[cols] = df[cols].astype(float)
    df = df.drop_duplicates(subset=['Timestamp']).reset_index(drop=True)
    print(f"下載完成！共 {len(df)} 根 K 棒")
    return df

# --- 2. 計算指標 (布林通道 + RSI + EMA) ---
def calculate_indicators(df):
    # EMA 200 (大趨勢過濾：只做順勢的回調)
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # 布林通道 (20, 2.5) - 2.5倍標準差，抓極端行情
    # 標準是用 2.0，改用 2.5 可以減少進場次數，但提高勝率
    window = 20
    std_dev = 2.5
    
    df['Middle_Band'] = df['Close'].rolling(window).mean()
    df['Std'] = df['Close'].rolling(window).std()
    df['Upper_Band'] = df['Middle_Band'] + (df['Std'] * std_dev)
    df['Lower_Band'] = df['Middle_Band'] - (df['Std'] * std_dev)

    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (用於防守止損)
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    return df

# --- 3. 執行策略 (布林回歸 Mean Reversion) ---
def run_strategy(df):
    df = calculate_indicators(df)
    
    initial_capital = 100000.0
    balance = initial_capital
    position = 0 
    entry_price = 0.0
    entry_size = 0.0
    
    sl_price = 0.0
    
    # 參數設定
    risk_per_trade = 0.02       # 2% 風險
    
    win_count = 0
    loss_count = 0
    
    # 統計過濾
    filter_counter_trend = 0

    print(f"\n{'='*10} 回測開始 (布林極限回歸 - 高勝率版) {'='*10}")

    for i in range(200, len(df)):
        curr_date = df['Date'].iloc[i]
        close = df['Close'].iloc[i]
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        
        # 指標
        upper = df['Upper_Band'].iloc[i]
        lower = df['Lower_Band'].iloc[i]
        mid = df['Middle_Band'].iloc[i]
        rsi = df['RSI'].iloc[i]
        ema200 = df['EMA200'].iloc[i]
        atr = df['ATR'].iloc[i]
        
        # 前一根 K 棒 (用於確認收回通道內)
        prev_close = df['Close'].iloc[i-1]
        prev_lower = df['Lower_Band'].iloc[i-1]
        prev_upper = df['Upper_Band'].iloc[i-1]

        # --------------------------------
        # 1. 持倉管理
        # --------------------------------
        if position != 0:
            # A. 止盈：回到中軌 (均值回歸)
            # 這能保證極高的勝率，雖然單次賺得不多，但積少成多
            if position == 1 and high >= mid:
                pnl = (mid - entry_price) * entry_size
                balance += pnl
                win_count += 1
                print(f"[止盈] {curr_date} | 獲利: +{pnl:.2f} (回歸中軌)")
                position = 0
                
            elif position == -1 and low <= mid:
                pnl = (entry_price - mid) * entry_size
                balance += pnl
                win_count += 1
                print(f"[止盈] {curr_date} | 獲利: +{pnl:.2f} (回歸中軌)")
                position = 0
            
            # B. 止損 (防守型)
            elif position == 1 and low <= sl_price:
                pnl = (sl_price - entry_price) * entry_size
                balance += pnl
                loss_count += 1
                print(f"[止損] {curr_date} | 虧損: {pnl:.2f}")
                position = 0
            elif position == -1 and high >= sl_price:
                pnl = (entry_price - sl_price) * entry_size
                balance += pnl
                loss_count += 1
                print(f"[止損] {curr_date} | 虧損: {pnl:.2f}")
                position = 0

        # --------------------------------
        # 2. 進場邏輯 (Mean Reversion)
        # --------------------------------
        if position == 0:
            if pd.isna(ema200) or pd.isna(upper): continue

            # 開多條件 (Buy the Dip)：
            # 1. 價格跌破下軌，並且 RSI < 30 (超賣)
            # 2. 大趨勢向上 (Close > EMA200) -> 避免接Falling Knife
            if low < lower and rsi < 30:
                if close > ema200: 
                    # 計算倉位
                    sl_dist = atr * 3.0 # 給予非常寬的止損空間，防止插針
                    sl_price = close - sl_dist
                    risk_per_share = close - sl_price
                    
                    risk_amount = balance * risk_per_trade
                    if risk_per_share > 0:
                        entry_size = min(risk_amount / risk_per_share, balance / close)
                        position = 1
                        entry_price = close
                        print(f"[開多] {curr_date} | 價:{close:.2f} | 觸底反彈")
                else:
                    filter_counter_trend += 1

            # 開空條件 (Sell the Rip)：
            # 1. 價格漲破上軌，並且 RSI > 70 (超買)
            # 2. 大趨勢向下 (Close < EMA200)
            elif high > upper and rsi > 70:
                if close < ema200:
                    sl_dist = atr * 3.0 
                    sl_price = close + sl_dist
                    risk_per_share = sl_price - close
                    
                    risk_amount = balance * risk_per_trade
                    if risk_per_share > 0:
                        entry_size = min(risk_amount / risk_per_share, balance / close)
                        position = -1
                        entry_price = close
                        print(f"[開空] {curr_date} | 價:{close:.2f} | 觸頂回落")
                else:
                    filter_counter_trend += 1

    # 結算
    final_equity = balance
    if position == 1:
        final_equity += (df['Close'].iloc[-1] - entry_price) * entry_size
    elif position == -1:
        final_equity += (entry_price - df['Close'].iloc[-1]) * entry_size
        
    profit = final_equity - initial_capital
    roi = (profit / initial_capital) * 100
    
    print(f"\n{'='*10} 回測結果 {'='*10}")
    print(f"交易次數: {win_count + loss_count}")
    if (win_count + loss_count) > 0:
        print(f"勝率: {win_count / (win_count + loss_count) * 100:.2f}%")
    else:
        print("無交易發生")
    print(f"逆勢單過濾: {filter_counter_trend} 次")
    print(f"初始本金: ${initial_capital:,.2f}")
    print(f"最終權益: ${final_equity:,.2f}")
    print(f"總盈利  : ${profit:,.2f}")
    print(f"報酬率  : {roi:.2f}%")

if __name__ == "__main__":
    df = get_binance_data(symbol='BTC/USDT', timeframe='1h', days=365)
    if not df.empty:
        run_strategy(df)
    else:
        print("無法獲取數據")