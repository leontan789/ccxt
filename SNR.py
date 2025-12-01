import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# 參數設定
# =========================

exchange = ccxt.binance()

symbol = "BTC/USDT"
timeframe = "1m"
limit = 200               # 取最新 200 根 K 線
update_interval = 10       # 每 10 秒更新一次

signal_ema_span = 12       # SNR 中使用的 signal EMA
var_window = 48            # signal/noise variance window
snr_threshold = 0.4        # SNR 閾值

entry_confirm = 2          # 連續 k 根訊號才進場（較穩定）
exit_snr_floor = 0.2       # SNR 低於此值代表噪聲變高 → 出場

fee = 0.0004               # 雙邊手續費（大約 Binance 現貨）
slippage = 0.0005          # 假設 0.05% 滑點


# =========================
# 函式區
# =========================

def fetch_klines():
    """從 Binance 取得最新 K 線"""
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def compute_snr(df, signal_ema_span=10, var_window=50):
    """計算 SNR 策略所需資訊"""
    df = df.copy()
    df["ret"] = np.log(df["close"]).diff().fillna(0)

    df["signal"] = df["ret"].ewm(span=signal_ema_span, adjust=False).mean()
    df["noise"] = df["ret"] - df["signal"]

    df["signal_var"] = df["signal"].rolling(var_window).var().fillna(0)
    df["noise_var"] = df["noise"].rolling(var_window).var().fillna(1e-9)

    df["snr"] = df["signal_var"] / df["noise_var"]
    return df


def trading_logic(df):
    """
    根據 SNR + Signal 方向決定持倉：
    return 最新 position: 1=多、-1=空、0=平
    """
    d = df.tail(5)   # 看最近 5 根

    snr_now = d["snr"].iloc[-1]
    sig_now = d["signal"].iloc[-1]

    # 基本多空判斷
    long_signal = (snr_now > snr_threshold) and (sig_now > 0)
    short_signal = (snr_now > snr_threshold) and (sig_now < 0)

    # 訊號確認（避免假突破）
    long_confirm = all(df["signal"].tail(entry_confirm) > 0)
    short_confirm = all(df["signal"].tail(entry_confirm) < 0)

    if long_signal and long_confirm:
        return 1
    if short_signal and short_confirm:
        return -1

    # 噪聲變高 → 出場
    if snr_now < exit_snr_floor:
        return 0

    return None  # 無變化

def simulate_trade(position, price, prev_price):
    """單純計算績效變化（適合 VSCode 測試）"""
    if position == 0:
        return 0

    ret = (price - prev_price) / prev_price

    # 加入手續費 + 滑點
    ret -= fee
    ret -= slippage

    return position * ret


# =========================
# 主程式主迴圈（即時執行）
# =========================

print("🔵 SNR BTC 即時策略啟動中...\n")

prev_position = 0
equity = 1.0

while True:
    try:
        df = fetch_klines()
        df = compute_snr(df, signal_ema_span, var_window)

        price = df["close"].iloc[-1]
        prev_price = df["close"].iloc[-2]

        new_position = trading_logic(df)

        # 若策略沒有變化，不動作
        if new_position is None:
            pnl = simulate_trade(prev_position, price, prev_price)
            equity *= (1 + pnl)
            print(f"[{datetime.now()}] price={price:.2f} SNR={df['snr'].iloc[-1]:.3f} pos={prev_position} equity={equity:.4f}")
        else:
            # 策略發出新訊號 → 換倉
            print(f"\n🟡 訊號更新 {prev_position} → {new_position} @ price={price:.2f}\n")
            prev_position = new_position

        time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\n⛔ 停止策略。")
        break

    except Exception as e:
        print(f"❗ Error: {e}")
        time.sleep(5)
