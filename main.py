import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta 
import requests
import pytz
import ccxt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SMART UPDATER
# ==========================================

file_name = 'BTC_1H_Historical.csv'
end_date = datetime.now()

print("--- INITIALIZING DATA ENGINE ---")

# 1. Load your safe, read-only historical anchor
df_btc = pd.read_csv(file_name, index_col=0)
df_btc.index = pd.to_datetime(df_btc.index, utc=True).tz_localize(None)

# 2. Fetch the live market reality directly from the exchange
print("[INFO] Fetching live data directly from Binance...")
binance = ccxt.binanceus({'enableRateLimit': True})

try:
    # Grab the last 200 hours of live data
    ohlcv = binance.fetch_ohlcv('BTC/USDT', '1h', limit=200)
    
    if ohlcv:
        new_data = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        new_data['Datetime'] = pd.to_datetime(new_data['Datetime'], unit='ms')
        new_data.set_index('Datetime', inplace=True)
        
        # Find exactly where the Binance data begins...
        earliest_binance_time = new_data.index[0]
        # ...and mercilessly delete any CSV rows that overlap it or claim to be in the "future"
        df_btc = df_btc[df_btc.index < earliest_binance_time]
        
        # 3. Stitch them together IN MEMORY. Binance is now guaranteed to be at the absolute end.
        df_btc = pd.concat([df_btc, new_data])
        df_btc.sort_index(inplace=True)
        print("[SUCCESS] Successfully stitched live Binance data in memory!")
except Exception as e:
    print(f"[ERROR] Error fetching Binance data: {e}")

# 4. Fetch the Global Macro Data
print("[INFO] Fetching Global Macro Context (SPY & DXY from Yahoo)...")
macro_start = end_date - timedelta(days=729)
spy_data = yf.download('SPY', start=macro_start, end=end_date, interval='1h', progress=False)
dxy_data = yf.download('DX-Y.NYB', start=macro_start, end=end_date, interval='1h', progress=False)

if isinstance(spy_data.columns, pd.MultiIndex): spy_data.columns = spy_data.columns.get_level_values(0)
if isinstance(dxy_data.columns, pd.MultiIndex): dxy_data.columns = dxy_data.columns.get_level_values(0)

spy_data.index = pd.to_datetime(spy_data.index, utc=True).tz_localize(None)
dxy_data.index = pd.to_datetime(dxy_data.index, utc=True).tz_localize(None)

spy_data['SPY_Return'] = spy_data['Close'].pct_change()
dxy_data['DXY_Return'] = dxy_data['Close'].pct_change()

df_btc = df_btc.join(spy_data[['SPY_Return']], how='left')
df_btc = df_btc.join(dxy_data[['DXY_Return']], how='left')

df_btc['SPY_Return'] = df_btc['SPY_Return'].ffill().fillna(0)
df_btc['DXY_Return'] = df_btc['DXY_Return'].ffill().fillna(0)

data = df_btc.copy()

# ==========================================
# 2. FEATURE ENGINEERING & TREND FILTER
# ==========================================
print("[INFO] Calculating Institutional Technical Indicators...")

data.rename(columns={'Open': 'BTC_Open', 'High': 'BTC_High', 'Low': 'BTC_Low', 'Close': 'BTC_Close', 'Volume': 'BTC_Volume'}, inplace=True)

data['EMA_9'] = data['BTC_Close'].ewm(span=9, adjust=False).mean()
data['EMA_21'] = data['BTC_Close'].ewm(span=21, adjust=False).mean()

# --- THE NEW TREND FILTER ---
data['EMA_200'] = data['BTC_Close'].ewm(span=200, adjust=False).mean() 

data['ROC_4h'] = data['BTC_Close'].pct_change(periods=4)
data['ROC_24h'] = data['BTC_Close'].pct_change(periods=24) 

high_low = data['BTC_High'] - data['BTC_Low']
high_close = np.abs(data['BTC_High'] - data['BTC_Close'].shift())
low_close = np.abs(data['BTC_Low'] - data['BTC_Close'].shift())
data['True_Range'] = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
data['ATR_14'] = data['True_Range'].rolling(14).mean()
data['ATR_Pct'] = data['ATR_14'] / data['BTC_Close'] 

hours = data.index.hour
data['Hour_Sin'] = np.sin(2 * np.pi * hours / 24)
data['Hour_Cos'] = np.cos(2 * np.pi * hours / 24)

data['Dist_EMA9'] = (data['BTC_Close'] - data['EMA_9']) / data['EMA_9']
data['Dist_EMA21'] = (data['BTC_Close'] - data['EMA_21']) / data['EMA_21']
data['Return_1h'] = data['BTC_Close'].pct_change()

# 1. Relative Strength Index (Is it overbought or oversold?)
data['RSI_14'] = ta.rsi(data['BTC_Close'], length=14)

# 2. On-Balance Volume (Is big institutional money flowing in or out?)
data['OBV'] = ta.obv(data['BTC_Close'], data['BTC_Volume'])

# 3. MACD (Is the immediate trend accelerating or dying?)
macd = ta.macd(data['BTC_Close'], fast=12, slow=26, signal=9)
data['MACD'] = macd['MACD_12_26_9']
data['MACD_Histogram'] = macd['MACDh_12_26_9']

# FIBONACCI ALGORITHMIC MEMORY
lookback = 72
data['Rolling_High'] = data['BTC_High'].rolling(window=lookback).max()
data['Rolling_Low'] = data['BTC_Low'].rolling(window=lookback).min()

move_range = data['Rolling_High'] - data['Rolling_Low']
data['Fib_618_Support'] = data['Rolling_High'] - (move_range * 0.618)
data['Fib_618_Resistance'] = data['Rolling_Low'] + (move_range * 0.618)

dist_to_support = np.abs(data['BTC_Close'] - data['Fib_618_Support']) / data['BTC_Close']
dist_to_resistance = np.abs(data['BTC_Close'] - data['Fib_618_Resistance']) / data['BTC_Close']
data['Dist_to_Golden_Pocket'] = np.minimum(dist_to_support, dist_to_resistance)

# ==========================================
# 3. TARGET CREATION & DATA CLEANUP
# ==========================================
print("[INFO] Structuring targets and creating train/test bounds...")

future_out = 6  
data['Future_Close'] = data['BTC_Close'].shift(-future_out)
data['Target'] = np.where(data['Future_Close'] > (data['BTC_Close'] * 1.001), 1, 0)

live_prediction_data = data.iloc[-1:].copy()
features_df = data.dropna()

feature_columns = [
    'Return_1h', 'Dist_EMA9', 'Dist_EMA21', 
    'ROC_4h', 'ROC_24h', 'ATR_Pct', 
    'Hour_Sin', 'Hour_Cos', 
    'SPY_Return', 'DXY_Return', 'Dist_to_Golden_Pocket',
    'RSI_14', 'OBV', 'MACD', 'MACD_Histogram'
]

X = features_df[feature_columns].values
y = features_df['Target'].values

# ==========================================
# 4. TRAIN / TEST SPLIT & STRICT SCALING
# ==========================================
train_size = int(len(X) * 0.85)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. PRO XGBOOST
# ==========================================
print("[INFO] Training Upgraded XGBoost Model...")

xgb_base = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,     
    max_depth=6,            
    subsample=0.75,
    colsample_bytree=0.75,
    gamma=2.0,              
    reg_alpha=0.5,          
    reg_lambda=2.0,        
    random_state=42, 
    n_jobs=-1, 
    eval_metric='auc'       
)
xgb_base.fit(X_train_scaled, y_train)
best_model = xgb_base

# ==========================================
# 6. THRESHOLD PREDICTION (WITH TREND FILTER)
# ==========================================
X_live = live_prediction_data[feature_columns].values
X_live_scaled = scaler.transform(X_live)

probabilities = best_model.predict_proba(X_live_scaled)[0]
confidence_down = probabilities[0]
confidence_up = probabilities[1]

TRADE_THRESHOLD = 0.56 

# --- PULL THE LIVE DATA POINTS ---
last_price = float(live_prediction_data['BTC_Close'].values[0])
current_high = float(live_prediction_data['BTC_High'].values[0])
current_low = float(live_prediction_data['BTC_Low'].values[0])
hourly_atr = float(live_prediction_data['ATR_14'].values[0])
ema_200 = float(live_prediction_data['EMA_200'].values[0])

# --- APPLY THE LOGIC & FILTERS ---
status_message = "Market is too choppy. Staying in cash."

if confidence_up >= TRADE_THRESHOLD:
    if last_price < ema_200:
        prediction = -1
        confidence = confidence_up
        direction = "NO TRADE (TREND FILTER)"
        status_message = f"Price (${last_price:.0f}) is below 200 EMA (${ema_200:.0f}). Longs disabled."
    else:
        prediction = 1
        confidence = confidence_up
        direction = "LONG (BUY)"
        
elif confidence_down >= TRADE_THRESHOLD:
    if last_price > ema_200:
        prediction = -1
        confidence = confidence_down
        direction = "NO TRADE (TREND FILTER)"
        status_message = f"Price (${last_price:.0f}) is above 200 EMA (${ema_200:.0f}). Shorts disabled."
    else:
        prediction = 0
        confidence = confidence_down
        direction = "SHORT (SELL)"
else:
    prediction = -1
    confidence = max(confidence_up, confidence_down)
    direction = "NO TRADE (CONFIDENCE TOO LOW)"

# ==========================================
# 7. EXECUTION, LOGGING, CHARTING & TELEGRAM
# ==========================================

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_SECRET')

ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist)
time_str = current_time_ist.strftime('%Y-%m-%d %I:%M %p')

# ------------------------------------------
# A. BINANCE TESTNET EXECUTION ENGINE
# ------------------------------------------
trade_status = "No Execution Attempted."
TRADE_SIZE_USDT = 100.0  # Spending $100 of fake money per trade

if prediction != -1:
    print("[INFO] Attempting Execution on Binance Testnet...")
    try:
        # Initialize Execution Exchange (Standard CCXT, but Sandbox Mode)
        exec_exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        exec_exchange.set_sandbox_mode(True) # CRITICAL: FAKE MONEY MODE

        symbol = 'BTC/USDT'
        raw_qty = TRADE_SIZE_USDT / last_price
        qty = exec_exchange.amount_to_precision(symbol, raw_qty)
        
        if prediction == 1: # LONG EXECUTION
            tp_price = float(exec_exchange.price_to_precision(symbol, last_price * 1.01))
            sl_price = float(exec_exchange.price_to_precision(symbol, last_price * 0.995))
            
            # 1. Market Buy Entry
            entry_order = exec_exchange.create_market_buy_order(symbol, qty)
            
            # 2. Place OCO Sell (Take Profit + Stop Loss simultaneously)
            oco_order = exec_exchange.create_order(
                symbol, 'limit', 'sell', qty, tp_price, 
                params={'stopPrice': sl_price}
            )
            trade_status = f"✅ LONG Executed at ${last_price:.2f}\nTP: ${tp_price} | SL: ${sl_price}"
            
        elif prediction == 0: # SHORT EXECUTION
            trade_status = f"⚠️ SHORT Signal. Spot Testnet does not allow shorting without margin. Skipped physical execution."
            
        print(f"[SUCCESS] {trade_status}")
            
    except Exception as e:
        trade_status = f"❌ EXECUTION FAILED: {e}"
        print(trade_status)


# ------------------------------------------
# B. UPDATE THE PERMANENT TRADE CSV LOG
# ------------------------------------------
log_file = 'trade_log.csv'
log_entry = pd.DataFrame([{
    'Datetime': time_str,
    'Price': last_price,
    'High': current_high,     
    'Low': current_low,       
    'Prediction': "LONG" if prediction == 1 else "SHORT" if prediction == 0 else "NONE",
    'Confidence': round(confidence * 100, 1)
}])

if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    log_df = pd.concat([log_df, log_entry], ignore_index=True)
else:
    log_df = log_entry.copy()

if len(log_df) > 100:
    log_df = log_df.tail(100)

log_df.to_csv(log_file, index=False)
print("[INFO] Trade logged to CSV successfully.")

# ------------------------------------------
# C. DRAW THE PERFORMANCE CHART
# ------------------------------------------
try:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(range(len(log_df)), log_df['Price'], color='white', linewidth=1.5, alpha=0.8, label="BTC Price")
    
    for i, row in log_df.iterrows():
        if row['Prediction'] == 'LONG':
            ax.scatter(i, row['Price'] - 50, color='lime', marker='^', s=100, zorder=5)
        elif row['Prediction'] == 'SHORT':
            ax.scatter(i, row['Price'] + 50, color='red', marker='v', s=100, zorder=5)

    ax.set_title(f"BTC/USDT Algorithm Tracker (Last {len(log_df)} Hours)", color='gold', fontsize=14)
    ax.set_ylabel("Price (USDT)")
    ax.grid(color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc="upper left")
    
    chart_filename = 'chart.png'
    plt.tight_layout()
    plt.savefig(chart_filename, dpi=150)
    plt.close()
    print("[INFO] Chart generated successfully.")
except Exception as e:
    print(f"[ERROR] Chart generation failed: {e}")

# ------------------------------------------
# D. SEND THE MESSAGE & CHART TO TELEGRAM
# ------------------------------------------
message = f"*BTC ALGO UPDATE*\n"
message += f"Time: {time_str}\n\n"
message += f"Asset: BTC/USDT (Testnet)\n"
message += f"Current Model Price: ${last_price:.2f}\n"

if prediction == -1:
    message += f"\n*NO TRADE*\n"
    message += f"Confidence: {confidence*100:.1f}%\n"
    message += f"Status: {status_message}"
else:
    prob_easy = confidence * 0.85
    
    message += f"\n*SIGNAL:* {direction}\n"
    message += f"Confidence: *{confidence * 100:.1f}%*\n"
    message += f"Win Prob (1% Target): {prob_easy*100:.1f}%\n\n"
    
    message += f"*SETUP ZONES*\n"
    if prediction == 1: 
        message += f"Entry/Pullback: ${last_price - (hourly_atr * 0.5):.2f}\n"
        message += f"Take Profit (1%): ${last_price * 1.01:.2f}\n"
        message += f"Stop Loss (-0.5%): ${last_price * 0.995:.2f}\n\n"
    else: 
        message += f"Entry/Pullback: ${last_price + (hourly_atr * 0.5):.2f}\n"
        message += f"Take Profit (1%): ${last_price * 0.99:.2f}\n"
        message += f"Stop Loss (+0.5%): ${last_price * 1.005:.2f}\n\n"
        
    message += f"⚙️ *EXECUTION LOG*\n{trade_status}"

print(message)

# Send Text Message
url_text = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
requests.post(url_text, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

# Send The Chart Picture!
if os.path.exists(chart_filename):
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(chart_filename, 'rb') as photo:
        requests.post(url_photo, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": photo})
