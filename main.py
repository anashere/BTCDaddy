import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta 
import requests
import pytz
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SMART UPDATER (NOW WITH MACRO DATA)
# ==========================================
stock_ticker = 'BTC-USD'
file_name = 'BTC_1H_Historical.csv'
end_date = datetime.now()

print(f"--- INITIALIZING DATA ENGINE ---")

df_btc = pd.read_csv(file_name, index_col=0)
df_btc.index = pd.to_datetime(df_btc.index, utc=True).tz_localize(None)

print("🌍 Fetching Global Macro Context (SPY & DXY)...")
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
# 2. FEATURE ENGINEERING
# ==========================================
print("⚙️ Calculating Institutional Technical Indicators...")

data.rename(columns={'Open': 'BTC_Open', 'High': 'BTC_High', 'Low': 'BTC_Low', 'Close': 'BTC_Close', 'Volume': 'BTC_Volume'}, inplace=True)

data['EMA_9'] = data['BTC_Close'].ewm(span=9, adjust=False).mean()
data['EMA_21'] = data['BTC_Close'].ewm(span=21, adjust=False).mean()
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

# ==========================================
# 3. TARGET CREATION & DATA CLEANUP
# ==========================================
print("🚀 Structuring targets and creating train/test bounds...")

future_out = 6  
data['Future_Close'] = data['BTC_Close'].shift(-future_out)
data['Target'] = np.where(data['Future_Close'] > (data['BTC_Close'] * 1.001), 1, 0)

live_prediction_data = data.iloc[-1:].copy()
features_df = data.dropna()

feature_columns = [
    'Return_1h', 'Dist_EMA9', 'Dist_EMA21', 
    'ROC_4h', 'ROC_24h', 'ATR_Pct', 
    'Hour_Sin', 'Hour_Cos', 
    'SPY_Return', 'DXY_Return'
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
# 5. AUTO-TUNE XGBOOST (GRID SEARCH)
# ==========================================
print("🧠 Auto-Tuning XGBoost Parameters (This will take a minute...)")
start_time = time.time()

from sklearn.model_selection import RandomizedSearchCV

xgb_base = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

param_grid = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=20,          
    cv=3,               
    scoring='accuracy',
    verbose=0,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_scaled, y_train)
best_model = search.best_estimator_

# ==========================================
# 6. THRESHOLD PREDICTION (THE SNIPER ALGO)
# ==========================================
X_live = live_prediction_data[feature_columns].values
X_live_scaled = scaler.transform(X_live)

probabilities = best_model.predict_proba(X_live_scaled)[0]
confidence_down = probabilities[0]
confidence_up = probabilities[1]

TRADE_THRESHOLD = 0.56 

last_price = float(live_prediction_data['BTC_Close'].values[0])
hourly_atr = float(live_prediction_data['ATR_14'].values[0])

if confidence_up >= TRADE_THRESHOLD:
    prediction = 1
    confidence = confidence_up
    direction = "📈 LONG (BUY)"
elif confidence_down >= TRADE_THRESHOLD:
    prediction = 0
    confidence = confidence_down
    direction = "📉 SHORT (SELL)"
else:
    prediction = -1
    confidence = max(confidence_up, confidence_down)
    direction = "🛑 NO TRADE (CONFIDENCE TOO LOW)"

# ==========================================
# 7. TELEGRAM INTEGRATION & IST TIMEZONE
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime('%Y-%m-%d %I:%M %p IST')

message = f"🤖 *QUANT ALGO UPDATE* 🤖\n🕒 {current_time_ist}\n\nAsset: BTC/USDT (Binance)\nCurrent Model Price: ${last_price:.2f}\n"

if prediction == -1:
    message += f"\n🛑 *NO TRADE*\nConfidence: {confidence*100:.1f}% (Need {TRADE_THRESHOLD*100:.1f}%)\nMarket is too choppy. Staying in cash."
else:
    prob_easy = confidence * 0.85
    message += f"\n{direction}\nConfidence: *{confidence * 100:.1f}%*\nWin Prob (1% Target): {prob_easy*100:.1f}%\n\n🎯 *SETUP ZONES*\n"
    if prediction == 1: 
        message += f"Entry/Pullback: ${last_price - (hourly_atr * 0.5):.2f}\nTake Profit (1%): ${last_price * 1.01:.2f}\nStop Loss (-0.5%): ${last_price * 0.995:.2f}\n"
    else: 
        message += f"Entry/Pullback: ${last_price + (hourly_atr * 0.5):.2f}\nTake Profit (1%): ${last_price * 0.99:.2f}\nStop Loss (+0.5%): ${last_price * 1.005:.2f}\n"

print(message)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("\n📲 Successfully sent to Telegram!")
        else:
            print(f"\n⚠️ Telegram Error: {response.text}")
    except Exception as e:
        print(f"\n⚠️ Failed to send Telegram message: {e}")

# Sending message no matter what so you can verify it works
send_telegram(message)
