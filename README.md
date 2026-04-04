# Serverless Quantitative Crypto Trading Bot

An end-to-end, fully autonomous Machine Learning trading algorithm built with Python and XGBoost. The bot runs on a serverless cloud infrastructure, ingests live market data, predicts directional price movement, executes paper trades, and logs performance via Telegram and Git.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-success)
![Binance](https://img.shields.io/badge/Binance-Testnet-yellow)

## Project Overview
This project serves as a proof-of-concept for deploying institutional-grade algorithmic trading infrastructure using zero-cost cloud tools. 

Instead of relying on basic moving average crossovers, this algorithm uses an **XGBoost Classifier** to evaluate intraday volatility, global macro correlations, and algorithmic Fibonacci retracements. It requires a strict >56% confidence threshold to issue a trade, effectively sitting in cash during choppy market regimes.

## System Architecture

### 1. The Data Engine (RAM-Only Memory Stitching)
To prevent file corruption on ephemeral cloud servers, the bot uses a read-only historical CSV as an anchor. Every hour, it fetches the latest live candles via `ccxt.binanceus` (bypassing geo-blocks), chronologically slices the data using a custom timezone handling method, and stitches it perfectly in memory. 

### 2. Feature Engineering
The model is fed a highly engineered dataset containing:
* **Volatility & Momentum:** ATR (Average True Range), Rate of Change (4h & 24h), and Distance to EMAs.
* **Algorithmic Memory:** Programmatic calculation of the recent 72-hour Swing High/Low and the distance to the **61.8% Fibonacci Golden Pocket**.
* **Global Macro Context:** Live S&P 500 (`SPY`) and US Dollar Index (`DXY`) hourly returns fetched via `yfinance` to gauge broader market liquidity.
* **Cyclical Time Encoding:** Sine/Cosine transformations of the current hour to help the model learn daily liquidity cycles.

### 3. Machine Learning (XGBoost)
* The model evaluates the engineered features and predicts if the price will move >0.1% over the next 6 hours.
* Utilizing pre-calculated optimal hyperparameters (Grid Search), the inference time is reduced to <3 seconds.
* Generates continuous probabilistic outputs (e.g., 61% Long, 39% Short).

### 4. Cloud DevOps & Execution (GitHub Actions)
* **Automation:** A YAML workflow triggers the Python script at the top of every hour.
* **Auto-Execution:** Uses `ccxt.binance` (Sandbox Mode) to place Limit entries and OCO (One-Cancels-the-Other) Take-Profit/Stop-Loss orders on the Binance Spot Testnet.
* **Visual Logging:** Uses `matplotlib` to plot recent price action overlaid with the bot's decisions (Green/Red arrows).
* **Alerts:** Pings a private Telegram bot with the chart, the predicted confidence, and exact limit order zones.
* **State Persistence:** Automatically executes a `git commit` to permanently log the trade details into `trade_log.csv` before the server destroys itself.

## Tech Stack
* **Core:** Python, Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-Learn
* **Market Data & Execution:** CCXT, yfinance
* **Visualization:** Matplotlib
* **Infrastructure:** GitHub Actions (Ubuntu-latest)

## Disclaimer
This code is for educational and portfolio demonstration purposes only. It is currently configured to run strictly on the Binance Testnet using simulated funds. Do not use this algorithm with real capital without extensive forward-testing and risk management modifications.
