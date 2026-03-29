import sqlite3
import time
import random
from datetime import datetime

# 1. Connect to the database (it creates 'live_data.db' automatically if it doesn't exist)
conn = sqlite3.connect('live_data.db')
cursor = conn.cursor()

# 2. Create a table to store our live data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_prices (
        timestamp TEXT, 
        price REAL
    )
''')
conn.commit()

print("Starting live data fetcher. Press Ctrl+C to stop.")

# 3. The Infinite Real-Time Loop
while True:
    try:
        # Get current time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Simulate fetching a live stock price (e.g., hovering around ₹2500)
        # In the future, replace this with your yfinance or broker API call
        live_price = random.uniform(2490.0, 2510.0) 
        
        # Insert the new data into the database
        cursor.execute('INSERT INTO stock_prices VALUES (?, ?)', (current_time, live_price))
        conn.commit()
        
        print(f"[{current_time}] Saved new price: ₹{live_price:.2f}")
        
        # Wait for 3 seconds before fetching the next data point
        time.sleep(3) 
        
    except KeyboardInterrupt:
        print("\nFetcher stopped by user.")
        break

conn.close()
