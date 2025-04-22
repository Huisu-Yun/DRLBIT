import os
from dotenv import load_dotenv

load_dotenv()

# Upbit API Configuration
UPBIT_API_URL = "https://api.upbit.com/v1"
UPBIT_WS_URL = "wss://api.upbit.com/websocket/v1"
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

# URL = "http://localhost:8086"  # Change if your InfluxDB runs on a different address
# TOKEN = "8-9q3iJM0QCP9UsUKcq-AheN1S4OMzED3st5J5TS3d0mzzt4N21xVPQ2fVtrJV-greW0nCHRqOsoGND38lF8pw=="
# ORG = "Macallan"
# BUCKET = "UPBIT"

# InfluxDB Configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "upbit_org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "upbit_bucket")

# Market Configuration
DEFAULT_MARKETS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-USDT", "KRW-USDC", "KRW-SOL", "KRW-USDC", "KRW-DOGE", "KRW-TRX", "KRW-ADA", "KRW-SUI", "KRW-LINK", "KRW-AVAX"]
