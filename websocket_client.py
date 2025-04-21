import json
import asyncio
import websockets
import logging
from datetime import datetime, timezone
from typing import List, Dict, Callable, Any

from config import UPBIT_WS_URL

logger = logging.getLogger(__name__)

class UpbitWebSocketClient:
    def __init__(self):
        self.ws_url = UPBIT_WS_URL
        self.websocket = None
        self.callbacks = []

    def add_callback(self, callback: Callable[[Dict], Any]):
        """Add a callback function to process received data."""
        self.callbacks.append(callback)

    async def connect(self, markets: List[str], channels: List[str]):
        """
        Connect to Upbit WebSocket and subscribe to specified markets and channels.
        Available channels: ticker, trade, orderbook
        """
        try:
            self.websocket = await websockets.connect(self.ws_url)
            subscribe_fmt = [
                {"ticket": "UNIQUE_TICKET"},
                {
                    "type": "trade",
                    "codes": markets,
                    "isOnlyRealtime": True
                },
                {"format": "SIMPLE"}
            ]
            
            await self.websocket.send(json.dumps(subscribe_fmt))
            logger.info(f"Connected to Upbit WebSocket")
            return True
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            return False

    def format_for_influx(self, data: Dict) -> Dict:
        """Format WebSocket data for InfluxDB storage."""
        timestamp = datetime.now(timezone.utc)
        # logger.info(timestamp)
        
        # Map the fields from WebSocket response
        fields = {
            "type": data.get("ty", ""),                    # type (trade)
            "market": data.get("cd", "unknown"),           # market code
            "timestamp_ms": data.get("tms", 0),            # timestamp in milliseconds
            "trade_date": data.get("td", ""),             # trade date
            "trade_time": data.get("ttm", ""),            # trade time
            "trade_timestamp_ms": data.get("ttms", 0),    # trade timestamp in milliseconds
            "trade_price": float(data.get("tp", 0)),      # trade price
            "trade_volume": float(data.get("tv", 0)),     # trade volume
            "ask_bid": data.get("ab", ""),                # ask/bid
            "prev_closing_price": float(data.get("pcp", 0)), # previous closing price
            #"change": data.get("c", ""),                  # change (RISE/FALL)
            "change_price": float(data.get("cp", 0)),     # change price
            "sequential_id": data.get("sid", ""),         # sequential ID
            "best_ask_price": float(data.get("bap", 0)),  # best ask price
            "best_ask_size": float(data.get("bas", 0)),   # best ask size
            "best_bid_price": float(data.get("bbp", 0)),  # best bid price
            "best_bid_size": float(data.get("bbs", 0)),   # best bid size
            # "stream_type": data.get("st", "")             # stream type
        }

        return {
            "measurement": "trade",
            "tags": {
                "market": data.get("cd", "unknown"),
                "type": data.get("ty", "unknown")
            },
            "time": timestamp,
            "fields": fields
        }

    async def process_messages(self):
        """Process incoming WebSocket messages."""
        try:
            while True:
                if self.websocket:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    # logger.info(data)
                    
                    # Process data through all registered callbacks
                    for callback in self.callbacks:
                        await callback(data)
                else:
                    logger.error("WebSocket connection not established")
                    break
        except websockets.exceptions.ConnectionClosed:
            logger.error("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    async def close(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed") 