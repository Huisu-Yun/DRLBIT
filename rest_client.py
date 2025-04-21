import jwt
import uuid
import requests
from urllib.parse import urlencode
import logging
from datetime import datetime
from typing import Dict, List, Optional

from config import UPBIT_API_URL, UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY

logger = logging.getLogger(__name__)

class UpbitRestClient:
    def __init__(self):
        self.base_url = UPBIT_API_URL
        self.access_key = UPBIT_ACCESS_KEY
        self.secret_key = UPBIT_SECRET_KEY

    def _get_auth_headers(self, query: Optional[Dict] = None) -> Dict:
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4())
        }
        
        if query:
            payload['query'] = urlencode(query)

        jwt_token = jwt.encode(payload, self.secret_key)
        return {'Authorization': f'Bearer {jwt_token}'}

    def get_ticker(self, markets: List[str]) -> List[Dict]:
        """Get current price ticker for specified markets."""
        try:
            url = f"{self.base_url}/ticker"
            params = {"markets": ",".join(markets)}
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching ticker data: {e}")
            return []

    def get_orderbook(self, markets: List[str]) -> List[Dict]:
        """Get orderbook for specified markets."""
        try:
            url = f"{self.base_url}/orderbook"
            params = {"markets": ",".join(markets)}
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook data: {e}")
            return []

    def get_market_trades(self, market: str, count: int = 100) -> List[Dict]:
        """Get recent trades for a specific market."""
        try:
            url = f"{self.base_url}/trades/ticks"
            params = {"market": market, "count": count}
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching trade data: {e}")
            return []

    def get_candles_minutes(self, market: str, unit: int = 1, count: int = 200) -> List[Dict]:
        """Get minute candles for a specific market."""
        try:
            url = f"{self.base_url}/candles/minutes/{unit}"
            params = {"market": market, "count": count}
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching candle data: {e}")
            return []

    def format_for_influx(self, data: Dict, measurement: str, market: str) -> Dict:
        """Format data for InfluxDB storage."""
        timestamp = datetime.utcnow()
        return {
            "measurement": measurement,
            "tags": {
                "market": market
            },
            "time": timestamp,
            "fields": data
        } 