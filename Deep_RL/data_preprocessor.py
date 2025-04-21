import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, client: InfluxDBClient, bucket: str, org: str):
        self.client = client
        self.bucket = bucket
        self.org = org
        self.query_api = client.query_api()
        
    def get_historical_data(self, market: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch historical data from InfluxDB and convert to pandas DataFrame"""
        # Format dates in RFC3339 format
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r["_measurement"] == "trade")
          |> filter(fn: (r) => r["market"] == "{market}")
          |> drop(columns: ["_start", "_stop", "_measurement", "market"])
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        try:
            logger.info(f"Querying InfluxDB: {query}")
            result = self.query_api.query_data_frame(query, org=self.org)
            if result.empty:
                logger.warning("No data returned from query")
                return pd.DataFrame()
                
            # Convert to proper datetime index
            result['_time'] = pd.to_datetime(result['_time'])
            result.set_index('_time', inplace=True)
            result.sort_index(inplace=True)
            
            # Add market column back
            result['market'] = market
            
            logger.info(f"Retrieved {len(result)} records")
            return result
        except Exception as e:
            logger.error(f"Error querying InfluxDB: {str(e)}")
            return pd.DataFrame()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features for the RL model"""
        if df.empty:
            return pd.DataFrame()
            
        # Calculate returns
        df['returns'] = df['trade_price'].pct_change()
        
        # Calculate moving averages
        df['sma_5'] = df['trade_price'].rolling(window=5).mean()
        df['sma_20'] = df['trade_price'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = df['trade_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['trade_price'].rolling(window=20).mean()
        df['bb_std'] = df['trade_price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calculate volume features
        df['volume_sma'] = df['trade_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['trade_volume'] / df['volume_sma']
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
        
    def prepare_training_data(self, market: str, start_time: datetime, end_time: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training the RL model"""
        df = self.get_historical_data(market, start_time, end_time)
        logger.info(f"df: {len(df)}")
        if df.empty:
            return np.array([]), np.array([])
            
        df = self.create_features(df)
        
        # Select features for the model
        features = [
            'returns', 'sma_5', 'sma_20', 'rsi', 
            'bb_middle', 'bb_upper', 'bb_lower',
            'volume_ratio'
        ]
        
        X = df[features].values
        y = np.where(df['returns'].shift(-1) > 0, 1, 0)  # 1 for buy, 0 for sell
        
        return X, y 