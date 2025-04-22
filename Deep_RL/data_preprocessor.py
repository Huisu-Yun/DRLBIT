import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import torch
from collections import deque

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, client: InfluxDBClient, bucket: str, org: str, window_size: int = 60, n_regimes: int = 3):
        self.client = client
        self.bucket = bucket
        self.org = org        
        self.query_api = client.query_api()
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.hmm = GaussianHMM(n_components=n_regimes)
        self.feature_buffer = deque(maxlen=window_size)
        
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

        logger.info(f"Features are created :\r\n {df.head()}")
        
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
        
        # Ensure all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return np.array([]), np.array([])
        
        # Scale features
        X = df[features].values
        if len(X) > 0:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            logger.info(f"Feature shape after scaling: {X.shape}")
        else:
            logger.error("No data available after feature creation")
            return np.array([]), np.array([])
        
        # Create labels (1 for positive returns, 0 for negative/zero returns)
        y = np.where(df['returns'].shift(-1) > 0, 1, 0)
        
        # Remove the last row since we don't have the next return
        X = X[:-1]
        y = y[:-1]
        
        logger.info(f"Final shapes - X: {X.shape}, y: {y.shape}")
        return X, y

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset."""
        df = data.copy()
        
        # RSI
        delta = df['trade_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # EMA
        df['ema_30'] = df['trade_price'].ewm(span=30, adjust=False).mean()
        
        # MACD
        exp1 = df['trade_price'].ewm(span=12, adjust=False).mean()
        exp2 = df['trade_price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # VWAP
        df['vwap'] = (df['trade_price'] * df['trade_volume']).cumsum() / df['trade_volume'].cumsum()
        
        # Bollinger Bands
        df['bb_middle'] = df['trade_price'].rolling(window=20).mean()
        bb_std = df['trade_price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high_price'] - df['low_price']
        high_close = np.abs(df['high_price'] - df['trade_price'].shift())
        low_close = np.abs(df['low_price'] - df['trade_price'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df
        
    def calculate_orderbook_features(self, orderbook: Dict) -> np.ndarray:
        """Calculate order book features."""
        bid_prices = np.array(orderbook['bid_prices'][:5])
        ask_prices = np.array(orderbook['ask_prices'][:5])
        bid_volumes = np.array(orderbook['bid_volumes'][:5])
        ask_volumes = np.array(orderbook['ask_volumes'][:5])
        
        # Price features
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        spread = ask_prices[0] - bid_prices[0]
        
        # Volume features
        volume_imbalance = (bid_volumes.sum() - ask_volumes.sum()) / (bid_volumes.sum() + ask_volumes.sum())
        
        # Order book pressure
        bid_pressure = (bid_volumes * (1 / np.abs(bid_prices - mid_price))).sum()
        ask_pressure = (ask_volumes * (1 / np.abs(ask_prices - mid_price))).sum()
        
        return np.concatenate([
            bid_prices, ask_prices, bid_volumes, ask_volumes,
            [mid_price, spread, volume_imbalance, bid_pressure, ask_pressure]
        ])
        
    def create_state_tensor(self, 
                          market_data: pd.DataFrame, 
                          orderbook: Dict,
                          position: float,
                          balance: float) -> torch.Tensor:
        """Create state tensor for the DRL model."""
        # Calculate the same features as in prepare_training_data
        df = market_data.copy()
        
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
        
        # Select the same features as in prepare_training_data
        features = [
            'returns', 'sma_5', 'sma_20', 'rsi', 
            'bb_middle', 'bb_upper', 'bb_lower',
            'volume_ratio'
        ]
        
        # Get the latest values for each feature
        try:
            state = df[features].iloc[-1].values
            # Handle NaN values
            state = np.nan_to_num(state, nan=0.0)
            
            # Scale the features using the same scaler
            state = self.scaler.transform(state.reshape(1, -1))[0]
            
            # Add to feature buffer
            self.feature_buffer.append(state)
            
            # Create sequence of states
            if len(self.feature_buffer) < self.window_size:
                # Pad with zeros if buffer not full
                pad_size = self.window_size - len(self.feature_buffer)
                padded_states = np.vstack([np.zeros_like(state)] * pad_size + list(self.feature_buffer))
            else:
                padded_states = np.vstack(list(self.feature_buffer))
            
            return torch.FloatTensor(padded_states).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error creating state tensor: {str(e)}")
            # Return zero tensor with correct shape in case of error
            return torch.zeros((1, self.window_size, len(features)))
        
    def detect_regime_change(self, data: pd.DataFrame) -> np.ndarray:
        """Detect market regime changes using HMM."""
        # Features for regime detection
        features = np.column_stack([
            data['trade_price'].pct_change(),
            data['trade_volume'].pct_change(),
            data['vwap'].pct_change()
        ])
        features = np.nan_to_num(features)
        
        # Fit HMM and predict regimes
        self.hmm.fit(features)
        regimes = self.hmm.predict(features)
        
        return regimes
        
    def update_online(self, new_data: pd.DataFrame):
        """Update scalers and models with new data."""
        self.scaler.partial_fit(new_data)
        # Update HMM if needed
        if len(new_data) >= self.window_size:
            self.detect_regime_change(new_data) 