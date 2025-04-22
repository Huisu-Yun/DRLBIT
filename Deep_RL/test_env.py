import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from influxdb_client import InfluxDBClient
from config import INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET
from Deep_RL.data_preprocessor import DataPreprocessor
from Deep_RL.trading_env import TradingEnv
from Deep_RL.dqn_model import DQNAgent, TradingDRL
from Deep_RL.config import DQN_CONFIG

def test_data_pipeline():
    """Test the data preprocessing pipeline"""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize InfluxDB client
        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(client, INFLUXDB_BUCKET, INFLUXDB_ORG)
        
        # Get some test data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=5)  # Use 5 days of data for testing
        market = "KRW-BTC"
        
        logger.info("Testing data preprocessing...")
        X, y = preprocessor.prepare_training_data(market, start_time, end_time)
        logger.info(f"Data shape - X: {X.shape}, y: {y.shape}")
        
        if len(X) == 0:
            logger.error("No data retrieved from InfluxDB")
            return False
            
        # Test environment initialization
        logger.info("Testing environment initialization...")
        env = TradingEnv(data=X, initial_balance=10000.0)
        
        # Test reset
        logger.info("Testing environment reset...")
        initial_state = env.reset()
        logger.info(f"Initial state shape: {initial_state.shape}")
        
        # Test step
        logger.info("Testing environment step...")
        next_state, reward, done, info = env.step(1)  # Try buying
        logger.info(f"Next state shape: {next_state.shape}")
        logger.info(f"Reward: {reward}")
        logger.info(f"Info: {info}")
        
        # Test DQN model
        logger.info("Testing DQN model...")
        state_dim = X.shape[1]  # Number of features
        action_dim = env.action_space.n
        agent = DQNAgent(state_dim, action_dim, DQN_CONFIG)
        
        # Move model to appropriate device
        agent.policy_net = agent.policy_net.to(device)
        agent.target_net = agent.target_net.to(device)
        
        # Test forward pass
        logger.info("Testing model forward pass...")
        state_tensor = torch.FloatTensor(initial_state).unsqueeze(0).to(device)  # Add batch dimension and move to device
        q_values = agent.policy_net(state_tensor)
        logger.info(f"Q-values shape: {q_values.shape}")
        
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        client.close()

if __name__ == "__main__":
    success = test_data_pipeline()
    if success:
        logger.info("All components are working correctly!")
    else:
        logger.error("Some tests failed. Please check the logs above.") 