import os
import torch
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

from data_preprocessor import DataPreprocessor
from trading_env import TradingEnv
from dqn_model import DQNAgent
from config import INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET, DQN_CONFIG
from influxdb_client import InfluxDBClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_agent(market: str, preprocessor: DataPreprocessor):
    try:

        # Get training data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # Get 30 days of data
        X, _ = preprocessor.prepare_training_data(market, start_time, end_time)
        
        if len(X) == 0:
            logger.error("No training data available")
            return
            
        # Create environment
        env = TradingEnv(data=X, initial_balance=10000.0)
        
        # Initialize agent
        state_dim = env.observation_space.shape[1]  # Features per timestep
        action_dim = env.action_space.n  # Number of discrete actions
        agent = DQNAgent(state_dim, action_dim, DQN_CONFIG)
        
        # Training loop
        best_reward = float('-inf')
        
        for episode in range(DQN_CONFIG['episodes']):
            state = env.reset()
            episode_reward = 0
            
            for step in range(DQN_CONFIG['max_steps']):
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition and train
                agent.store_transition(state, action, reward, next_state, done)
                agent.update_model()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Log progress
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                       f"Portfolio Value = {info['portfolio_value']:.2f}, "
                       f"Trades = {info['trade_count']}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                model_path = model_dir / f"best_model_{market}.pth"
                agent.save_model(str(model_path))
            
            # Regular checkpointing
            if episode % DQN_CONFIG['save_frequency'] == 0:
                checkpoint_path = model_dir / f"checkpoint_{market}_episode_{episode}.pth"
                agent.save_model(str(checkpoint_path))
                
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA is not available. Using CPU")
    
    # Initialize InfluxDB client and preprocessor
    client = InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG
    )
    
    preprocessor = DataPreprocessor(client, INFLUXDB_BUCKET, INFLUXDB_ORG)
    
    # Train for specific market
    market = "KRW-BTC"  # Bitcoin market
    train_agent(market, preprocessor) 