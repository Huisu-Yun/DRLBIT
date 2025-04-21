import os
import torch
import numpy as np
from datetime import datetime, timedelta
from data_preprocessor import DataPreprocessor
from trading_env import TradingEnv
from dqn_model import DQNAgent
import logging
import sys
from pathlib import Path
import time
from datetime import datetime, timezone

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from influx_client import InfluxDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(
    market: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float = 10000.0,
    num_episodes: int = 1000,
    batch_size: int = 32,
    target_update: int = 10,
    save_interval: int = 100
):
    # Initialize InfluxDB client
    try:
        influx_client = InfluxDBManager()
        logger.info("Successfully connected to InfluxDB")
    except Exception as e:
        logger.error(f"Failed to connect to InfluxDB: {str(e)}")
        return
        
    try:
        # Initialize data preprocessor
        preprocessor = DataPreprocessor(
            client=influx_client.client,
            bucket=influx_client.bucket,
            org=influx_client.org
        )
        
        # Prepare training data
        logger.info(f"Preparing training data for market: {market} {start_date} {end_date}")
        X, y = preprocessor.prepare_training_data(market, start_date, end_date)





        
        if len(X) == 0:
            logger.error("No data available for training")
            return
            
        return
        # Initialize environment and agent
        env = TradingEnv(X, initial_balance)
        state_size = X.shape[1]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Training loop
        best_reward = float('-inf')
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.memory.push(state, action, reward, next_state, done)
                agent.train(batch_size)
                
                state = next_state
                total_reward += reward
                
            if episode % target_update == 0:
                agent.update_target_network()
                
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_model(f'models/best_model_{market}.pth')
                
            # Save model at intervals
            if episode % save_interval == 0:
                agent.save_model(f'models/checkpoint_{market}_episode_{episode}.pth')
                
            # Log training progress
            episode_time = time.time() - episode_start_time
            logger.info(
                f'Episode: {episode}, '
                f'Total Reward: {total_reward:.2f}, '
                f'Epsilon: {agent.epsilon:.2f}, '
                f'Time: {episode_time:.2f}s'
            )
            
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
    finally:
        influx_client.close()
        logger.info("Training completed and InfluxDB connection closed")
    
if __name__ == '__main__':
    # Example usage
    market = 'KRW-XRP'
    # Use historical data from the past 30 days
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA is not available. Using CPU")
    
    train_model(
        market=market,
        start_date=start_date,
        end_date=end_date,
        num_episodes=1000,
        batch_size=32
    ) 