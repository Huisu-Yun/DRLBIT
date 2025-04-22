"""Configuration for the Deep RL trading agent"""

# Import InfluxDB configuration from parent directory's config
import os
from pathlib import Path

# Get the parent directory's config file path
parent_config_path = Path(__file__).parent.parent / 'config.py'

# Read and execute the parent config file to get the variables
with open(parent_config_path, 'r') as f:
    exec(f.read())

DQN_CONFIG = {
    # Network parameters
    'hidden_size': 128,
    'learning_rate': 1e-4,
    
    # Training parameters
    'batch_size': 32,
    'memory_size': 100000,
    'gamma': 0.99,  # Discount factor
    
    # Exploration parameters
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    
    # Target network update
    'target_update': 1000,  # Update target network every N steps
    'tau': 0.005,  # Soft update parameter
    
    # Training duration
    'episodes': 1000,
    'max_steps': 10000,
    
    # Checkpointing
    'save_frequency': 100,  # Save model every N episodes
    'model_dir': 'models'
} 