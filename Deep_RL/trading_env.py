import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, data: np.ndarray, initial_balance: float = 10000.0):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # State space: 8 features (returns, sma_5, sma_20, rsi, bb_middle, bb_upper, bb_lower, volume_ratio)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8,),
            dtype=np.float32
        )
        
        # Calculate prices from returns (first feature)
        self.prices = np.exp(np.cumsum(self.data[:, 0]))  # Convert returns to prices
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_values = [self.initial_balance]
        return self.data[self.current_step].astype(np.float32)
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.current_step >= len(self.data) - 1:
            return self.data[self.current_step].astype(np.float32), 0, True, self._get_info()
            
        current_price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1]
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:  # Only buy if we don't have a position
                self.position = self.balance / current_price
                self.balance = 0
        elif action == 2:  # Sell
            if self.position > 0:  # Only sell if we have a position
                self.balance = self.position * current_price
                self.position = 0
                
        # Calculate portfolio value
        portfolio_value = self.balance + (self.position * next_price)
        self.portfolio_values.append(portfolio_value)
        
        # Calculate reward (log returns)
        reward = np.log(portfolio_value / self.portfolio_values[-2]) if len(self.portfolio_values) > 1 else 0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self.data[self.current_step].astype(np.float32), reward, done, self._get_info()
        
    def _get_info(self) -> Dict:
        return {
            'portfolio_value': self.portfolio_values[-1],
            'balance': self.balance,
            'position': self.position,
            'trade_count': sum(1 for i in range(1, len(self.portfolio_values)) 
                             if self.portfolio_values[i] != self.portfolio_values[i-1])
        }
        
    def render(self, mode='human'):
        pass 