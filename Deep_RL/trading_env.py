import gym
import numpy as np
from gym import spaces
from typing import Dict, Tuple, Any

class TradingEnv(gym.Env):
    def __init__(self, data: np.ndarray, initial_balance: float = 10000.0):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(data.shape[1],), 
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.balance = self.initial_balance
        self.holdings = 0
        self.current_step = 0
        self.trades = []
        self.total_reward = 0
        
        return self._get_observation()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        current_price = self.data[self.current_step, 0]  # Assuming price is the first feature
        reward = 0
        done = False
        
        if action == 1:  # Buy
            if self.balance > 0:
                self.holdings = self.balance / current_price
                self.balance = 0
                self.trades.append(('buy', current_price))
                
        elif action == 2:  # Sell
            if self.holdings > 0:
                self.balance = self.holdings * current_price
                self.holdings = 0
                self.trades.append(('sell', current_price))
                
        # Calculate reward based on portfolio value change
        portfolio_value = self.balance + (self.holdings * current_price)
        reward = portfolio_value - self.initial_balance
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            done = True
            
        return self._get_observation(), reward, done, {}
        
    def _get_observation(self) -> np.ndarray:
        """Get the current observation"""
        return self.data[self.current_step]
        
    def render(self, mode='human'):
        """Render the environment to the screen"""
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance:.2f}')
            print(f'Holdings: {self.holdings:.2f}')
            print(f'Total Reward: {self.total_reward:.2f}') 