import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)

class AttentionModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, hidden_size)
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class TradingDRL(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_actions: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Attention mechanism
        self.attention = AttentionModule(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Dueling DQN architecture
        self.value_stream = nn.Linear(32, 1)
        self.advantage_stream = nn.Linear(32, num_actions)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Ensure input is 3D: (batch_size, seq_len, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # LSTM layers with residual connections
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.layer_norm1(lstm1_out)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.layer_norm2(lstm2_out)
        lstm2_out = self.dropout(lstm2_out)
        
        # Attention mechanism
        lstm2_out = lstm2_out.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        attn_out = self.attention(lstm2_out)
        attn_out = attn_out[-1]  # Take the last timestep
        
        # Fully connected layers
        x = F.relu(self.fc1(attn_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Dueling DQN
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = value + (advantages - advantages.mean())
        
        return qvals

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> tuple:
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
        
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = TradingDRL(state_size, config['hidden_size'], action_size).to(self.device)
        self.target_net = deepcopy(self.policy_net)
        
        # Training components
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), 
                                         lr=config['learning_rate'],
                                         weight_decay=0.01)
        self.memory = deque(maxlen=config['memory_size'])
        self.writer = SummaryWriter()
        
        # Training parameters
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.gamma = config['gamma']
        self.tau = config['tau']  # For soft updates
        self.update_frequency = config['target_update']
        self.batch_size = config['batch_size']
        self.step_counter = 0
        
        self.logger = logging.getLogger(__name__)
        
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Logging
        self.writer.add_scalar('Loss/train', loss.item(), self.step_counter)
        self.writer.add_scalar('Q_values/max', current_q_values.max().item(), self.step_counter)
        self.writer.add_scalar('Q_values/min', current_q_values.min().item(), self.step_counter)
        
        # Update target network
        if self.step_counter % self.update_frequency == 0:
            self.soft_update()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_counter += 1
        
    def soft_update(self):
        """Soft update of target network parameters."""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
            
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_counter': self.step_counter
        }, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_counter = checkpoint['step_counter']
        self.logger.info(f"Model loaded from {path}") 