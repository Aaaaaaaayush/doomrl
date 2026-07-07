import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.agent.dqn import DQN
from src.agent.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    DQNAgent manages:
    1. Action selection (epsilon-greedy).
    2. Model weight updates (sampling, loss computation, backpropagation).
    3. Target network synchronization.
    """
    def __init__(self, state_dim, num_actions, config):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate networks
        self.q_network = DQN(state_dim, num_actions).to(self.device)
        self.target_network = DQN(state_dim, num_actions).to(self.device)
        
        # Initial sync
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['training']['learning_rate'])
        self.loss_fn = nn.SmoothL1Loss()
        
        self.replay_buffer = ReplayBuffer(config['training']['replay_buffer_size'])
        
        self.epsilon = config['training']['epsilon_start']
        self.epsilon_end = config['training']['epsilon_end']
        self.epsilon_decay_steps = config['training']['epsilon_decay_steps']
        self.epsilon_decay_rate = (self.epsilon - self.epsilon_end) / self.epsilon_decay_steps
        
        self.steps_done = 0

    def select_action(self, state):
        """Chooses action index using epsilon-greedy exploration."""
        self.steps_done += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_rate
            
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_t)
                return int(q_values.argmax(dim=1).item())

    def update_model(self):
        """Samples experiences and updates network parameters."""
        if len(self.replay_buffer) < self.config['training']['learning_starts']:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config['training']['batch_size'])
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        
        # Reward clipping for stability
        rewards_t = torch.clamp(rewards_t, min=-1.0, max=1.0)
        
        # Predict Q-values
        current_q_values = self.q_network(states_t).gather(1, actions_t).squeeze(1)
        
        # Compute target Q-values (Bellman equation)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1)[0]
            target_q_values = rewards_t + self.config['training']['gamma'] * next_q_values * (~dones_t)
            
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Sync target network periodically
        if self.steps_done % self.config['training']['target_network_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return float(loss.item())

    def save(self, filepath):
        """Saves current network parameters and states."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Loads network parameters and states from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
