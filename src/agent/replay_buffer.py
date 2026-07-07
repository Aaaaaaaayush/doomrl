import collections
import random
import numpy as np

class ReplayBuffer:
    """
    A circular memory buffer that stores agent experiences (state, action, reward, next_state, done)
    and allows random sampling for training batches.
    """
    def __init__(self, capacity):
        # We use a deque with a fixed maximum size to act as a circular buffer.
        # When capacity is reached, adding a new item automatically discards the oldest.
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Saves a single experience tuple into the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly samples a batch of experiences from the buffer."""
        # Draw random experiences from our memory bank
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch of tuples into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)
