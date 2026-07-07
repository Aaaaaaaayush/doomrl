import collections
import cv2
import numpy as np

def preprocess_frame(frame, width=84, height=84):
    """
    Resizes the raw frame to 84x84 and normalizes pixel values to [0, 1].
    """
    if frame is None:
        return np.zeros((height, width), dtype=np.float32)
        
    # Resize using bilinear/area interpolation (OpenCV expects width, height)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel integers [0, 255] to floats [0.0, 1.0]
    normalized = resized.astype(np.float32) / 255.0
    return normalized

class FrameStackWrapper:
    """
    Stacks the last N frames to capture temporal information (movement).
    Returns observations of shape (stack_size, height, width).
    """
    def __init__(self, env, stack_size=4, img_size=(84, 84)):
        self.env = env
        self.stack_size = stack_size
        self.img_size = img_size
        
        # A queue with a max limit automatically discards the oldest element
        # when a new one is added.
        self.frames = collections.deque(maxlen=stack_size)
        
        # Expose action properties of the inner environment directly
        self.action_space_size = env.action_space_size

    def reset(self):
        """Resets the environment and fills the stack with the initial frame."""
        raw_obs = self.env.reset()
        processed = preprocess_frame(raw_obs, self.img_size[0], self.img_size[1])
        
        # Fill the queue with duplicate copies of the first frame to initialize
        for _ in range(self.stack_size):
            self.frames.append(processed)
            
        return np.stack(self.frames, axis=0)

    def step(self, action_idx, frame_skip=4):
        """Steps the environment and pushes the new frame into the stack."""
        raw_obs, reward, done, info = self.env.step(action_idx, frame_skip)
        
        if not done:
            processed = preprocess_frame(raw_obs, self.img_size[0], self.img_size[1])
            self.frames.append(processed)
        else:
            # If the episode ends, we fill the slot with a zero frame
            processed = np.zeros(self.img_size, dtype=np.float32)
            self.frames.append(processed)
            
        stacked_obs = np.stack(self.frames, axis=0)
        return stacked_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self.env.close()
