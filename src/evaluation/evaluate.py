import argparse
import os
import numpy as np
import torch
import imageio
from tqdm import tqdm

from src.environment.doom_env import VizDoomEnv
from src.environment.frame_processor import FrameStackWrapper, preprocess_frame
from src.agent.dqn_agent import DQNAgent

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def record_gameplay(config_path, checkpoint_path, output_video_path, num_episodes=3, fps=30, epsilon=0.0):
    config = load_config(config_path)
    scenario_name = config['scenario']['name']
    
    # Locate WAD file
    import vizdoom as zd
    scenarios_dir = zd.scenarios_path
    wad_file = config['scenario']['wad_file']
    wad_path = os.path.join(scenarios_dir, wad_file)
    cfg_file = f"{scenario_name}.cfg"
    cfg_path = os.path.join("configs", cfg_file)
    
    # 1. Initialize environment (window_visible is False since we record screen buffer)
    reward_shaping = config.get('reward_shaping', None)
    raw_env = VizDoomEnv(
        config_path=cfg_path, 
        wad_path=wad_path, 
        window_visible=False,
        reward_shaping=reward_shaping
    )
    env = FrameStackWrapper(raw_env, stack_size=config['environment']['stack_frames'])
    
    state_dim = (config['environment']['stack_frames'], 84, 84)
    num_actions = env.action_space_size
    
    # 2. Initialize Agent and load weights if provided
    agent = DQNAgent(state_dim, num_actions, config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        agent.load(checkpoint_path)
    else:
        print("No valid checkpoint found. Running with random policy initialization.")
        
    # Force evaluation epsilon
    agent.epsilon = epsilon
    agent.epsilon_end = epsilon
    
    # Ensure target output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    print(f"Recording {num_episodes} episodes to: {output_video_path} (Epsilon: {epsilon})")
    
    # We will use imageio to write frames to a video file
    # imageio-ffmpeg is used for MP4 writing
    writer = imageio.get_writer(output_video_path, fps=fps, format='FFMPEG', mode='I')
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < config['training']['max_steps_per_episode']:
            # Predict action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action, frame_skip=config['environment']['frame_skip'])
            
            # Retrieve the raw, full-resolution screen buffer directly from the engine
            raw_state = raw_env.game.get_state()
            if raw_state is not None and raw_state.screen_buffer is not None:
                # Transpose from (C, H, W) to (H, W, C) for imageio, or duplicate channels for grayscale
                frame = raw_state.screen_buffer
                if len(frame.shape) == 2: # Grayscale frame
                    # Duplicate to 3 color channels so it displays properly in common video players
                    frame = np.stack((frame,)*3, axis=-1)
                elif len(frame.shape) == 3 and frame.shape[0] == 3: # Channel-first RGB
                    frame = np.transpose(frame, (1, 2, 0))
                
                # Write frame to video
                writer.append_data(frame)
                
            state = next_state
            episode_reward += reward
            steps += 1
            
        print(f"  Episode {episode+1} Finished | Steps: {steps} | Reward: {episode_reward:.1f}")
        
    writer.close()
    env.close()
    print(f"✓ Video saved successfully to {output_video_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Agent and Record Gameplay Videos")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model weights checkpoint .pth file")
    parser.add_argument("--output", type=str, required=True, help="Path to output video file (.mp4)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon exploration probability during evaluation")
    args = parser.parse_args()
    
    record_gameplay(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_video_path=args.output,
        num_episodes=args.episodes,
        epsilon=args.epsilon
    )
