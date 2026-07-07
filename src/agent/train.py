import argparse
import json
import os
import time
import yaml
import numpy as np
import mlflow

from src.environment.doom_env import VizDoomEnv
from src.environment.frame_processor import FrameStackWrapper
from src.agent.dqn_agent import DQNAgent

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def export_logs_to_json(run_id, scenario_name, metrics_dict, output_dir="logs/json_exports"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{scenario_name}_metrics.json")
    
    export_data = {
        "run_id": run_id,
        "scenario": scenario_name,
        "timestamp": time.time(),
        "metrics": metrics_dict
    }
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=4)
    print(f"[OK] Exported metrics for {scenario_name} to {filepath}")

def train(config_path):
    config = load_config(config_path)
    scenario_name = config['scenario']['name']
    
    # Locate WAD file
    import vizdoom as zd
    scenarios_dir = zd.scenarios_path
    wad_file = config['scenario']['wad_file']
    wad_path = os.path.join(scenarios_dir, wad_file)
    
    # 1. Initialize environment
    cfg_file = f"{scenario_name}.cfg"
    cfg_path = os.path.join("configs", cfg_file)
    
    reward_shaping = config.get('reward_shaping', None)
    raw_env = VizDoomEnv(
        config_path=cfg_path, 
        wad_path=wad_path, 
        window_visible=False,
        reward_shaping=reward_shaping
    )
    env = FrameStackWrapper(raw_env, stack_size=config['environment']['stack_frames'])
    
    # 2. Initialize Agent
    state_dim = (config['environment']['stack_frames'], 84, 84)
    num_actions = env.action_space_size
    agent = DQNAgent(state_dim, num_actions, config)
    
    # Prepare save directories
    save_dir = config['model']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. Setup MLflow Tracking (using SQLite to avoid file-store maintenance warnings)
    os.makedirs("logs", exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///logs/mlflow.db")
    mlflow.set_experiment(config['logging']['mlflow_experiment'])
    
    # Metrics dictionaries for local JSON export
    metrics_history = {
        "episodes": [],
        "episode_rewards": [],
        "mean_rewards_100": [],
        "kill_counts": [],
        "survival_times": [],
        "losses": [],
        "epsilons": [],
        "fps_list": []
    }
    
    recent_rewards = collections.deque(maxlen=100) if 'collections' in globals() else []
    if not isinstance(recent_rewards, list):
        import collections
        recent_rewards = collections.deque(maxlen=100)
    
    print(f"\n=======================================================")
    print(f" Starting training: {scenario_name.upper()} scenario")
    print(f"=======================================================")
    
    total_steps = 0
    best_mean_reward = -float('inf')
    
    with mlflow.start_run() as run:
        # Log hyperparameters to MLflow
        mlflow.log_params({
            "learning_rate": config['training']['learning_rate'],
            "gamma": config['training']['gamma'],
            "batch_size": config['training']['batch_size'],
            "replay_buffer_size": config['training']['replay_buffer_size'],
            "target_network_update_freq": config['training']['target_network_update_freq'],
            "epsilon_decay_steps": config['training']['epsilon_decay_steps']
        })
        
        for episode in range(1, config['training']['total_episodes'] + 1):
            start_time = time.time()
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            kill_count = 0
            
            # Keep track of initial health to detect kills in logs
            last_health = 100.0
            
            while episode_steps < config['training']['max_steps_per_episode']:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action, frame_skip=config['environment']['frame_skip'])
                
                # Check for kills (specific to Doom: base_reward is high for kills in Basic/Defend center, 
                # or we can check when health drops of enemies. For basic, the game ends when Kakodemon dies)
                # We can approximate kills: if reward is positive or episode ends with success.
                # In defend center/deadly corridor we can count kills via internal game variables if set.
                # For Basic, a reward > 90 implies a kill.
                if scenario_name == "basic" and reward >= 90.0:
                    kill_count += 1
                elif scenario_name == "defend_the_center" and reward >= 1.0:
                    kill_count += 1
                elif scenario_name == "deadly_corridor" and reward >= 90.0:
                    kill_count += 1
                
                # Store experience in buffer
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Update network
                loss = agent.update_model()
                if loss > 0.0:
                    episode_losses.append(loss)
                    
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                if done:
                    break
            
            # Compute stats
            duration = time.time() - start_time
            fps = (episode_steps * config['environment']['frame_skip']) / duration if duration > 0 else 0
            mean_loss = np.mean(episode_losses) if len(episode_losses) > 0 else 0.0
            
            recent_rewards.append(episode_reward)
            mean_reward_100 = np.mean(recent_rewards)
            
            # Save history
            metrics_history["episodes"].append(episode)
            metrics_history["episode_rewards"].append(float(episode_reward))
            metrics_history["mean_rewards_100"].append(float(mean_reward_100))
            metrics_history["kill_counts"].append(int(kill_count))
            metrics_history["survival_times"].append(float(episode_steps))
            metrics_history["losses"].append(float(mean_loss))
            metrics_history["epsilons"].append(float(agent.epsilon))
            metrics_history["fps_list"].append(float(fps))
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "episode_reward": episode_reward,
                "mean_reward_100": mean_reward_100,
                "kill_count": kill_count,
                "survival_steps": episode_steps,
                "loss": mean_loss,
                "epsilon": agent.epsilon,
                "fps": fps
            }, step=episode)
            
            # Print periodic logs
            if episode % config['logging']['log_every'] == 0:
                print(f"Ep {episode:4d}/{config['training']['total_episodes']} | "
                      f"Steps: {episode_steps:3d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Mean(100): {mean_reward_100:6.1f} | "
                      f"Loss: {mean_loss:.5f} | "
                      f"Eps: {agent.epsilon:.3f} | "
                      f"FPS: {fps:4.0f}")
                      
            # Save best model checkpoint
            if mean_reward_100 > best_mean_reward and episode >= 100:
                best_mean_reward = mean_reward_100
                checkpoint_path = os.path.join(save_dir, f"{config['model']['checkpoint_name']}_best.pth")
                agent.save(checkpoint_path)
                print(f"New best model saved to {checkpoint_path} (Mean Reward: {best_mean_reward:.1f})")
                
            # Periodically save regular checkpoint
            if episode % config['model']['save_every'] == 0:
                checkpoint_path = os.path.join(save_dir, f"{config['model']['checkpoint_name']}_ep_{episode}.pth")
                agent.save(checkpoint_path)
                
        # Export logs to JSON for dashboard
        export_logs_to_json(run.info.run_id, scenario_name, metrics_history)
        
    env.close()
    print(f"\n=======================================================")
    print(f" Training of {scenario_name.upper()} completed successfully!")
    print(f"=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent on ViZDoom Scenarios")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    train(args.config)
