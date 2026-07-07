import os
import random
import vizdoom as zd

def main():
    # 1. Create a game session instance
    game = zd.DoomGame()
    
    # 2. Locate the standard scenarios provided with the ViZDoom installation
    scenarios_dir = zd.scenarios_path
    
    # 3. Load the default configuration file for the Basic scenario
    config_path = os.path.join(scenarios_dir, "basic.cfg")
    game.load_config(config_path)
    
    # 4. Explicitly point the engine to the Basic scenario's WAD file
    wad_path = os.path.join(scenarios_dir, "basic.wad")
    game.set_doom_scenario_path(wad_path)
    
    # 5. Set screen dimensions to 160x120
    game.set_screen_resolution(zd.ScreenResolution.RES_160X120)
    
    # 6. Request grayscale frames instead of color (RGB)
    game.set_screen_format(zd.ScreenFormat.GRAY8)
    
    # 7. Run the game invisibly (headless mode)
    game.set_window_visible(False)
    
    # 8. Start the Doom engine process
    game.init()
    
    # 9. Define our possible actions (Turn Left, Turn Right, Shoot)
    actions = [
        [1, 0, 0],  # turn left
        [0, 1, 0],  # turn right
        [0, 0, 1]   # shoot
    ]
    
    print("ViZDoom initialized successfully!")
    
    # 10. Pull the first state to inspect the frame data shape
    state = game.get_state()
    print(f"Screen buffer shape: {state.screen_buffer.shape}")
    print(f"Screen buffer data type: {state.screen_buffer.dtype}")
    
    # 11. Run a test episode with random actions
    game.new_episode()
    total_reward = 0
    while not game.is_episode_finished():
        state = game.get_state()
        action = random.choice(actions)
        
        # Take the action, hold it for 4 frames, and receive the reward
        reward = game.make_action(action, 4)
        total_reward += reward
        
    print(f"Episode finished! Total reward: {total_reward}")
    
    # 12. Shut down the engine process safely
    game.close()

if __name__ == "__main__":
    main()
