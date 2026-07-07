import os
import numpy as np
import vizdoom as zd

class VizDoomEnv:
    """
    A custom wrapper for the ViZDoom game engine.
    Converts raw game outputs to a clean gym-like step/reset interface,
    configures composite actions, and calculates shaped progress rewards.
    """
    def __init__(self, config_path, wad_path, window_visible=False, reward_shaping=None):
        self.game = zd.DoomGame()
        
        config_path = os.path.abspath(config_path)
        wad_path = os.path.abspath(wad_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        if not os.path.exists(wad_path):
            raise FileNotFoundError(f"WAD file not found at: {wad_path}")
            
        self.game.load_config(config_path)
        self.game.set_doom_scenario_path(wad_path)
        self.game.set_window_visible(window_visible)
        
        self.game.set_screen_resolution(zd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(zd.ScreenFormat.GRAY8)
        self.game.init()
        
        # 1. Define custom action combinations depending on the scenario name
        scenario_name = os.path.basename(config_path).replace(".cfg", "")
        self._setup_actions(scenario_name)
        
        self.reward_shaping = reward_shaping
        
        # Tracker variables for reward shaping
        self.last_health = 100.0
        self.last_ammo = 0.0
        self.last_distance = 0.0

    def _setup_actions(self, scenario_name):
        """
        Builds discrete composite actions to allow the agent to run and combat 
        simultaneously, rather than forcing single button constraints.
        """
        # Buttons mapped:
        # Basic: [MOVE_LEFT, MOVE_RIGHT, ATTACK]
        # Defend Center: [TURN_LEFT, TURN_RIGHT, ATTACK]
        # Deadly Corridor: [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]
        
        if scenario_name == "basic":
            self.actions = [
                [1, 0, 0], # Move Left
                [0, 1, 0], # Move Right
                [0, 0, 1], # Attack
                [1, 0, 1], # Move Left + Attack
                [0, 1, 1]  # Move Right + Attack
            ]
        elif scenario_name == "defend_the_center":
            self.actions = [
                [1, 0, 0], # Turn Left
                [0, 1, 0], # Turn Right
                [0, 0, 1], # Attack
                [1, 0, 1], # Turn Left + Attack
                [0, 1, 1]  # Turn Right + Attack
            ]
        else: # deadly_corridor
            # Custom action combinations (7 inputs: L, R, Attack, Fwd, Bwd, TurnL, TurnR)
            self.actions = [
                [0, 0, 0, 1, 0, 0, 0], # 0. Move Forward
                [0, 0, 0, 0, 0, 1, 0], # 1. Turn Left
                [0, 0, 0, 0, 0, 0, 1], # 2. Turn Right
                [0, 0, 1, 0, 0, 0, 0], # 3. Attack (Shoot)
                [0, 0, 1, 1, 0, 0, 0], # 4. Move Forward + Attack
                [0, 0, 0, 1, 0, 1, 0], # 5. Move Forward + Turn Left
                [0, 0, 0, 1, 0, 0, 1], # 6. Move Forward + Turn Right
                [0, 0, 1, 0, 0, 1, 0], # 7. Turn Left + Attack
                [0, 0, 1, 0, 0, 0, 1], # 8. Turn Right + Attack
                [1, 0, 0, 0, 0, 0, 0], # 9. Move Left
                [0, 1, 0, 0, 0, 0, 0], # 10. Move Right
            ]
            
        self.action_space_size = len(self.actions)

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state()
        
        self.last_health = 100.0
        
        if state is not None and state.game_variables is not None:
            self.last_ammo = state.game_variables[1] if len(state.game_variables) > 1 else 0.0
            self.last_distance = self.get_distance_to_target(state)
        else:
            self.last_ammo = 0.0
            self.last_distance = 0.0
            
        if state is None or state.screen_buffer is None:
            return np.zeros((120, 160), dtype=np.uint8)
            
        return state.screen_buffer

    def step(self, action_idx, frame_skip=4):
        action = self.actions[action_idx]
        
        base_reward = self.game.make_action(action, frame_skip)
        done = self.game.is_episode_finished()
        
        if done:
            obs = None
            reward = base_reward
            if self.reward_shaping and "death_penalty" in self.reward_shaping:
                reward += self.reward_shaping["death_penalty"]
        else:
            state = self.game.get_state()
            if state is not None and state.screen_buffer is not None:
                obs = state.screen_buffer
            else:
                obs = np.zeros((120, 160), dtype=np.uint8)
                
            reward = base_reward
            if self.reward_shaping:
                reward += self.calculate_shaped_reward(state, base_reward)
                
        return obs, reward, done, {}

    def get_distance_to_target(self, state):
        """Calculates distance to target exit coordinate (X=0, Y=512)."""
        if state is None or state.game_variables is None or len(state.game_variables) < 4:
            return 0.0
            
        # index 2: X, index 3: Y
        x = state.game_variables[2]
        y = state.game_variables[3]
        
        target_x = 0.0
        target_y = 512.0
        
        return float(np.sqrt((x - target_x)**2 + (y - target_y)**2))

    def calculate_shaped_reward(self, state, base_reward):
        shaped_part = 0.0
        if state is None or state.game_variables is None or len(state.game_variables) == 0:
            return shaped_part

        # 1. Health damage penalty (index 0)
        current_health = state.game_variables[0]
        health_diff = current_health - self.last_health
        if health_diff < 0:
            shaped_part += health_diff * 1.0  
        self.last_health = current_health

        # 2. Ammo waste penalty (index 1)
        if len(state.game_variables) > 1:
            current_ammo = state.game_variables[1]
            ammo_diff = current_ammo - self.last_ammo
            if ammo_diff < 0:
                if "ammo_waste_penalty" in self.reward_shaping:
                    shaped_part += self.reward_shaping["ammo_waste_penalty"]
            self.last_ammo = current_ammo

        # 3. Movement progress toward target (indexes 2, 3)
        if len(state.game_variables) > 3:
            current_distance = self.get_distance_to_target(state)
            distance_diff = self.last_distance - current_distance  # Positive means got closer
            if distance_diff > 0 and "progress_reward_scale" in self.reward_shaping:
                shaped_part += distance_diff * self.reward_shaping["progress_reward_scale"]
            self.last_distance = current_distance

        return shaped_part

    def close(self):
        self.game.close()
