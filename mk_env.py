"""
Gymnasium environment skeleton for Mortal Kombat via MAME.

TODO: Replace the dummy implementation with real MAME communication.
See README for integration options (MAMEToolkit, Lua socket, screen capture).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MortalKombatMAMEEnv(gym.Env):
    """Placeholder env. Implement MAME bridge here."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, frame_skip=4):
        super().__init__()
        self.frame_skip = frame_skip
        # Observation: stacked grayscale frames (or feature vector)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        # Discrete actions: simplified button combos
        # 0=idle, 1=left, 2=right, 3=up, 4=down, 5=punch, 6=kick, 7=block, 8=special, ...
        self.action_space = spaces.Discrete(16)
        self.render_mode = render_mode
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        # TODO: reset MAME instance, load savestate at fight start
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        # TODO:
        # 1. Send action to MAME (Lua / input)
        # 2. Step emulator frame_skip times
        # 3. Read health delta, round state from RAM or screen
        # 4. Compute shaped reward
        # 5. Return obs, reward, terminated, truncated, info

        self._step_count += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        reward = 0.0  # replace with real damage/win reward
        terminated = False
        truncated = self._step_count > 2000  # safety timeout
        info = {"step": self._step_count}

        return obs, reward, terminated, truncated, info

    def render(self):
        # TODO: return current frame
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def close(self):
        # TODO: kill MAME process
        pass
