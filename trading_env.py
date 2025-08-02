import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0 for no position, 1 for long
        self.entry_price = 0
        self.current_step = 0

        # Actions: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: OHLCV + indicators + position
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(df.columns) + 1,), dtype=np.float32
        )

    def reset(self):
        """
        Resets the state of the environment to an initial state.
        """
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        """
        Gets the observation for the current step.
        """
        obs = self.df.iloc[self.current_step].values
        obs = np.append(obs, self.position)
        return obs

    def _calculate_reward(self, action):
        """
        Calculates the reward for a given action.
        """
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0

        if action == 2 and self.position == 1:  # Sell
            reward = current_price - self.entry_price

        return reward

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0
        done = False

        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:  # Sell
            reward = self._calculate_reward(action)
            self.position = 0
            self.balance += current_price - self.entry_price
            self.entry_price = 0

        # Move to the next step
        self.current_step += 1

        # Check if the episode is done
        if self.current_step >= len(self.df) - 1 or self.balance <= 0:
            done = True

        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': 'Long' if self.position == 1 else 'None',
            'price': current_price
        }

        return obs, reward, done, info
