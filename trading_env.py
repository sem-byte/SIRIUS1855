import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, feature_columns=None):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0 for no position, 1 for long
        self.entry_price = 0
        self.current_step = 0
        self.hold_duration = 0

        # Define the feature columns for the observation space
        if feature_columns is None:
            self.feature_columns = ['Close', 'Volume', 'RSI_14', 'MACDh_12_26_9']
        else:
            self.feature_columns = feature_columns

        # Actions: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: features + position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_columns) + 1,), dtype=np.float32
        )

    def reset(self):
        """
        Resets the state of the environment to an initial state.
        """
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.hold_duration = 0
        return self._get_observation()

    def _get_observation(self):
        """
        Gets the observation for the current step using the specified feature set.
        """
        obs = self.df[self.feature_columns].iloc[self.current_step].values
        obs = np.append(obs, self.position)
        return obs

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0

        # Penalty for holding no position
        if self.position == 0:
            reward -= 0.00001 # Very small penalty for holding no position

        # Penalty for holding a position
        if self.position == 1:
            self.hold_duration += 1
            reward -= 0.0001 * self.hold_duration # Time decay penalty

        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.hold_duration = 0
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            profit = current_price - self.entry_price
            self.balance += profit
            self.entry_price = 0
            self.hold_duration = 0

            if profit > 0:
                reward += profit # Large positive reward for profitable trade
            else:
                reward += profit # Large negative reward (penalty) for unprofitable trade

        # Move to the next step
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= len(self.df) - 1 or self.balance <= 0

        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': 'Long' if self.position == 1 else 'None',
            'price': current_price
        }

        return obs, reward, done, info
