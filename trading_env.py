import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, risk_percentage=0.02):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_percentage = risk_percentage
        self.position = 0  # 0 for no position, > 0 for long
        self.entry_price = 0
        self.current_step = 0
        self.peak_balance = initial_balance

        # Define the feature columns for the observation space
        self.feature_columns = [
            'Close', 'Volume', 'RSI_14', 'MACDh_12_26_9',
            'open_interest', 'long_short_ratio', 'funding_rate', 'ATR_p'
        ]

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
        self.peak_balance = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        """
        Gets the observation for the current step.
        """
        obs = self.df[self.feature_columns].iloc[self.current_step].values
        obs = np.append(obs, self.position)
        return obs

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        current_price = self.df['Close'].iloc[self.current_step]
        atr = self.df['ATRr_14'].iloc[self.current_step]
        reward = 0

        # Drawdown penalty
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        reward -= drawdown * 0.01

        # Execute action
        if action == 1 and self.position == 0:  # Buy
            # Dynamic position sizing
            risk_amount = self.balance * self.risk_percentage
            stop_loss_price = current_price - (atr * 2) # Example stop-loss

            if current_price > stop_loss_price:
                position_size = risk_amount / (current_price - stop_loss_price)
                self.position = position_size
                self.entry_price = current_price

        elif action == 2 and self.position > 0:  # Sell
            profit = (current_price - self.entry_price) * self.position
            self.balance += profit

            if profit > 0:
                reward += profit
            else:
                reward += profit * 2 # Penalize losses more severely

            self.position = 0
            self.entry_price = 0

        # Move to the next step
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= len(self.df) - 1 or self.balance <= 0

        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': 'Long' if self.position > 0 else 'None',
            'price': current_price
        }

        return obs, reward, done, info
