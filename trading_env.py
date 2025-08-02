import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, risk_free_rate=0.0):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0 for no position, 1 for long
        self.entry_price = 0
        self.current_step = 0
        self.portfolio_values = [initial_balance]
        self.risk_free_rate = risk_free_rate

        # Define the feature columns to be used for the observation
        # This makes the observation space independent of the number of columns in the df
        self.feature_columns = [col for col in df.columns if col.startswith(('EMA', 'VWAP', 'ADX', 'ATR', 'RSI', 'MACD'))]
        self.feature_columns.extend(['Open', 'High', 'Low', 'Close', 'Volume'])

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
        self.portfolio_values = [self.initial_balance]
        return self._get_observation()

    def _get_observation(self):
        """
        Gets the observation for the current step.
        """
        obs = self.df[self.feature_columns].iloc[self.current_step].values
        obs = np.append(obs, self.position)
        return obs

    def _calculate_reward(self):
        """
        Calculates a Sharpe Ratio-based reward.
        """
        if len(self.portfolio_values) < 2:
            return 0

        portfolio_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1

        # Simplified risk penalty using standard deviation of recent returns
        returns_window = pd.Series(self.portfolio_values).pct_change().dropna()
        risk = returns_window.std() if len(returns_window) > 1 else 0

        # Sharpe-like reward
        reward = portfolio_return - self.risk_free_rate - risk
        return reward if np.isfinite(reward) else 0


    def step(self, action):
        """
        Executes one time step within the environment.
        """
        current_price = self.df['Close'].iloc[self.current_step]

        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            self.balance += current_price - self.entry_price
            self.entry_price = 0

        # Update portfolio value
        # If we are in a long position, the portfolio value is the balance + the current value of the position
        # If we are not in a position, the portfolio value is just the balance
        if self.position == 1:
            portfolio_value = self.balance + (current_price - self.entry_price)
        else:
            portfolio_value = self.balance
        self.portfolio_values.append(portfolio_value)


        # Calculate reward
        reward = self._calculate_reward()

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
