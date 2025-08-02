from data_preparer import prepare_data
from trading_env import TradingEnv
import numpy as np

def main():
    """
    Main function to run the trading bot.
    """
    # Define parameters
    symbol = 'BTC-USD'
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Prepare data
    try:
        data = prepare_data(symbol, start_date, end_date)
        if data.empty:
            print("No data loaded, please check the symbol and date range.")
            return
        print("Data prepared successfully.")
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    # Initialize environment
    env = TradingEnv(df=data)

    # Test loop
    print("\nStarting test loop with random actions...")
    obs = env.reset()
    for i in range(100):
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, info = env.step(action)

        action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

        print(f"Step {i+1}:")
        print(f"  Action: {action_map[action]}")
        print(f"  Reward: {reward}")
        print(f"  Info: {info}")

        if done:
            print("Episode finished.")
            break

    print("\nTest loop finished.")

if __name__ == "__main__":
    main()
