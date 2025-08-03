import os
import pandas as pd
from stable_baselines3 import PPO
from data_preparer import prepare_data, filter_trend_data, filter_range_data
from trading_env import TradingEnv

def train_specialist(data, model_name, timesteps=300000):
    """
    Trains a specialist PPO model on the given data.

    Args:
        data (pd.DataFrame): The data to train on.
        model_name (str): The name of the model to save.
        timesteps (int): The number of timesteps to train for.
    """
    if data.empty:
        print(f"No data to train {model_name}. Skipping.")
        return

    print(f"Training {model_name}...")
    env = TradingEnv(df=data)
    model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)

    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model.save(os.path.join(models_dir, model_name))
    print(f"{model_name} trained and saved.")

def main():
    """
    Main function to train the specialist agents.
    """
    # Define parameters
    symbol = 'BTC-USD'
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Prepare data
    try:
        full_data = prepare_data(symbol, start_date, end_date)
        print("Data prepared successfully.")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    # Filter data for specialists
    trend_data = filter_trend_data(full_data)
    range_data = filter_range_data(full_data)

    print(f"Trend data shape: {trend_data.shape}")
    print(f"Range data shape: {range_data.shape}")

    # Train specialists
    train_specialist(trend_data, "trend_specialist.zip")
    train_specialist(range_data, "range_specialist.zip")

if __name__ == "__main__":
    main()
