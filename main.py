import os
import pandas as pd
from stable_baselines3 import PPO
from data_preparer import prepare_data
from trading_env import TradingEnv
from master_agent import MasterAgent
import quantstats as qs
import matplotlib.pyplot as plt

def backtest_committee(results_df):
    """
    Calculates and prints the final performance metrics.
    """
    print("\n--- Backtest Results ---")

    # Calculate returns
    returns = results_df['balance'].pct_change().dropna()

    # --- Performance Metrics ---
    total_return = (results_df['balance'].iloc[-1] / results_df['balance'].iloc[0]) - 1
    sharpe_ratio = qs.stats.sharpe(returns)
    max_drawdown = qs.stats.max_drawdown(results_df['balance'])

    # --- Win Rate ---
    trades = results_df[results_df['position'] != results_df['position'].shift(1)]
    wins = trades[trades['balance'] > trades['balance'].shift(1)]
    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0

    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")

    # --- Phase Analysis ---
    phase_distribution = results_df['phase'].value_counts(normalize=True)
    print("\n--- Market Phase Distribution ---")
    print(phase_distribution)

    trade_phases = trades['phase'].value_counts(normalize=True)
    print("\n--- Trade Distribution by Phase ---")
    print(trade_phases)

def plot_results(results_df, symbol):
    """
    Generates and saves the final plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot Price and Phases
    ax1.plot(results_df.index, results_df['price'], label='Price')

    phase_colors = {'TREND': (0, 1, 0, 0.2), 'RANGE': (1, 0, 0, 0.2), 'UNCERTAIN': (1, 1, 0, 0.2)}
    for phase, color in phase_colors.items():
        ax1.fill_between(results_df.index, results_df['price'].min(), results_df['price'].max(),
                         where=results_df['phase'] == phase, facecolor=color, label=phase)

    ax1.set_title(f'{symbol} Price and Market Phases')
    ax1.legend()

    # Plot Equity Curve
    ax2.plot(results_df.index, results_df['balance'], label='Equity Curve', color='blue')
    ax2.set_title('Portfolio Equity Curve')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('backtest_results.png')
    print("\nPlot saved to backtest_results.png")

def main():
    """
    Main function to run the backtest for the expert committee system.
    """
    # Define parameters
    symbol = 'BTC-USD'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    initial_balance = 10000

    # Prepare data
    try:
        full_data = prepare_data(symbol, start_date, end_date)
        print("Data prepared successfully.")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    # Load trained models
    models_dir = "models"
    trend_model_path = os.path.join(models_dir, "trend_specialist.zip")
    range_model_path = os.path.join(models_dir, "range_specialist.zip")

    if not os.path.exists(trend_model_path) or not os.path.exists(range_model_path):
        print("Trained models not found. Please run train_specialists.py first.")
        return

    trend_model = PPO.load(trend_model_path)
    range_model = PPO.load(range_model_path)
    print("Specialist models loaded successfully.")

    # Initialize agents and environment
    master_agent = MasterAgent()
    features = ['Close', 'Volume', 'RSI_14', 'ADX_14']
    env = TradingEnv(df=full_data, initial_balance=initial_balance, feature_columns=features)
    obs = env.reset()

    results = {
        'date': [],
        'balance': [],
        'position': [],
        'phase': [],
        'price': []
    }

    # Set initial values
    results['date'].append(full_data.index[0])
    results['balance'].append(initial_balance)
    results['position'].append('None')
    results['phase'].append('None')
    results['price'].append(full_data['Close'].iloc[0])


    print("\nStarting backtest with Expert Committee...")
    for i in range(len(full_data) - 1):
        current_data = full_data.iloc[i]
        phase = master_agent.get_market_phase(current_data)

        if phase == 'TREND':
            action, _ = trend_model.predict(obs, deterministic=True)
        elif phase == 'RANGE':
            action, _ = range_model.predict(obs, deterministic=True)
        else:  # UNCERTAIN
            action = 0  # Hold

        obs, reward, done, info = env.step(action)

        results['date'].append(full_data.index[i + 1])
        results['balance'].append(info['balance'])
        results['position'].append(info['position'])
        results['phase'].append(phase)
        results['price'].append(info['price'])

        if done:
            print("Backtest finished.")
            break

    print("\nBacktest completed.")

    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)

    backtest_committee(results_df)
    plot_results(results_df, symbol)


if __name__ == "__main__":
    main()
