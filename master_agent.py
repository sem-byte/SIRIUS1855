import numpy as np

class MasterAgent:
    """
    The Master Agent determines the market phase and selects the appropriate specialist.
    """
    def __init__(self, adx_threshold=25, range_threshold=20):
        self.adx_threshold = adx_threshold
        self.range_threshold = range_threshold

    def get_market_phase(self, current_data):
        """
        Determines the market phase based on ADX and other indicators.

        Args:
            current_data (pd.Series): The current market data, including indicators.

        Returns:
            str: The market phase ('TREND', 'RANGE', or 'UNCERTAIN').
        """
        adx = current_data['ADX_14']

        if adx > self.adx_threshold:
            return 'TREND'
        elif adx < self.range_threshold:
            return 'RANGE'
        else:
            return 'UNCERTAIN'
