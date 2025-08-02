import asyncio
import json
import websockets
import time

class L3Collector:
    """
    A WebSocket client for collecting L3 market data from Bybit.
    """
    def __init__(self, symbols):
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        self.symbols = symbols
        self.topics = ["orderbook.50", "publicTrade", "liquidation"]

    async def _subscribe(self, ws):
        """
        Sends subscription messages to the WebSocket for the specified topics and symbols.
        """
        args = [f"{topic}.{symbol}" for symbol in self.symbols for topic in self.topics]

        subscription_message = {
            "op": "subscribe",
            "args": args
        }
        await ws.send(json.dumps(subscription_message))
        print(f"Subscribed to: {args}")

    def _handle_message(self, message):
        """
        Parses and prints a formatted summary of the received message.
        """
        data = json.loads(message)

        if "topic" in data:
            topic = data["topic"]
            if topic.startswith("orderbook"):
                orderbook_data = data["data"]
                symbol = orderbook_data["s"]
                best_bid = orderbook_data["b"][0][0] if orderbook_data["b"] else "N/A"
                best_ask = orderbook_data["a"][0][0] if orderbook_data["a"] else "N/A"
                print(f"[ORDERBOOK] {symbol} | Best Bid: {best_bid} | Best Ask: {best_ask}")

            elif topic.startswith("publicTrade"):
                for trade in data["data"]:
                    symbol = trade["s"]
                    side = trade["S"]
                    price = trade["p"]
                    qty = trade["v"]
                    print(f"[TRADE] {symbol} | SIDE: {side} | PRICE: {price} | QTY: {qty}")

            elif topic.startswith("liquidation"):
                liquidation_data = data["data"]
                symbol = liquidation_data["symbol"]
                side = liquidation_data["side"]
                price = liquidation_data["price"]
                qty = liquidation_data["size"]
                status = liquidation_data["updatedTime"] # Using updatedTime as a proxy for status
                print(f"[LIQUIDATION] {symbol} | SIDE: {side} | PRICE: {price} | QTY: {qty} | STATUS: Filled at {status}")


    async def run(self):
        """
        Main loop to connect to the WebSocket and handle messages.
        Includes auto-reconnect logic.
        """
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    print("Connected to Bybit WebSocket.")
                    await self._subscribe(ws)
                    while True:
                        message = await ws.recv()
                        self._handle_message(message)

            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
                print(f"Connection lost: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

if __name__ == "__main__":
    symbols_to_track = ["BTCUSDT", "ETHUSDT"]
    collector = L3Collector(symbols=symbols_to_track)
    asyncio.run(collector.run())
