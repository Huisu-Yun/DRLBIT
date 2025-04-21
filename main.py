import asyncio
import logging
from datetime import datetime
import signal
import sys
from typing import Dict
import json

from config import DEFAULT_MARKETS
from rest_client import UpbitRestClient
from websocket_client import UpbitWebSocketClient
from influx_client import InfluxDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UpbitDataCollector:
    def __init__(self):
        self.rest_client = UpbitRestClient()
        self.ws_client = UpbitWebSocketClient()
        self.influx_client = InfluxDBManager()
        self.running = True
        self.markets = DEFAULT_MARKETS
        self._tasks = []

    async def collect_rest_data(self):
        """Collect data using REST API periodically."""
        try:
            while self.running:
                try:
                    # Collect ticker data
                    ticker_data = self.rest_client.get_ticker(self.markets)
                    #logger.info(f"Received ticker data for {len(ticker_data)} markets")
                    for data in ticker_data:
                        # market = data.get("market", "unknown")
                        # price = data.get("trade_price", "N/A")
                        # volume = data.get("acc_trade_volume_24h", "N/A")
                        #logger.info(f"Ticker - Market: {market}, Price: {price}, 24h Volume: {volume}")
                        
                        influx_data = self.rest_client.format_for_influx(
                            data,
                            "ticker",
                            market
                        )
                        success = self.influx_client.write_data(influx_data)
                        if success:
                            logger.debug(f"Successfully wrote ticker data for {market} to InfluxDB")
                        else:
                            logger.warning(f"Failed to write ticker data for {market} to InfluxDB")

                    # Collect orderbook data
                    orderbook_data = self.rest_client.get_orderbook(self.markets)
                    #logger.info(f"Received orderbook data for {len(orderbook_data)} markets")
                    for data in orderbook_data:
                        market = data.get("market", "unknown")
                        total_ask = data.get("total_ask_size", "N/A")
                        total_bid = data.get("total_bid_size", "N/A")
                        #logger.info(f"Orderbook - Market: {market}, Total Ask: {total_ask}, Total Bid: {total_bid}")
                        
                        # Log orderbook units details
                        orderbook_units = data.get("orderbook_units", [])
                        # for i, unit in enumerate(orderbook_units):
                        #     logger.info(
                        #         f"Orderbook Unit {i} - Market: {market}, "
                        #         f"Ask Price: {unit.get('ask_price', 'N/A')}, "
                        #         f"Bid Price: {unit.get('bid_price', 'N/A')}, "
                        #         f"Ask Size: {unit.get('ask_size', 'N/A')}, "
                        #         f"Bid Size: {unit.get('bid_size', 'N/A')}"
                        #     )

                        # Prepare influx data with orderbook units
                        influx_data = {
                            "measurement": "orderbook",
                            "tags": {
                                "market": market
                            },
                            "time": datetime.utcnow(),
                            "fields": {
                                "total_ask_size": float(total_ask) if total_ask != "N/A" else 0.0,
                                "total_bid_size": float(total_bid) if total_bid != "N/A" else 0.0
                            }
                        }

                        # Add orderbook units data
                        for i, unit in enumerate(orderbook_units):
                            prefix = f"unit_{i}_"
                            influx_data["fields"].update({
                                f"{prefix}ask_price": float(unit.get('ask_price', 0)),
                                f"{prefix}bid_price": float(unit.get('bid_price', 0)),
                                f"{prefix}ask_size": float(unit.get('ask_size', 0)),
                                f"{prefix}bid_size": float(unit.get('bid_size', 0))
                            })

                        success = self.influx_client.write_data(influx_data)
                        if success:
                            logger.debug(f"Successfully wrote orderbook data for {market} to InfluxDB")
                        else:
                            logger.warning(f"Failed to write orderbook data for {market} to InfluxDB")

                    await asyncio.sleep(1)  # Wait for 1 second before next collection
                except Exception as e:
                    logger.error(f"Error in REST data collection: {e}")
                    await asyncio.sleep(5)  # Wait longer on error
        except asyncio.CancelledError:
            logger.info("REST data collection task cancelled")
            raise

    async def handle_ws_message(self, message: Dict):
        """Handle incoming WebSocket messages."""
        try:
            market = message.get("cd", "unknown")
            trade_price = message.get("tp", "N/A")
            trade_volume = message.get("tv", "N/A")
            change = message.get("c", "N/A")
            change_price = message.get("cp", "N/A")
            ask_bid = message.get("ab", "N/A")
            
            # logger.info(
            #     f"WebSocket Trade - Market: {market}, "
            #     f"Price: {trade_price}, Volume: {trade_volume}, "
            #     f"Change: {change} ({change_price}), Type: {ask_bid}, "                    
            #     f"Best Ask: {message.get('bap', 'N/A')}({message.get('bas', 'N/A')}), "
            #     f"Best Bid: {message.get('bbp', 'N/A')}({message.get('bbs', 'N/A')})"
            # )
            
            influx_data = self.ws_client.format_for_influx(message)
            # logger.info(influx_data)
            success = self.influx_client.write_data(influx_data)
            if success:
                logger.debug(f"Successfully wrote trade data for {market} to InfluxDB")
            else:
                logger.warning(f"Failed to write trade data for {market} to InfluxDB")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    async def start_websocket(self):
        """Start WebSocket connection and message processing."""
        try:
            self.ws_client.add_callback(self.handle_ws_message)
            while self.running:
                try:
                    logger.info("Attempting to connect to WebSocket...")
                    connected = await self.ws_client.connect(
                        self.markets,
                        ["trade"]
                    )
                    if connected:
                        logger.info("Successfully connected to WebSocket")
                        await self.ws_client.process_messages()
                    else:
                        logger.warning("Failed to connect to WebSocket, retrying in 5 seconds...")
                        await asyncio.sleep(5)  # Wait before reconnecting
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("WebSocket task cancelled")
            raise

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Initiating shutdown...")
        self.running = False
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        await self.ws_client.close()
        self.influx_client.close()
        logger.info("Cleanup completed")

    async def run(self):
        """Run the data collector."""
        try:
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.handle_shutdown)
            signal.signal(signal.SIGTERM, self.handle_shutdown)

            logger.info(f"Starting data collection for markets: {', '.join(self.markets)}")
            
            # Create tasks for REST and WebSocket data collection
            self._tasks = [
                asyncio.create_task(self.collect_rest_data()),
                asyncio.create_task(self.start_websocket())
            ]

            # Wait for tasks to complete or be cancelled
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled, starting cleanup...")
            finally:
                await self.cleanup()

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await self.cleanup()

async def main():
    """Main entry point with proper signal handling."""
    collector = UpbitDataCollector()
    try:
        await collector.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user") 