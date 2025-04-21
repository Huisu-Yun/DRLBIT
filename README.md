# Upbit Data Collector

A Python-based data collection system for Upbit cryptocurrency exchange data.

## Features

- Collects real-time market data from Upbit
- Supports both REST API and WebSocket connections
- Stores data in InfluxDB
- Handles multiple markets simultaneously
- Configurable data collection intervals

## Requirements

- Python 3.8+
- InfluxDB 2.0+
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your environment variables in `.env` file

## Configuration

Create a `.env` file with the following variables:

```
INFLUXDB_URL=your_influxdb_url
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=your_influxdb_org
INFLUXDB_BUCKET=your_influxdb_bucket
```

## Usage

Run the data collector:

```bash
python main.py
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 