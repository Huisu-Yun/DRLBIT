# Upbit Data Collector and Trading Bot

A Python-based cryptocurrency trading bot that collects data from Upbit and implements deep reinforcement learning for automated trading.

## Features

- Real-time data collection from Upbit exchange
- Data storage in InfluxDB for time-series analysis
- Deep Reinforcement Learning (DRL) trading agent
  - LSTM-based architecture with attention mechanism
  - Dueling DQN for better action value estimation
  - Experience replay for stable training
- Comprehensive technical indicators and feature engineering
- Real-time market regime detection using HMM

## Project Structure

```
upbit_data_collector/
├── config.py                 # Main configuration file
├── influx_client.py         # InfluxDB client wrapper
├── Deep_RL/                 # Deep RL trading module
│   ├── config.py           # DRL configuration
│   ├── data_preprocessor.py # Data preprocessing
│   ├── dqn_model.py        # DQN model architecture
│   ├── trading_env.py      # Trading environment
│   ├── train.py           # Training script
│   └── test_env.py        # Testing script
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/upbit_data_collector.git
cd upbit_data_collector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure InfluxDB:
- Install InfluxDB
- Create a bucket named "UPBIT"
- Update `config.py` with your InfluxDB credentials

## Usage

1. Data Collection:
```bash
python influx_client.py
```

2. Train the DRL agent:
```bash
python Deep_RL/train.py
```

3. Test the environment:
```bash
python Deep_RL/test_env.py
```

## Configuration

1. Main Configuration (`config.py`):
- InfluxDB connection settings
- Data collection parameters

2. DRL Configuration (`Deep_RL/config.py`):
- Model hyperparameters
- Training settings
- Checkpointing configuration

## Dependencies

- Python 3.8+
- PyTorch
- InfluxDB Client
- Gym
- NumPy
- Pandas
- scikit-learn
- hmmlearn

## License

MIT License 