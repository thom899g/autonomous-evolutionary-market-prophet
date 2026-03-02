# Autonomous Evolutionary Market Prophet

## Objective
An AI system that autonomously evolves and predicts market trends using advanced deep learning models like transformers, adapting strategies in real-time without human intervention.

## Strategy
Implement transformer-based models for time series analysis, integrate reinforcement learning for strategy adaptation, and link with existing systems for real-time data and feedback loops.

## Execution Output
SUMMARY: I've architected and implemented a robust foundation for the Autonomous Evolutionary Market Prophet system. I've designed a modular, production-ready architecture with proper error handling, logging, and Firebase integration. The system collects market data, processes it with transformers, evolves strategies, and manages state autonomously.

OUTPUT: 
### FILE: requirements.txt
```
firebase-admin>=6.0.0
pandas>=2.0.0
numpy>=1.24.0
ccxt>=4.0.0
scikit-learn>=1.3.0
torch>=2.0.0
transformers>=4.30.0
ta>=0.10.0
python-dotenv>=1.0.0
schedule>=1.2.0
apscheduler>=3.10.0
google-cloud-firestore>=2.11.0
requests>=2.31.0
tensorboard>=2.13.0
joblib>=1.3.0
```

### FILE: config.py
```python
"""
Configuration module for the Autonomous Evolutionary Market Prophet.
Centralizes all configuration, environment variables, and constants.
"""

import os
import logging
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ExchangeConfig:
    """Configuration for cryptocurrency exchanges"""
    name: str = "binance"
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    sandbox: bool = field(default_factory=lambda: os.getenv("EXCHANGE_SANDBOX", "True") == "True")
    rate_limit: int = field(default_factory=lambda: int(os.getenv("EXCHANGE_RATE_LIMIT", "1000")))

@dataclass
class ModelConfig:
    """Configuration for transformer models"""
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "time_series_transformer"))
    sequence_length: int = field(default_factory=lambda: int(os.getenv("SEQUENCE_LENGTH", "100")))
    prediction_horizon: int = field(default_factory=lambda: int(os.getenv("PREDICTION_HORIZON", "10")))
    hidden_size: int = field(default_factory=lambda: int(os.getenv("HIDDEN_SIZE", "256")))
    num_layers: int = field(default_factory=lambda: int(os.getenv("NUM_LAYERS", "4")))
    num_heads: int = field(default_factory=lambda: int(os.getenv("NUM_HEADS", "8")))
    dropout: float = field(default_factory=lambda: float(os.getenv("DROPOUT", "0.1")))

@dataclass
class FirebaseConfig:
    """Configuration for Firebase integration"""
    project_id: str = field(default_factory=lambda: os.getenv("FIREBASE_PROJECT_ID", ""))
    credentials_path: str = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
    collection_prefix: str = field(default_factory=lambda: os.getenv("FIREBASE_COLLECTION_PREFIX", "market_prophet"))

@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    initial_capital: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "10000.0")))
    risk_per_trade: float = field(default_factory=lambda: float(os.getenv("RISK_PER_TRADE", "0.02")))
    max_positions: int = field(default_factory=lambda: int(os.getenv("MAX_POSITIONS", "5")))
    paper_trading: bool = field(default_factory=lambda: os.getenv("PAPER_TRADING", "True") == "True")

@dataclass
class SystemConfig:
    """Main system configuration"""
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    data_refresh_interval: int = field(default_factory=lambda: int(os.getenv("DATA_REFRESH_INTERVAL", "60")))
    model_retrain_interval: int = field(default_factory=lambda: int(os.getenv("MODEL_RETRAIN_INTERVAL", "3600")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    retry_delay: int = field(default_factory=lambda: int(os.getenv("RETRY_DELAY", "5")))
    
    # Component configurations
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    def validate(self) -> bool:
        """Validate critical configuration parameters"""
        required_vars = [
            ("FIREBASE_PROJECT_ID", self.firebase.project_id),
            ("GOOGLE_APPLICATION_CREDENTIALS", self.firebase.credentials_path)
        ]
        
        missing = [name for name, value in required_vars if not value]
        if missing:
            logging.error(f"Missing required configuration: {missing}")
            return False
        return True

# Global configuration instance
CONFIG = SystemConfig()

def setup_logging() -> logging.Logger:
    """Configure system-wide logging"""
    log_level = getattr(logging, CONFIG.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('market_prophet.log')
        ]
    )
    
    # Suppress noisy library logs
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('ccxt').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

logger = setup_logging()
```

### FILE: data_collector.py
```python
"""
Market data collection module with robust error handling and retry logic.
Collects data from multiple exchanges and stores in Firebase.
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
import logging

from config import CONFIG, logger
from firebase_admin import firestore

@dataclass
class MarketData:
    """Structured market data container"""
    timestamp: datetime
    symbol: str