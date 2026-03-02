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