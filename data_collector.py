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