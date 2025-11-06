"""
OANDA API configuration and credentials.

NOTE: You need to generate an API token from OANDA dashboard.
The master password CANNOT be used for API access.

To get your API token:
1. Login to https://www.oanda.com/demo-account/login
   Username: 1600037272
   Password: 8?p1?$$*kW3

2. Go to "Manage API Access" or Account Settings
3. Generate a Personal Access Token (PAT)
4. Copy the token and paste it below in 'api_token'

Your token will look like: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
"""
import os

# OANDA Demo Account Credentials
OANDA_CONFIG = {
    'account_id': '1600037272',
    'api_token': os.getenv('OANDA_API_TOKEN', 'PLEASE_GENERATE_TOKEN_FROM_OANDA_DASHBOARD'),
    'environment': 'practice',  # 'practice' for demo, 'live' for real
    'server': 'OANDA-Demo-1',
}

# API Endpoints
API_ENDPOINTS = {
    'practice': 'https://api-fxpractice.oanda.com',
    'live': 'https://api-fxtrade.oanda.com',
    'stream_practice': 'https://stream-fxpractice.oanda.com',
    'stream_live': 'https://stream-fxtrade.oanda.com',
}

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'USD_JPY',  # OANDA uses underscore format
    'initial_balance': 100000,
    'risk_per_trade': 0.01,  # 1%
    'max_open_positions': 3,
    'max_daily_loss': 0.03,  # 3%
    'max_drawdown': 0.10,  # 10%
}

# Risk Guardrails
RISK_LIMITS = {
    'max_risk_per_trade': 0.01,  # 1%
    'max_daily_loss': 0.03,  # 3%
    'max_weekly_loss': 0.05,  # 5%
    'max_drawdown_halt': 0.10,  # 10% - emergency stop
    'max_open_positions': 3,
    'min_model_confidence': 0.80,  # Only trade when prob >= 0.80
    'min_win_rate_alert': 0.65,  # Alert if win rate drops below 65%
}

def get_api_url(environment='practice'):
    """Get API URL for environment."""
    return API_ENDPOINTS.get(environment, API_ENDPOINTS['practice'])

def get_stream_url(environment='practice'):
    """Get streaming URL for environment."""
    key = f'stream_{environment}'
    return API_ENDPOINTS.get(key, API_ENDPOINTS['stream_practice'])
