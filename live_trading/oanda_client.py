"""
OANDA API client for live trading.

Handles:
- Authentication
- Account management
- Market data streaming
- Order execution
- Position management
"""
import requests
import json
import time
from datetime import datetime
import pandas as pd
from oanda_config import OANDA_CONFIG, get_api_url, get_stream_url


class OandaClient:
    """OANDA API client."""
    
    def __init__(self, environment='practice'):
        """
        Initialize OANDA client.
        
        Parameters:
        -----------
        environment : str
            'practice' for demo or 'live' for real account
        """
        self.environment = environment
        self.account_id = OANDA_CONFIG['account_id']
        self.api_token = OANDA_CONFIG['api_token']
        self.api_url = get_api_url(environment)
        self.stream_url = get_stream_url(environment)
        
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
    
    def get_account_info(self):
        """Get account information."""
        url = f"{self.api_url}/v3/accounts/{self.account_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None
    
    def get_account_summary(self):
        """Get account summary (balance, equity, positions)."""
        url = f"{self.api_url}/v3/accounts/{self.account_id}/summary"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            summary = data.get('account', {})
            return {
                'balance': float(summary.get('balance', 0)),
                'equity': float(summary.get('NAV', 0)),  # Net Asset Value
                'unrealized_pl': float(summary.get('unrealizedPL', 0)),
                'margin_used': float(summary.get('marginUsed', 0)),
                'margin_available': float(summary.get('marginAvailable', 0)),
                'open_trades': int(summary.get('openTradeCount', 0)),
                'open_positions': int(summary.get('openPositionCount', 0)),
            }
        except Exception as e:
            print(f"Error getting account summary: {e}")
            return None
    
    def get_candles(self, instrument, granularity='M15', count=500):
        """
        Get historical candles.
        
        Parameters:
        -----------
        instrument : str
            Currency pair (e.g., 'USD_JPY')
        granularity : str
            Timeframe (M1, M5, M15, M30, H1, H4, D)
        count : int
            Number of candles to retrieve
            
        Returns:
        --------
        pd.DataFrame
            OHLCV data
        """
        url = f"{self.api_url}/v3/instruments/{instrument}/candles"
        params = {
            'granularity': granularity,
            'count': count,
            'price': 'MBA'  # Mid, Bid, Ask
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            candles = []
            for candle in data.get('candles', []):
                if candle.get('complete'):
                    mid = candle.get('mid', {})
                    candles.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(mid.get('o', 0)),
                        'high': float(mid.get('h', 0)),
                        'low': float(mid.get('l', 0)),
                        'close': float(mid.get('c', 0)),
                        'volume': int(candle.get('volume', 0))
                    })
            
            df = pd.DataFrame(candles)
            return df
            
        except Exception as e:
            print(f"Error getting candles: {e}")
            return None
    
    def get_current_price(self, instrument):
        """
        Get current bid/ask price.
        
        Parameters:
        -----------
        instrument : str
            Currency pair (e.g., 'USD_JPY')
            
        Returns:
        --------
        dict
            {'bid': float, 'ask': float, 'mid': float, 'spread': float}
        """
        url = f"{self.api_url}/v3/accounts/{self.account_id}/pricing"
        params = {'instruments': instrument}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            prices = data.get('prices', [])
            if prices:
                price = prices[0]
                bid = float(price.get('bids', [{}])[0].get('price', 0))
                ask = float(price.get('asks', [{}])[0].get('price', 0))
                mid = (bid + ask) / 2
                spread = ask - bid
                
                return {
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'spread': spread,
                    'time': pd.to_datetime(price.get('time'))
                }
            
            return None
            
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None
    
    def get_open_positions(self):
        """Get all open positions."""
        url = f"{self.api_url}/v3/accounts/{self.account_id}/openPositions"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            positions = []
            for pos in data.get('positions', []):
                long_units = float(pos.get('long', {}).get('units', 0))
                short_units = float(pos.get('short', {}).get('units', 0))
                net_units = long_units + short_units
                
                if net_units != 0:
                    positions.append({
                        'instrument': pos.get('instrument'),
                        'units': net_units,
                        'side': 'long' if net_units > 0 else 'short',
                        'avg_price': float(pos.get('long' if net_units > 0 else 'short', {}).get('averagePrice', 0)),
                        'unrealized_pl': float(pos.get('unrealizedPL', 0))
                    })
            
            return positions
            
        except Exception as e:
            print(f"Error getting open positions: {e}")
            return []
    
    def place_market_order(self, instrument, units, stop_loss=None, take_profit=None):
        """
        Place market order.
        
        Parameters:
        -----------
        instrument : str
            Currency pair (e.g., 'USD_JPY')
        units : float
            Position size (positive=buy, negative=sell)
            For standard lot (100k), use 100000 or -100000
        stop_loss : float, optional
            Stop loss price
        take_profit : float, optional
            Take profit price
            
        Returns:
        --------
        dict
            Order response
        """
        url = f"{self.api_url}/v3/accounts/{self.account_id}/orders"
        
        order_data = {
            'order': {
                'type': 'MARKET',
                'instrument': instrument,
                'units': str(int(units)),
                'timeInForce': 'FOK',  # Fill or Kill
                'positionFill': 'DEFAULT'
            }
        }
        
        # Add stop loss
        if stop_loss is not None:
            order_data['order']['stopLossOnFill'] = {
                'price': str(round(stop_loss, 5))
            }
        
        # Add take profit
        if take_profit is not None:
            order_data['order']['takeProfitOnFill'] = {
                'price': str(round(take_profit, 5))
            }
        
        try:
            response = requests.post(url, headers=self.headers, json=order_data)
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Order placed: {units} units of {instrument}")
            if stop_loss:
                print(f"   SL: {stop_loss}")
            if take_profit:
                print(f"   TP: {take_profit}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            if hasattr(e, 'response'):
                print(f"   Response: {e.response.text}")
            return None
    
    def close_position(self, instrument, units='ALL'):
        """
        Close position.
        
        Parameters:
        -----------
        instrument : str
            Currency pair
        units : str or float
            'ALL' to close entire position, or specific units
            
        Returns:
        --------
        dict
            Close response
        """
        # Get current position
        positions = self.get_open_positions()
        position = next((p for p in positions if p['instrument'] == instrument), None)
        
        if not position:
            print(f"No open position for {instrument}")
            return None
        
        # Calculate units to close (opposite sign)
        if units == 'ALL':
            close_units = -position['units']
        else:
            close_units = -abs(float(units)) if position['units'] > 0 else abs(float(units))
        
        # Place market order to close
        return self.place_market_order(instrument, close_units)
    
    def get_trade_history(self, count=50):
        """Get recent trade history."""
        url = f"{self.api_url}/v3/accounts/{self.account_id}/trades"
        params = {'count': count, 'state': 'CLOSED'}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            trades = []
            for trade in data.get('trades', []):
                trades.append({
                    'id': trade.get('id'),
                    'instrument': trade.get('instrument'),
                    'units': float(trade.get('initialUnits', 0)),
                    'price': float(trade.get('price', 0)),
                    'pl': float(trade.get('realizedPL', 0)),
                    'time': pd.to_datetime(trade.get('openTime'))
                })
            
            return trades
            
        except Exception as e:
            print(f"Error getting trade history: {e}")
            return []


def test_connection():
    """Test OANDA API connection."""
    print("=" * 80)
    print("TESTING OANDA API CONNECTION")
    print("=" * 80)
    
    client = OandaClient(environment='practice')
    
    # Test 1: Account info
    print("\n1. Getting account info...")
    account_info = client.get_account_info()
    if account_info:
        print("   ✅ Account info retrieved")
    else:
        print("   ❌ Failed to get account info")
        return False
    
    # Test 2: Account summary
    print("\n2. Getting account summary...")
    summary = client.get_account_summary()
    if summary:
        print(f"   ✅ Balance: ${summary['balance']:,.2f}")
        print(f"   ✅ Equity: ${summary['equity']:,.2f}")
        print(f"   ✅ Open positions: {summary['open_positions']}")
    else:
        print("   ❌ Failed to get account summary")
        return False
    
    # Test 3: Get candles
    print("\n3. Getting USD_JPY candles...")
    df = client.get_candles('USD_JPY', granularity='M15', count=100)
    if df is not None and len(df) > 0:
        print(f"   ✅ Retrieved {len(df)} candles")
        print(f"   Latest close: {df.iloc[-1]['close']:.3f}")
    else:
        print("   ❌ Failed to get candles")
        return False
    
    # Test 4: Current price
    print("\n4. Getting current USD_JPY price...")
    price = client.get_current_price('USD_JPY')
    if price:
        print(f"   ✅ Bid: {price['bid']:.3f}")
        print(f"   ✅ Ask: {price['ask']:.3f}")
        print(f"   ✅ Spread: {price['spread']:.5f} ({price['spread']*100:.2f} pips)")
    else:
        print("   ❌ Failed to get current price")
        return False
    
    # Test 5: Open positions
    print("\n5. Checking open positions...")
    positions = client.get_open_positions()
    print(f"   ✅ Open positions: {len(positions)}")
    for pos in positions:
        print(f"      {pos['instrument']}: {pos['units']} units, P&L: ${pos['unrealized_pl']:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - OANDA CONNECTION READY")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    test_connection()
