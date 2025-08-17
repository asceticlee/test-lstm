#!/usr/bin/env python3
"""
Trading Performance Analysis Script

This script evaluates trading performance based on prediction thresholds and calculates
various performance metrics including Sharpe ratio, maximum drawdown, win rate, etc.

The script supports both upside and downside trading strategies:
- Upside: Trade when predicted > positive threshold, gain if actual > 0
- Downside: Trade when predicted < negative threshold, gain if actual < 0

Trading hours are restricted to 10:36 AM to 12:00 PM (38160000 to 43200000 ms).
Transaction fees are specified in dollars (e.g., 0.02 = $0.02 per trade).

Usage:
    from trading_performance import TradingPerformanceAnalyzer
    
    analyzer = TradingPerformanceAnalyzer(transaction_fee=0.02)  # $0.02 per trade
    results = analyzer.evaluate_performance(thresholds, trading_days, trading_ms, actual, predicted)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingPerformanceAnalyzer:
    """
    Analyze trading performance based on prediction thresholds
    """
    
    def __init__(self, transaction_fee=0.02):
        """
        Initialize the trading performance analyzer
        
        Args:
            transaction_fee: Total transaction fee for opening and closing position in dollars (default: $0.02)
        """
        self.transaction_fee = transaction_fee
        self.trading_start_ms = 38160000  # 10:36 AM
        self.trading_end_ms = 43200000    # 12:00 PM
        
        print(f"Trading Performance Analyzer initialized")
        print(f"Transaction fee: ${transaction_fee:.4f} per trade")
        print(f"Trading hours: {self.ms_to_time(self.trading_start_ms)} to {self.ms_to_time(self.trading_end_ms)}")
    
    def ms_to_time(self, ms):
        """Convert milliseconds of day to readable time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{hours:02d}:{minutes:02d}"
    
    def filter_trading_hours(self, trading_days, trading_ms, actual, predicted):
        """
        Filter data to only include trading hours (10:36 AM to 12:00 PM)
        
        Args:
            trading_days: Array of trading days
            trading_ms: Array of trading milliseconds of day
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            tuple: Filtered arrays (trading_days, trading_ms, actual, predicted)
        """
        # Convert to numpy arrays if not already
        trading_days = np.array(trading_days)
        trading_ms = np.array(trading_ms)
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Filter for trading hours
        mask = (trading_ms >= self.trading_start_ms) & (trading_ms <= self.trading_end_ms)
        
        return (
            trading_days[mask],
            trading_ms[mask],
            actual[mask],
            predicted[mask]
        )
    
    def filter_trading_hours_with_thresholds(self, trading_days, trading_ms, actual, predicted, thresholds):
        """
        Filter data to only include trading hours (10:36 AM to 12:00 PM) with thresholds
        
        Args:
            trading_days: Array of trading days
            trading_ms: Array of trading milliseconds of day
            actual: Array of actual values
            predicted: Array of predicted values
            thresholds: Array of threshold values
            
        Returns:
            tuple: Filtered arrays (trading_days, trading_ms, actual, predicted, thresholds)
        """
        # Convert to numpy arrays if not already
        trading_days = np.array(trading_days)
        trading_ms = np.array(trading_ms)
        actual = np.array(actual)
        predicted = np.array(predicted)
        thresholds = np.array(thresholds)
        
        # Filter for trading hours
        mask = (trading_ms >= self.trading_start_ms) & (trading_ms <= self.trading_end_ms)
        
        return (
            trading_days[mask],
            trading_ms[mask],
            actual[mask],
            predicted[mask],
            thresholds[mask]
        )
    
    def calculate_trades(self, threshold, actual, predicted):
        """
        Calculate trades based on threshold and return P&L
        
        Args:
            threshold: Trading threshold (positive for upside, negative for downside)
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            dict: Trade statistics including P&L, trade signals, etc.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        if threshold > 0:
            # Upside trading: trade when predicted > threshold (exclusive)
            trade_signals = predicted > threshold
            # Gain when actual > 0, loss when actual <= 0
            raw_pnl = np.where(trade_signals, 
                              np.where(actual > 0, actual, actual),  # actual value (positive gain, negative loss)
                              0)  # No trade
        elif threshold < 0:
            # Downside trading: trade when predicted < threshold (exclusive)
            trade_signals = predicted < threshold
            # Gain when actual < 0 (take absolute value), loss when actual >= 0
            raw_pnl = np.where(trade_signals,
                              np.where(actual < 0, -actual, actual),  # -actual for gain, actual for loss
                              0)  # No trade
        else:
            # threshold == 0, no trading
            trade_signals = np.zeros_like(predicted, dtype=bool)
            raw_pnl = np.zeros_like(actual)
        
        # Apply transaction fees to trades
        pnl_after_fees = np.where(trade_signals, raw_pnl - self.transaction_fee, 0)
        
        # Calculate statistics
        num_trades = np.sum(trade_signals)
        winning_trades = np.sum((trade_signals) & (raw_pnl > 0))
        losing_trades = np.sum((trade_signals) & (raw_pnl <= 0))
        
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # Average gains and losses (before fees)
        winning_pnl = raw_pnl[(trade_signals) & (raw_pnl > 0)]
        losing_pnl = raw_pnl[(trade_signals) & (raw_pnl <= 0)]
        
        avg_win = np.mean(winning_pnl) if len(winning_pnl) > 0 else 0
        avg_loss = np.mean(losing_pnl) if len(losing_pnl) > 0 else 0
        
        return {
            'threshold': threshold,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl_before_fees': np.sum(raw_pnl),
            'total_pnl_after_fees': np.sum(pnl_after_fees),
            'total_fees': np.sum(trade_signals) * self.transaction_fee,
            'trade_signals': trade_signals,
            'raw_pnl': raw_pnl,
            'pnl_after_fees': pnl_after_fees
        }
    
    def calculate_trades_with_thresholds(self, actual, predicted, thresholds):
        """
        Calculate trades based on per-row thresholds and return P&L
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            thresholds: Array of threshold values (one per row)
            
        Returns:
            dict: Trade statistics including P&L, trade signals, etc.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        thresholds = np.array(thresholds)
        
        # Initialize arrays
        trade_signals = np.zeros_like(predicted, dtype=bool)
        raw_pnl = np.zeros_like(actual)
        
        # Handle upside trades (positive thresholds)
        upside_mask = thresholds > 0
        upside_trade_signals = (predicted > thresholds) & upside_mask
        trade_signals |= upside_trade_signals
        # Gain when actual > 0, loss when actual <= 0
        raw_pnl = np.where(upside_trade_signals, actual, raw_pnl)
        
        # Handle downside trades (negative thresholds)
        downside_mask = thresholds < 0
        downside_trade_signals = (predicted < thresholds) & downside_mask
        trade_signals |= downside_trade_signals
        # For short positions: P&L = -actual (profit when actual < 0, loss when actual >= 0)
        raw_pnl = np.where(downside_trade_signals, -actual, raw_pnl)
        
        # Note: threshold == 0 results in no trades (trade_signals and raw_pnl remain zeros)
        
        # Apply transaction fees to trades
        pnl_after_fees = np.where(trade_signals, raw_pnl - self.transaction_fee, 0)
        
        # Calculate statistics - Fixed bug: ensure counts are consistent
        num_trades = np.sum(trade_signals)
        
        # For trades that actually happened, count wins and losses
        trade_pnl = raw_pnl[trade_signals]  # Only P&L values where trades occurred
        winning_trades = np.sum(trade_pnl > 0)
        losing_trades = np.sum(trade_pnl <= 0)
        
        # Verify counts add up correctly
        assert winning_trades + losing_trades == num_trades, f"Count mismatch: {winning_trades} + {losing_trades} != {num_trades}"
        
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # Average gains and losses (before fees) - only for actual trades
        winning_pnl = trade_pnl[trade_pnl > 0]
        losing_pnl = trade_pnl[trade_pnl <= 0]
        
        avg_win = np.mean(winning_pnl) if len(winning_pnl) > 0 else 0
        avg_loss = np.mean(losing_pnl) if len(losing_pnl) > 0 else 0
        
        return {
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl_before_fees': np.sum(raw_pnl),
            'total_pnl_after_fees': np.sum(pnl_after_fees),
            'total_fees': np.sum(trade_signals) * self.transaction_fee
        }
    
    def calculate_performance_metrics(self, pnl_series, trading_days=None):
        """
        Calculate comprehensive performance metrics
        
        Args:
            pnl_series: Series of P&L values
            trading_days: Optional array of trading days for time-based metrics
            
        Returns:
            dict: Performance metrics
        """
        pnl_series = np.array(pnl_series)
        
        # Remove zeros for certain calculations
        non_zero_pnl = pnl_series[pnl_series != 0]
        
        # Basic metrics
        total_pnl = np.sum(pnl_series)
        num_trades = np.sum(pnl_series != 0)
        
        if num_trades == 0:
            return {
                'total_pnl': 0,
                'num_trades': 0,
                'avg_pnl_per_trade': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'volatility': 0,
                'calmar_ratio': 0
            }
        
        # Average P&L per trade
        avg_pnl_per_trade = total_pnl / num_trades
        
        # Volatility (standard deviation of non-zero P&L)
        volatility = np.std(non_zero_pnl) if len(non_zero_pnl) > 1 else 0
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = avg_pnl_per_trade / volatility if volatility > 0 else 0
        
        # Cumulative P&L for drawdown calculation
        cumulative_pnl = np.cumsum(pnl_series)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        max_drawdown = np.max(drawdown)
        
        # Maximum drawdown percentage
        max_drawdown_pct = (max_drawdown / np.max(peak)) * 100 if np.max(peak) > 0 else 0
        
        # Profit factor (gross profit / gross loss)
        gross_profit = np.sum(non_zero_pnl[non_zero_pnl > 0])
        gross_loss = abs(np.sum(non_zero_pnl[non_zero_pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = total_pnl / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def evaluate_performance(self, trading_days, trading_ms, actual, predicted, thresholds):
        """
        Evaluate trading performance using per-row thresholds
        
        Args:
            trading_days: Array of trading days
            trading_ms: Array of trading milliseconds of day
            actual: Array of actual values
            predicted: Array of predicted values
            thresholds: Array of threshold values (one per row)
            
        Returns:
            dict: Performance results for the given threshold strategy
        """
        print(f"Evaluating performance with per-row thresholds...")
        
        # Filter for trading hours
        filtered_days, filtered_ms, filtered_actual, filtered_predicted, filtered_thresholds = self.filter_trading_hours_with_thresholds(
            trading_days, trading_ms, actual, predicted, thresholds
        )
        
        original_count = len(trading_days)
        filtered_count = len(filtered_days)
        print(f"Filtered to trading hours: {filtered_count:,} records ({original_count - filtered_count:,} excluded)")
        
        if filtered_count == 0:
            print("Warning: No data in trading hours")
            return {}
        
        # Calculate trades using per-row thresholds
        trade_stats = self.calculate_trades_with_thresholds(filtered_actual, filtered_predicted, filtered_thresholds)
        
        # Calculate performance metrics from raw data (we'll create the pnl series)
        # For performance metrics, we need the P&L series, not just totals
        # We'll regenerate the needed arrays for performance calculation
        actual_array = np.array(filtered_actual)
        predicted_array = np.array(filtered_predicted)
        thresholds_array = np.array(filtered_thresholds)
        
        # Recreate trade signals and P&L arrays for performance metrics
        trade_signals = np.zeros_like(predicted_array, dtype=bool)
        raw_pnl = np.zeros_like(actual_array)
        
        # Handle upside trades
        upside_mask = thresholds_array > 0
        upside_trade_signals = (predicted_array > thresholds_array) & upside_mask
        trade_signals |= upside_trade_signals
        raw_pnl = np.where(upside_trade_signals, actual_array, raw_pnl)
        
        # Handle downside trades  
        downside_mask = thresholds_array < 0
        downside_trade_signals = (predicted_array < thresholds_array) & downside_mask
        trade_signals |= downside_trade_signals
        raw_pnl = np.where(downside_trade_signals, -actual_array, raw_pnl)
        
        # Note: threshold == 0 results in no trades (trade_signals and raw_pnl remain zeros)
        
        # Apply transaction fees
        pnl_after_fees = np.where(trade_signals, raw_pnl - self.transaction_fee, 0)
        
        # Calculate performance metrics using the P&L series
        perf_metrics = self.calculate_performance_metrics(pnl_after_fees, filtered_days)
        
        # Get unique threshold value (assuming all rows have same threshold for now)
        unique_thresholds = np.unique(filtered_thresholds)
        if len(unique_thresholds) == 1:
            threshold_value = unique_thresholds[0]
            trade_type = 'upside' if threshold_value > 0 else 'downside' if threshold_value < 0 else 'none'
        else:
            threshold_value = f"mixed ({len(unique_thresholds)} values)"
            trade_type = 'mixed'
        
        # Combine results - ensure performance metrics use correct totals from trade_stats
        result = {
            'threshold': threshold_value,
            'trade_type': trade_type,
            **trade_stats,
            **perf_metrics
        }
        
        # Recalculate avg_pnl_per_trade using the correct total_pnl_after_fees
        if result['num_trades'] > 0:
            result['avg_pnl_per_trade'] = result['total_pnl_after_fees'] / result['num_trades']
        else:
            result['avg_pnl_per_trade'] = 0
        
        print(f"Performance evaluation completed")
        return result
    
    def print_performance_summary(self, result):
        """
        Print a summary of the performance result
        
        Args:
            result: Dictionary result from evaluate_performance
        """
        if not result:
            print("No results to display")
            return
        
        print(f"\n" + "="*80)
        print(f"TRADING PERFORMANCE SUMMARY")
        print(f"="*80)
        print(f"Transaction fee: ${self.transaction_fee:.4f} per trade")
        print(f"Trading hours: {self.ms_to_time(self.trading_start_ms)} to {self.ms_to_time(self.trading_end_ms)}")
        
        # Performance summary
        print(f"\nPerformance Results:")
        print("-" * 60)
        print(f"  Threshold: {result['threshold']}")
        print(f"  Trade Type: {result['trade_type']}")
        print(f"  Number of Trades: {result['num_trades']:,}")
        print(f"  Win Rate: {result['win_rate']:.1%}")
        print(f"  Total P&L (After Fees): ${result['total_pnl_after_fees']:.4f}")
        print(f"  Total P&L (Before Fees): ${result['total_pnl_before_fees']:.4f}")
        print(f"  Total Fees: ${result['total_fees']:.4f}")
        print(f"  Average Win: ${result['avg_win']:.4f}")
        print(f"  Average Loss: ${result['avg_loss']:.4f}")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print(f"  Maximum Drawdown: ${result['max_drawdown']:.4f} ({result['max_drawdown_pct']:.2f}%)")
        print(f"  Profit Factor: {result['profit_factor']:.4f}")
        print(f"  Volatility: {result['volatility']:.4f}")
        print(f"  Calmar Ratio: {result['calmar_ratio']:.4f}")
        
        # Additional statistics
        if result['num_trades'] > 0:
            print(f"\nTrade Breakdown:")
            print("-" * 40)
            print(f"  Winning Trades: {result['winning_trades']:,}")
            print(f"  Losing Trades: {result['losing_trades']:,}")
            print(f"  Gross Profit: ${result['gross_profit']:.4f}")
            print(f"  Gross Loss: ${result['gross_loss']:.4f}")
        else:
            print(f"\nNo trades executed with current threshold")

def create_sample_data(n_samples=1000):
    """
    Create sample trading data for testing
    
    Args:
        n_samples: Number of sample records to create
        
    Returns:
        tuple: (trading_days, trading_ms, actual, predicted)
    """
    np.random.seed(42)
    
    trading_days = []
    trading_ms = []
    actual = []
    predicted = []
    
    base_day = 20200102
    
    for i in range(n_samples):
        # Random trading day and time within trading hours
        day_offset = i // 100  # Roughly 100 records per day
        current_day = base_day + day_offset
        
        # Random time within trading hours (10:36 to 12:00)
        time_range = 43200000 - 38160000  # Trading window in ms
        random_time = 38160000 + (i % 100) * (time_range // 100)
        
        # Generate correlated actual and predicted values
        true_signal = np.random.normal(0, 0.1)  # True underlying signal
        noise = np.random.normal(0, 0.05)       # Prediction noise
        
        actual_val = true_signal + np.random.normal(0, 0.02)
        predicted_val = true_signal + noise
        
        trading_days.append(current_day)
        trading_ms.append(random_time)
        actual.append(actual_val)
        predicted.append(predicted_val)
    
    return trading_days, trading_ms, actual, predicted

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TradingPerformanceAnalyzer(transaction_fee=0.02)
    
    # Create sample data
    print("\nGenerating sample data for testing...")
    trading_days, trading_ms, actual, predicted = create_sample_data(1000)
    
    # Test a single threshold
    test_threshold = 0.05
    
    # Create threshold array (all rows have same threshold for now)
    thresholds = np.full(len(trading_days), test_threshold)
    
    # Evaluate performance
    result = analyzer.evaluate_performance(trading_days, trading_ms, actual, predicted, thresholds)
    
    # Print summary
    analyzer.print_performance_summary(result)
