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
    
    def calculate_trades(self, threshold, actual, predicted, side=None):
        """
        Calculate trades based on threshold and return P&L
        
        Args:
            threshold: Trading threshold value
            actual: Array of actual values
            predicted: Array of predicted values
            side: Trade direction ('up' for upside, 'down' for downside). 
                  If None, inferred from threshold sign (positive=up, negative=down)
                  Required when threshold=0 to resolve ambiguity
            
        Returns:
            dict: Trade statistics including P&L, trade signals, etc.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Determine trade direction
        if side is not None:
            # Explicit side specified
            trade_direction = side.lower()
            if trade_direction not in ['up', 'down']:
                raise ValueError("side must be 'up' or 'down'")
        else:
            # Infer from threshold sign
            if threshold > 0:
                trade_direction = 'up'
            elif threshold < 0:
                trade_direction = 'down'
            else:
                # threshold == 0 and no side specified - ambiguous
                raise ValueError("When threshold=0, 'side' parameter must be specified ('up' or 'down')")
        
        # Calculate trade signals based on direction
        if trade_direction == 'up':
            # Upside trading: trade when predicted > threshold (exclusive)
            trade_signals = predicted > threshold
            # Gain when actual > 0, loss when actual <= 0
            raw_pnl = np.where(trade_signals, 
                              np.where(actual > 0, actual, actual),  # actual value (positive gain, negative loss)
                              0)  # No trade
        else:  # trade_direction == 'down'
            # Downside trading: trade when predicted < threshold (exclusive)
            trade_signals = predicted < threshold
            # Gain when actual < 0 (take absolute value), loss when actual >= 0
            raw_pnl = np.where(trade_signals,
                              np.where(actual < 0, -actual, actual),  # -actual for gain, actual for loss
                              0)  # No trade
        
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
    
    def calculate_trades_with_thresholds(self, actual, predicted, thresholds, sides):
        """
        Calculate trades based on per-row thresholds and return P&L
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            thresholds: Array of threshold values (one per row)
            sides: Array of trade directions ('up' or 'down').
                   Required - specifies trade direction for each row
            
        Returns:
            dict: Trade statistics including P&L, trade signals, etc.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        thresholds = np.array(thresholds)
        
        # Validate sides parameter
        if sides is None:
            raise ValueError("The 'sides' parameter is required. Must be an array of 'up' or 'down' values.")
            
        sides = np.array(sides)
        if len(sides) != len(thresholds):
            raise ValueError("sides array must have same length as thresholds array")
        
        # Initialize arrays
        trade_signals = np.zeros_like(predicted, dtype=bool)
        raw_pnl = np.zeros_like(actual)
        
        # Process each row
        for i in range(len(thresholds)):
            threshold = thresholds[i]
            
            # Use the specified side directly
            trade_direction = sides[i].lower()
            if trade_direction not in ['up', 'down']:
                raise ValueError(f"sides[{i}] must be 'up' or 'down', got '{sides[i]}'")
            
            # Calculate trade signal for this row
            if trade_direction == 'up':
                # Upside trading: trade when predicted > threshold
                if predicted[i] > threshold:
                    trade_signals[i] = True
                    raw_pnl[i] = actual[i]  # P&L = actual value
            else:  # trade_direction == 'down'
                # Downside trading: trade when predicted < threshold  
                if predicted[i] < threshold:
                    trade_signals[i] = True
                    raw_pnl[i] = -actual[i]  # P&L = -actual (profit when actual < 0)
        
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
    
    def evaluate_performance(self, trading_days, trading_ms, actual, predicted, thresholds, sides):
        """
        Evaluate trading performance using per-row thresholds
        
        Args:
            trading_days: Array of trading days
            trading_ms: Array of trading milliseconds of day
            actual: Array of actual values
            predicted: Array of predicted values
            thresholds: Array of threshold values (one per row)
            sides: Array of trade directions ('up' or 'down'). 
                   Required - specifies trade direction for each row
            
        Returns:
            dict: Performance results for the given threshold strategy
        """
        print(f"Evaluating performance with per-row thresholds...")
        
        # Validate sides parameter
        if sides is None:
            raise ValueError("The 'sides' parameter is required. Must be an array of 'up' or 'down' values.")
        
        # Filter for trading hours
        filtered_days, filtered_ms, filtered_actual, filtered_predicted, filtered_thresholds = self.filter_trading_hours_with_thresholds(
            trading_days, trading_ms, actual, predicted, thresholds
        )
        # Also filter sides array to match
        sides_array = np.array(sides)
        mask = (np.array(trading_ms) >= self.trading_start_ms) & (np.array(trading_ms) <= self.trading_end_ms)
        filtered_sides = sides_array[mask]
        
        original_count = len(trading_days)
        filtered_count = len(filtered_days)
        print(f"Filtered to trading hours: {filtered_count:,} records ({original_count - filtered_count:,} excluded)")
        
        if filtered_count == 0:
            print("Warning: No data in trading hours")
            return {}
        
        # Calculate trades using per-row thresholds
        trade_stats = self.calculate_trades_with_thresholds(filtered_actual, filtered_predicted, filtered_thresholds, filtered_sides)
        
        # Calculate performance metrics from raw data (we'll create the pnl series)
        # For performance metrics, we need the P&L series, not just totals
        # We'll regenerate the needed arrays for performance calculation
        actual_array = np.array(filtered_actual)
        predicted_array = np.array(filtered_predicted)
        thresholds_array = np.array(filtered_thresholds)
        
        # Recreate trade signals and P&L arrays for performance metrics
        trade_signals = np.zeros_like(predicted_array, dtype=bool)
        raw_pnl = np.zeros_like(actual_array)
        
        # Process each row individually to handle sides correctly
        for i in range(len(thresholds_array)):
            threshold = thresholds_array[i]
            
            # Determine trade direction for this row
            if filtered_sides is not None and filtered_sides[i] is not None:
                trade_direction = filtered_sides[i].lower()
            else:
                # Infer from threshold sign
                if threshold > 0:
                    trade_direction = 'up'
                elif threshold < 0:
                    trade_direction = 'down'
                else:
                    # threshold == 0 and no side specified - skip this row (no trade)
                    continue
            
            # Calculate trade signal for this row
            if trade_direction == 'up':
                # Upside trading: trade when predicted > threshold
                if predicted_array[i] > threshold:
                    trade_signals[i] = True
                    raw_pnl[i] = actual_array[i]  # P&L = actual value
            else:  # trade_direction == 'down'
                # Downside trading: trade when predicted < threshold  
                if predicted_array[i] < threshold:
                    trade_signals[i] = True
                    raw_pnl[i] = -actual_array[i]  # P&L = -actual (profit when actual < 0)
        
        # Apply transaction fees
        pnl_after_fees = np.where(trade_signals, raw_pnl - self.transaction_fee, 0)
        
        # Calculate performance metrics using the P&L series
        perf_metrics = self.calculate_performance_metrics(pnl_after_fees, filtered_days)
        
        # Get unique threshold value (assuming all rows have same threshold for now)
        unique_thresholds = np.unique(filtered_thresholds)
        if len(unique_thresholds) == 1:
            threshold_value = unique_thresholds[0]
            threshold_display = None  # Will be set for zero thresholds
            
            # Determine trade type based on threshold and sides parameter
            if threshold_value > 0:
                trade_type = 'upside'
            elif threshold_value < 0:
                trade_type = 'downside'
            else:  # threshold_value == 0
                if filtered_sides is not None:
                    # For zero threshold, use the sides parameter
                    unique_sides = np.unique(filtered_sides)
                    if len(unique_sides) == 1:
                        side = unique_sides[0]
                        trade_type = 'upside' if side == 'up' else 'downside'
                        # Set display value for zero threshold to distinguish up/down
                        threshold_display = '+0.0' if side == 'up' else '-0.0'
                    else:
                        trade_type = 'mixed'
                else:
                    trade_type = 'none'
        else:
            threshold_value = f"mixed ({len(unique_thresholds)} values)"
            trade_type = 'mixed'
            threshold_display = None
        
        # Combine results - ensure performance metrics use correct totals from trade_stats
        result = {
            'threshold': threshold_value,
            'trade_type': trade_type,
            **trade_stats,
            **perf_metrics
        }
        
        # Add threshold_display if it was set
        if threshold_display is not None:
            result['threshold_display'] = threshold_display
        
        # Recalculate avg_pnl_per_trade using the correct total_pnl_after_fees
        if result['num_trades'] > 0:
            result['avg_pnl_per_trade'] = result['total_pnl_after_fees'] / result['num_trades']
        else:
            result['avg_pnl_per_trade'] = 0
        
        # Add composite score
        result['composite_score'] = self.calculate_composite_score(result)
        
        print(f"Performance evaluation completed")
        return result
    
    def calculate_composite_score(self, result, weights=None):
        """
        Calculate a composite performance score from multiple metrics
        
        This function combines key performance metrics into a single score for easy comparison.
        The score is normalized and weighted to balance profitability, risk, and consistency.
        
        Args:
            result: Dictionary result from evaluate_performance
            weights: Dictionary of weights for each metric. If None, uses default weights.
                    Keys: 'profitability', 'risk_adjusted', 'consistency', 'efficiency'
                    
        Returns:
            float: Composite score (higher is better, typically 0-100 scale)
        """
        if not result or result['num_trades'] == 0:
            return 0.0
        
        # Default weights - can be customized based on trading strategy preferences
        if weights is None:
            weights = {
                'profitability': 0.35,  # Total P&L and profit factor
                'risk_adjusted': 0.30,  # Sharpe ratio and max drawdown
                'consistency': 0.25,    # Win rate and volatility
                'efficiency': 0.10      # Calmar ratio and trade frequency
            }
        
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            for key in weights:
                weights[key] /= total_weight
        
        # Extract key metrics
        total_pnl = result['total_pnl_after_fees']
        sharpe_ratio = result['sharpe_ratio']
        max_drawdown = result['max_drawdown']
        win_rate = result['win_rate']
        profit_factor = result['profit_factor']
        volatility = result['volatility']
        calmar_ratio = result['calmar_ratio']
        num_trades = result['num_trades']
        
        # Component 1: Profitability (0-100 scale)
        # Positive P&L gets higher score, scaled by magnitude
        pnl_score = max(0, min(100, 50 + total_pnl * 10))  # Center at 50, +/- based on P&L
        
        # Profit factor contribution (>1.0 is good, 2.0+ is excellent)
        pf_score = min(100, max(0, (profit_factor - 0.5) * 50))
        profitability_score = (pnl_score * 0.7) + (pf_score * 0.3)
        
        # Component 2: Risk-Adjusted Returns (0-100 scale)
        # Sharpe ratio contribution (>0.5 is good, >1.0 is excellent)
        sharpe_score = min(100, max(0, (sharpe_ratio + 0.5) * 50))
        
        # Max drawdown penalty (lower drawdown is better)
        if max_drawdown > 0:
            # Penalize based on drawdown as percentage of total P&L
            if total_pnl > 0:
                drawdown_ratio = max_drawdown / max(total_pnl, max_drawdown)
            else:
                drawdown_ratio = 1.0
            drawdown_score = max(0, 100 - drawdown_ratio * 100)
        else:
            drawdown_score = 100
        
        risk_adjusted_score = (sharpe_score * 0.6) + (drawdown_score * 0.4)
        
        # Component 3: Consistency (0-100 scale)
        # Win rate contribution (50% is neutral, 60%+ is good)
        win_rate_score = max(0, min(100, win_rate * 100))
        
        # Volatility penalty (lower volatility is better for consistency)
        volatility_score = max(0, min(100, 100 - volatility * 50))
        consistency_score = (win_rate_score * 0.7) + (volatility_score * 0.3)
        
        # Component 4: Efficiency (0-100 scale)
        # Calmar ratio contribution
        calmar_score = min(100, max(0, (calmar_ratio + 1) * 25))
        
        # Trade frequency consideration (too few trades may not be statistically significant)
        if num_trades >= 100:
            frequency_score = 100
        elif num_trades >= 50:
            frequency_score = 80
        elif num_trades >= 20:
            frequency_score = 60
        elif num_trades >= 10:
            frequency_score = 40
        else:
            frequency_score = 20
        
        efficiency_score = (calmar_score * 0.7) + (frequency_score * 0.3)
        
        # Calculate weighted composite score
        composite_score = (
            profitability_score * weights['profitability'] +
            risk_adjusted_score * weights['risk_adjusted'] +
            consistency_score * weights['consistency'] +
            efficiency_score * weights['efficiency']
        )
        
        # Ensure score is within 0-100 range
        composite_score = max(0, min(100, composite_score))
        
        return round(composite_score, 2)
    
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
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(result)
        
        # Performance summary
        print(f"\nPerformance Results:")
        print("-" * 60)
        # Show threshold display if available, otherwise show threshold
        threshold_display = result.get('threshold_display', result['threshold'])
        print(f"  Threshold: {threshold_display}")
        print(f"  Trade Type: {result['trade_type']}")
        print(f"  Composite Score: {composite_score:.2f}/100")
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
