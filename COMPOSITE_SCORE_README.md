#!/usr/bin/env python3
"""
Composite Score Function Documentation

The composite score function in trading_performance.py provides a single metric (0-100) 
to evaluate and compare trading strategies across multiple performance dimensions.

COMPOSITE SCORE COMPONENTS:
==========================

1. Profitability (35% weight):
   - Total P&L after fees (primary factor)
   - Profit factor (secondary factor)
   
2. Risk-Adjusted Returns (30% weight):
   - Sharpe ratio (reward-to-risk ratio)
   - Maximum drawdown penalty
   
3. Consistency (25% weight):
   - Win rate (percentage of profitable trades)
   - Volatility penalty (lower volatility = more consistent)
   
4. Efficiency (10% weight):
   - Calmar ratio (return vs max drawdown)
   - Trade frequency consideration (statistical significance)

SCORE INTERPRETATION:
====================
- 80-100: Excellent strategy (high profit, low risk, consistent)
- 60-79:  Good strategy (profitable with acceptable risk)
- 40-59:  Moderate strategy (mixed performance)
- 20-39:  Poor strategy (marginal or unprofitable)
- 0-19:   Very poor strategy (significant losses/high risk)

USAGE:
======
The composite score is automatically calculated and included in:
- Performance summaries (printed output)
- CSV export files (composite_score column)
- Top performers analysis (ranked by composite score)

CUSTOMIZATION:
==============
The scoring weights can be customized by passing a weights dictionary:

weights = {
    'profitability': 0.4,   # Increase focus on profits
    'risk_adjusted': 0.3,   # Standard risk focus
    'consistency': 0.2,     # Reduce consistency importance
    'efficiency': 0.1       # Standard efficiency focus
}

score = analyzer.calculate_composite_score(result, weights)
"""
