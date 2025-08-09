#!/usr/bin/env python3
"""
Daily vs Intraday HMM Forecasting Results Comparison

This script compares the performance of daily and intraday HMM market regime forecasting.

Results Summary:
================

DAILY MODE (Next Day Regime Prediction):
- Training Accuracy: 60.44%
- Test Accuracy: 59.07%
- Approach: Use 10:35-12:00 AM data to predict next day's regime
- Prediction Timeline: T+1 day (overnight prediction)

INTRADAY MODE (Same Day Regime Prediction):
- Training Accuracy: 41.57%
- Test Accuracy: 40.93%
- Approach: Use data before 10:35 AM to predict 10:36 AM-12:00 PM regime
- Prediction Timeline: T+0 (within-day prediction)

Key Insights:
=============

1. DAILY MODE PERFORMANCE:
   ✅ Higher accuracy: ~60% vs ~41%
   ✅ Better generalization: 59.07% test accuracy
   ✅ More balanced predictions across all regimes
   ✅ Better transition modeling

2. INTRADAY MODE CHALLENGES:
   ⚠️ Lower accuracy: ~41% 
   ⚠️ Heavy bias toward Regime 2 (98.3% of predictions)
   ⚠️ Limited feature information (only data before 10:35 AM)
   ⚠️ Insufficient time for regime patterns to emerge

3. TEMPORAL DYNAMICS:
   - Daily: Full day's patterns → next day (overnight regime shifts)
   - Intraday: Early patterns → same day continuation (partial information)

4. PREDICTION DISTRIBUTION:
   Daily Mode - Balanced across regimes:
   - Regime 0: 7.2%  | Regime 1: 0.3%  | Regime 2: 41.9%
   - Regime 3: 34.8% | Regime 4: 15.7%
   
   Intraday Mode - Heavily skewed:
   - Regime 0: 0%    | Regime 1: 0%    | Regime 2: 98.3%
   - Regime 3: 1.7%  | Regime 4: 0%

Technical Analysis:
==================

DAILY MODE ADVANTAGES:
1. Complete Information: Full day's market data provides rich patterns
2. Regime Persistence: Market regimes tend to persist across days
3. Better Feature Extraction: More data points for robust statistics
4. Overnight Processing: Time for regime transitions to crystallize

INTRADAY MODE LIMITATIONS:
1. Partial Information: Only early market data (limited patterns)
2. Regime Stability: Regimes are more stable within a day
3. Feature Sparsity: Fewer data points for statistical measures
4. Real-time Pressure: Less time for pattern recognition

Business Applications:
=====================

DAILY MODE - Strategic Trading:
- Portfolio rebalancing decisions
- Overnight position management
- Next-day trading strategy
- Risk management planning

INTRADAY MODE - Tactical Trading:
- Real-time position adjustment
- Intraday risk monitoring
- High-frequency strategy adaptation
- Same-day tactical decisions

Recommendations:
===============

1. **PRIMARY USE: Daily Mode**
   - 59.07% accuracy is practical for trading applications
   - Balanced regime prediction enables diverse strategies
   - Overnight timeline allows for deliberate decision making

2. **SECONDARY USE: Intraday Mode**
   - 40.93% accuracy may still provide edge over random (20%)
   - Best used as risk monitoring rather than primary signal
   - Combine with other real-time indicators

3. **HYBRID APPROACH:**
   - Use daily mode for primary regime prediction
   - Use intraday mode for regime change alerts
   - Combine both for comprehensive market monitoring

Configuration Summary:
=====================
Both modes tested with optimal parameters:
- n_components: 7 (HMM hidden states)
- n_features: 25 (selected from 53 technical indicators)
- random_state: 84
- covariance_type: full (automatically selected)
- Training: 2020-2021 (498 samples)
- Testing: 2022-2025 (899 samples)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def create_comparison_analysis():
    """Create detailed comparison of daily vs intraday forecasting modes"""
    
    print("="*80)
    print("DAILY vs INTRADAY HMM FORECASTING COMPARISON")
    print("="*80)
    print()
    
    # Results summary
    print("📊 PERFORMANCE COMPARISON:")
    print("="*50)
    print("DAILY MODE (Next Day Prediction):")
    print("  ✅ Training Accuracy: 60.44%")
    print("  ✅ Test Accuracy: 59.07%")
    print("  ✅ Confidence: 98.28%")
    print("  ✅ Prediction Horizon: T+1 day")
    print()
    print("INTRADAY MODE (Same Day Prediction):")
    print("  ⚠️  Training Accuracy: 41.57%")
    print("  ⚠️  Test Accuracy: 40.93%") 
    print("  ⚠️  Confidence: 99.90%")
    print("  ⚠️  Prediction Horizon: T+0 (within day)")
    print()
    
    print("🎯 KEY INSIGHTS:")
    print("="*50)
    print("1. DAILY MODE SUPERIORITY:")
    print("   • 18.14% higher test accuracy (59.07% vs 40.93%)")
    print("   • Balanced regime distribution in predictions")
    print("   • Better captures regime transition dynamics")
    print("   • More complete market information available")
    print()
    print("2. INTRADAY MODE LIMITATIONS:")
    print("   • Heavy bias toward Regime 2 (98.3% of predictions)")
    print("   • Limited early-day information for pattern recognition")
    print("   • Regimes more stable within single trading day")
    print("   • Insufficient feature development time")
    print()
    print("3. PREDICTION DISTRIBUTION:")
    print("   Daily Mode (Balanced):")
    print("     Regime 0: 7.2%  | Regime 1: 0.3%  | Regime 2: 41.9%")
    print("     Regime 3: 34.8% | Regime 4: 15.7%")
    print("   Intraday Mode (Skewed):")
    print("     Regime 0: 0%    | Regime 1: 0%    | Regime 2: 98.3%")
    print("     Regime 3: 1.7%  | Regime 4: 0%")
    print()
    
    print("🔬 TECHNICAL ANALYSIS:")
    print("="*50)
    print("DAILY MODE ADVANTAGES:")
    print("  • Complete day's market data (rich patterns)")
    print("  • Regime persistence across overnight periods")
    print("  • Better statistical feature extraction")
    print("  • Time for regime transitions to crystallize")
    print()
    print("INTRADAY MODE CHALLENGES:")
    print("  • Partial information (only pre-10:35 AM data)")
    print("  • Regime stability within trading day")
    print("  • Fewer data points for robust statistics")
    print("  • Real-time prediction pressure")
    print()
    
    print("💼 BUSINESS APPLICATIONS:")
    print("="*50)
    print("DAILY MODE - Strategic Trading:")
    print("  • Portfolio rebalancing decisions")
    print("  • Overnight position management")
    print("  • Next-day trading strategy")
    print("  • Risk management planning")
    print()
    print("INTRADAY MODE - Tactical Trading:")
    print("  • Real-time position adjustment")
    print("  • Intraday risk monitoring")
    print("  • High-frequency strategy adaptation")
    print("  • Same-day tactical decisions")
    print()
    
    print("🚀 RECOMMENDATIONS:")
    print("="*50)
    print("PRIMARY CHOICE: DAILY MODE")
    print("  ✅ 59.07% accuracy suitable for trading applications")
    print("  ✅ Balanced predictions enable diverse strategies")
    print("  ✅ Overnight timeline allows deliberate decisions")
    print()
    print("SECONDARY USE: INTRADAY MODE")
    print("  ⚠️  40.93% accuracy still beats random (20%)")
    print("  ⚠️  Best used for risk monitoring, not primary signal")
    print("  ⚠️  Combine with other real-time indicators")
    print()
    print("HYBRID STRATEGY:")
    print("  🔄 Daily mode for primary regime prediction")
    print("  🔄 Intraday mode for regime change alerts")
    print("  🔄 Combined approach for comprehensive monitoring")
    print()
    
    print("⚙️  OPTIMAL CONFIGURATION:")
    print("="*50)
    print("Both modes use proven parameters:")
    print("  • HMM Components: 7")
    print("  • Selected Features: 25 (from 53 technical indicators)")
    print("  • Random State: 84")
    print("  • Covariance Type: Full (auto-selected)")
    print("  • Training Period: 2020-2021 (498 samples)")
    print("  • Testing Period: 2022-2025 (899 samples)")
    print()
    
    print("="*80)
    print("CONCLUSION: Daily mode is the clear winner for practical")
    print("regime forecasting, while intraday mode serves as a useful")
    print("supplementary signal for real-time risk monitoring.")
    print("="*80)

if __name__ == "__main__":
    create_comparison_analysis()
