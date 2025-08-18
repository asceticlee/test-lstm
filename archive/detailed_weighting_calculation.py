#!/usr/bin/env python3
"""
Detailed Weighting Calculation Example
Shows exact calculation for specific trading day, regime, and models with column names and weights
"""

def create_detailed_calculation_example():
    print("=" * 100)
    print("DETAILED WEIGHTING CALCULATION EXAMPLE")
    print("=" * 100)
    
    # Trading Day and Scenario
    trading_day = "20200102"
    target_regime = 1  # Regime 1 
    
    print(f"\n📅 TRADING DAY: {trading_day}")
    print(f"🎯 TARGET REGIME: {target_regime}")
    print(f"🏷️  WEIGHTING STRATEGIES: 2 (Accuracy-focused vs PnL-focused)")
    print(f"🔢 MODELS TO COMPARE: 2 (model_00001 vs model_00002)")
    
    # Define weighting arrays
    accuracy_weights = [2.0, 0.5, 2.0, 0.5] * 218  # 872 total fields, pattern repeats every 4
    pnl_weights = [0.5, 2.0, 0.5, 2.0] * 218      # 872 total fields, pattern repeats every 4
    
    print(f"\n📊 WEIGHTING ARRAYS:")
    print(f"   • Accuracy-focused: [2.0, 0.5, 2.0, 0.5] repeating (emphasizes even-indexed fields)")
    print(f"   • PnL-focused:      [0.5, 2.0, 0.5, 2.0] repeating (emphasizes odd-indexed fields)")
    print(f"   • Total fields per model: 872")
    print(f"   • Weight ratio: 4:1 (high:low)")
    
    # Sample data from actual files for model_00001
    print(f"\n" + "="*100)
    print("MODEL 00001 DATA EXTRACTION")
    print("="*100)
    
    # Daily performance data (from model_00001_daily_performance.csv)
    print(f"\n📁 SOURCE: model_performance/model_daily_performance/model_00001_daily_performance.csv")
    print(f"📅 ROW: TradingDay={trading_day}")
    
    daily_data = {
        'daily_up_acc_thr_0.0': 0.6333333333333333,
        'daily_up_num_thr_0.0': 19,
        'daily_up_den_thr_0.0': 30,
        'daily_up_pnl_thr_0.0': 2.0200000000000005,
        'daily_up_acc_thr_0.1': 0.5652173913043478,
        'daily_up_num_thr_0.1': 13,
        'daily_up_den_thr_0.1': 23,
        'daily_up_pnl_thr_0.1': 0.7699999999999998,
        'daily_down_acc_thr_0.0': 0.44642857142857145,
        'daily_down_num_thr_0.0': 25,
        'daily_down_den_thr_0.0': 56,
        'daily_down_pnl_thr_0.0': -0.09000000000000002
    }
    
    print(f"\n🔍 DAILY PERFORMANCE SAMPLES (first 12 of 72 daily fields):")
    for i, (field, value) in enumerate(list(daily_data.items())[:12]):
        weight_acc = accuracy_weights[i]
        weight_pnl = pnl_weights[i]
        weighted_acc = value * weight_acc
        weighted_pnl = value * weight_pnl
        print(f"   [{i:2d}] {field:25s} = {value:12.6f} | Acc_Weight={weight_acc:3.1f} → {weighted_acc:12.6f} | PnL_Weight={weight_pnl:3.1f} → {weighted_pnl:12.6f}")
    
    # Regime performance data (from model_00001_regime_performance.csv)
    print(f"\n📁 SOURCE: model_performance/model_regime_performance/model_00001_regime_performance.csv")
    print(f"📅 ROW: TradingDay={trading_day}, Regime={target_regime}")
    
    regime_data = {
        '1day_up_acc_thr_0.0': 0.6333333333333333,
        '1day_up_num_thr_0.0': 19,
        '1day_up_den_thr_0.0': 30,
        '1day_up_pnl_thr_0.0': 2.0200000000000005,
        '1day_up_acc_thr_0.1': 0.5652173913043478,
        '1day_up_num_thr_0.1': 13,
        '1day_up_den_thr_0.1': 23,
        '1day_up_pnl_thr_0.1': 0.7699999999999998,
        '1day_down_acc_thr_0.0': 0.44642857142857145,
        '1day_down_num_thr_0.0': 25,
        '1day_down_den_thr_0.0': 56,
        '1day_down_pnl_thr_0.0': -0.09000000000000002
    }
    
    print(f"\n🔍 REGIME PERFORMANCE SAMPLES (first 12 of 800 regime fields):")
    regime_offset = 72  # Daily fields come first
    for i, (field, value) in enumerate(list(regime_data.items())[:12]):
        actual_index = regime_offset + i
        weight_acc = accuracy_weights[actual_index]
        weight_pnl = pnl_weights[actual_index]
        weighted_acc = value * weight_acc
        weighted_pnl = value * weight_pnl
        print(f"   [{actual_index:2d}] {field:25s} = {value:12.6f} | Acc_Weight={weight_acc:3.1f} → {weighted_acc:12.6f} | PnL_Weight={weight_pnl:3.1f} → {weighted_pnl:12.6f}")
    
    # Calculate partial scores for demonstration
    print(f"\n" + "="*100)
    print("CALCULATION DEMONSTRATION (First 24 fields only)")
    print("="*100)
    
    all_sample_data = list(daily_data.values()) + list(regime_data.values())
    
    acc_partial_score = sum(value * accuracy_weights[i] for i, value in enumerate(all_sample_data))
    pnl_partial_score = sum(value * pnl_weights[i] for i, value in enumerate(all_sample_data))
    
    print(f"\n🧮 PARTIAL WEIGHTED SCORES (first 24 fields only):")
    print(f"   • Model 00001 - Accuracy Strategy: {acc_partial_score:12.6f}")
    print(f"   • Model 00001 - PnL Strategy:      {pnl_partial_score:12.6f}")
    
    print(f"\n" + "="*100)
    print("MODEL 00002 COMPARISON (Simulated)")
    print("="*100)
    
    # Simulate slightly different values for model_00002
    print(f"\n📁 SOURCE: model_performance/model_daily_performance/model_00002_daily_performance.csv")
    print(f"📁 SOURCE: model_performance/model_regime_performance/model_00002_regime_performance.csv")
    print(f"📅 ROW: TradingDay={trading_day}, Regime={target_regime}")
    
    # Simulate model_00002 with slightly different performance
    model2_daily_data = {k: v * 0.95 for k, v in daily_data.items()}  # 5% lower performance
    model2_regime_data = {k: v * 1.02 for k, v in regime_data.items()}  # 2% higher regime performance
    
    print(f"\n🔍 MODEL 00002 DAILY SAMPLES (first 6 fields, simulated):")
    for i, (field, value) in enumerate(list(model2_daily_data.items())[:6]):
        print(f"   [{i:2d}] {field:25s} = {value:12.6f}")
    
    print(f"\n🔍 MODEL 00002 REGIME SAMPLES (first 6 fields, simulated):")
    for i, (field, value) in enumerate(list(model2_regime_data.items())[:6]):
        actual_index = regime_offset + i
        print(f"   [{actual_index:2d}] {field:25s} = {value:12.6f}")
    
    # Calculate model comparison
    all_model2_data = list(model2_daily_data.values()) + list(model2_regime_data.values())
    
    model2_acc_score = sum(value * accuracy_weights[i] for i, value in enumerate(all_model2_data))
    model2_pnl_score = sum(value * pnl_weights[i] for i, value in enumerate(all_model2_data))
    
    print(f"\n🧮 PARTIAL WEIGHTED SCORES COMPARISON (first 24 fields only):")
    print(f"   Model 00001 - Accuracy Strategy: {acc_partial_score:12.6f}")
    print(f"   Model 00002 - Accuracy Strategy: {model2_acc_score:12.6f}")
    print(f"   Model 00001 - PnL Strategy:      {pnl_partial_score:12.6f}")
    print(f"   Model 00002 - PnL Strategy:      {model2_pnl_score:12.6f}")
    
    print(f"\n🏆 WINNER SELECTION:")
    if acc_partial_score > model2_acc_score:
        print(f"   • Accuracy Strategy: Model 00001 WINS ({acc_partial_score:12.6f} > {model2_acc_score:12.6f})")
    else:
        print(f"   • Accuracy Strategy: Model 00002 WINS ({model2_acc_score:12.6f} > {acc_partial_score:12.6f})")
        
    if pnl_partial_score > model2_pnl_score:
        print(f"   • PnL Strategy:      Model 00001 WINS ({pnl_partial_score:12.6f} > {model2_pnl_score:12.6f})")
    else:
        print(f"   • PnL Strategy:      Model 00002 WINS ({model2_pnl_score:12.6f} > {pnl_partial_score:12.6f})")
    
    print(f"\n" + "="*100)
    print("FILE STRUCTURE AND FIELD ORGANIZATION")
    print("="*100)
    
    print(f"\n📂 DATA SOURCES:")
    print(f"   1. Daily Performance:  72 fields per model")
    print(f"      • File pattern: model_performance/model_daily_performance/model_XXXXX_daily_performance.csv")
    print(f"      • Fields: daily_[up/down]_[acc/num/den/pnl]_thr_[0.0-0.8] across 11 time periods")
    print(f"      • Time periods: daily, 2day, 3day, 1week, 2week, 4week, 8week, 13week, 26week, 52week, from_begin")
    
    print(f"\n   2. Regime Performance: 800 fields per model per regime")
    print(f"      • File pattern: model_performance/model_regime_performance/model_XXXXX_regime_performance.csv")
    print(f"      • Fields: [1-30]day_[up/down]_[acc/num/den/pnl]_thr_[0.0-0.8]")
    print(f"      • Time periods: 1day, 2day, 3day, 4day, 5day, 10day, 20day, 30day (10 periods)")
    
    print(f"\n📊 WEIGHTING PATTERN:")
    print(f"   • Total fields per model: 72 (daily) + 800 (regime) = 872 fields")
    print(f"   • Accuracy weights: [2.0, 0.5, 2.0, 0.5, ...] - emphasizes acc and den fields")
    print(f"   • PnL weights:      [0.5, 2.0, 0.5, 2.0, ...] - emphasizes num and pnl fields")
    print(f"   • Field order:      acc, num, den, pnl, acc, num, den, pnl, ...")
    
    print(f"\n⚠️  NOTE: This example shows calculation for first 24 fields only.")
    print(f"    Complete calculation uses all 872 fields per model.")
    print(f"    FastModelTradingWeighter uses this exact same logic with optimized vectorized operations.")
    
    print(f"\n" + "="*100)

if __name__ == "__main__":
    create_detailed_calculation_example()
