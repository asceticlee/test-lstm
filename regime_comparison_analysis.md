# Market Regime Model Comparison Analysis

## Model Performance Comparison

### Summary of Models Tested:
1. **Standard Baseline (6 regimes)**: Original model without overnight features
2. **Overnight Gap Simplified (5 regimes)**: Model with single overnight_gap_absolute feature
3. **Forced 7-Regime Model**: Manually forced to produce 7 regimes for comparison

## Progressive Accuracy Results Comparison

| Time Period | Standard (6R) | Overnight (5R) | Forced 7R | 
|-------------|---------------|----------------|-----------|
| 9:30-09:45  | 0.093 (9.3%)  | 0.116 (11.6%) | 0.105 (10.5%) |
| 9:30-10:00  | 0.142 (14.2%) | 0.170 (17.0%) | 0.118 (11.8%) |
| 9:30-10:15  | 0.212 (21.2%) | 0.240 (24.0%) | 0.170 (17.0%) |
| 9:30-10:30  | 0.329 (32.9%) | 0.359 (35.9%) | 0.271 (27.1%) |
| 9:30-10:45  | 0.462 (46.2%) | 0.477 (47.7%) | 0.402 (40.2%) |
| 9:30-11:00  | 0.569 (56.9%) | 0.573 (57.3%) | 0.515 (51.5%) |
| 9:30-11:15  | 0.645 (64.5%) | 0.634 (63.4%) | 0.601 (60.1%) |
| 9:30-11:30  | 0.722 (72.2%) | 0.712 (71.2%) | 0.687 (68.7%) |
| 9:30-11:45  | 0.823 (82.3%) | 0.812 (81.2%) | 0.798 (79.8%) |
| 9:30-12:00  | 1.000 (100%)  | 1.000 (100%)  | 1.000 (100%) |

## Key Findings

### 1. Early Prediction Performance (15-60 minutes)
- **Overnight Gap Model (5R)** performs BEST in early periods
- **Standard Model (6R)** is middle performer
- **Forced 7-Regime Model** performs WORST in early periods

### 2. Mid-Period Performance (75-105 minutes)
- **Overnight Gap Model (5R)** maintains slight edge
- **Standard Model (6R)** very close behind
- **Forced 7-Regime Model** consistently lower

### 3. Late Period Performance (120-135 minutes)
- **Standard Model (6R)** performs BEST
- **Overnight Gap Model (5R)** very close second
- **Forced 7-Regime Model** consistently lowest

### 4. Overall Performance Ranking
1. **Overnight Gap Simplified (5R)**: Best early prediction, competitive throughout
2. **Standard Baseline (6R)**: Strong overall, best late-period performance
3. **Forced 7-Regime Model**: Consistently underperforms both optimized models

## Statistical Analysis

### Regime Count vs Performance
- **5 Regimes (Overnight)**: Excellent early prediction, maintains competitiveness
- **6 Regimes (Standard)**: Well-balanced performance across all time periods
- **7 Regimes (Forced)**: Poor performance suggests over-segmentation

### BIC Optimization Validation
The BIC optimization clearly made the right choice:
- 6-regime model (BIC: 44,196.6) outperforms forced 7-regime model
- 5-regime overnight model performs even better in early prediction
- Forced 7-regime model (BIC: 44,555.3) shows worst performance

## Conclusions

### 1. **Regime Count Discrepancy Explained**
The current algorithm producing 5-6 regimes instead of 7 is actually **OPTIMAL**:
- Lower regime counts provide better predictive accuracy
- BIC correctly identified 6 regimes as optimal balance
- Forced 7 regimes shows clear over-segmentation with worse performance

### 2. **Model Recommendations**
- **For Early Prediction**: Use **Overnight Gap Simplified (5R)** model
- **For Balanced Performance**: Use **Standard Baseline (6R)** model
- **Avoid**: Forcing 7 regimes - it degrades performance

### 3. **Algorithm Validation**
The current GMM implementation is working correctly:
- BIC optimization prevents over-fitting
- Lower regime counts improve generalization
- Historical 7-regime preference may have been sub-optimal

### 4. **Feature Engineering Success**
The overnight gap feature provides genuine early prediction improvement:
- +2.3% accuracy at 15 minutes (11.6% vs 9.3%)
- +2.8% accuracy at 30 minutes (17.0% vs 14.2%)
- Maintains competitiveness throughout the day

## Recommendation

**Use the Overnight Gap Simplified model (5 regimes)** as the primary model because:
1. Best early prediction performance when it matters most
2. Competitive performance throughout the day
3. Uses optimal regime count as determined by BIC
4. Incorporates valuable overnight market information
