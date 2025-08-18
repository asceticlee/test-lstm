# Test Directory Structure

This directory contains organized verification scripts and analysis tools for the FastModelTradingWeighter project.

## 📁 Directory Structure

```
tests/
├── run_all_tests.py          # Master test runner
├── verification/             # Core functionality verification
│   ├── test_fast_weighter_complete.py
│   └── test_regime_filtering.py
├── analysis/                 # Data analysis tools
│   └── analyze_weighting_fields.py
└── demos/                    # Educational demonstrations
    └── calculation_demo.py
```

## 🧪 Verification Scripts

### `verification/test_fast_weighter_complete.py`
Comprehensive verification of FastModelTradingWeighter functionality:
- ✅ Alltime files integration
- ✅ Regime-specific filtering
- ✅ Complete model selection workflow

### `verification/test_regime_filtering.py`
Detailed verification of regime filtering:
- ✅ Different regimes produce different values
- ✅ Raw data file analysis
- ✅ Filtering mechanism validation

## 📊 Analysis Tools

### `analysis/analyze_weighting_fields.py`
Analyzes weighting field structure and usage:
- Field distribution analysis
- Weight importance ranking
- File usage patterns

## 🎓 Demo Scripts

### `demos/calculation_demo.py`
Educational demonstrations of:
- Weighting calculation formulas
- Example calculations with different strategies
- Performance field structure explanation

## 🚀 Usage

### Run All Tests
```bash
cd /home/stephen/projects/Testing/TestPy/test-lstm
python tests/run_all_tests.py
```

### Run Individual Tests
```bash
# Complete verification
python tests/verification/test_fast_weighter_complete.py

# Regime filtering only
python tests/verification/test_regime_filtering.py

# Field analysis
python tests/analysis/analyze_weighting_fields.py

# Calculation demo
python tests/demos/calculation_demo.py
```

## ✅ Test Coverage

The test suite verifies:
1. **Alltime Integration**: Confirms alltime files are loaded and used
2. **Regime Filtering**: Validates regime-specific data filtering
3. **Model Selection**: Tests complete model selection workflow
4. **Data Integrity**: Verifies data consistency and accuracy
5. **Performance**: Ensures reasonable execution times

## 📋 Test Results

All tests should pass with the following confirmations:
- ✅ Alltime performance data loaded (425 models)
- ✅ Alltime regime performance data loaded (2,125 records)
- ✅ Different regimes produce different values
- ✅ Model selection completes successfully
- ✅ Performance data contains 1,442+ fields per model

## 🔧 Maintenance

To update tests:
1. Modify individual test scripts in their respective directories
2. Update this README if new tests are added
3. Ensure all tests pass before committing changes

---

*This test suite ensures the FastModelTradingWeighter functions correctly with comprehensive verification and analysis capabilities.*
