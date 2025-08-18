# Archive Directory

This directory contains the original verification scripts that have been replaced by the organized test suite in the `tests/` directory.

## 📁 Archived Files

### Original Verification Scripts
- `test_fast_weighter_alltime.py` - Original alltime integration test
- `test_regime_filtering.py` - Original regime filtering test  
- `test_regime_verification_detailed.py` - Detailed regime verification

### Analysis Scripts
- `complete_weighting_file_analysis.py` - File usage analysis
- `weighting_fields_analysis.py` - Field structure analysis
- `weighting_verification.py` - General weighting verification

### Demo Scripts
- `detailed_calculation_demo.py` - Detailed calculation demonstrations
- `detailed_weighting_calculation.py` - Weighting calculation examples
- `calculation_example.py` - Basic calculation examples

## 🔄 Migration to New Structure

These scripts have been consolidated and improved in the new `tests/` directory:

| Old Script | New Location | Improvements |
|------------|--------------|--------------|
| `test_fast_weighter_alltime.py` | `tests/verification/test_fast_weighter_complete.py` | Combined with other tests |
| `test_regime_filtering.py` | `tests/verification/test_regime_filtering.py` | Cleaned up and streamlined |
| `weighting_fields_analysis.py` | `tests/analysis/analyze_weighting_fields.py` | Simplified and focused |
| `calculation_example.py` | `tests/demos/calculation_demo.py` | Better examples and explanations |

## 🗃️ Purpose

These files are kept for reference and historical purposes. The new test suite in `tests/` provides:
- Better organization
- Cleaner code
- Comprehensive coverage
- Easier maintenance

## ⚠️ Usage Note

These archived scripts may not work with the current codebase structure. Use the scripts in the `tests/` directory instead for all verification and analysis needs.

---

*Files archived during cleanup on August 18, 2025*
