#!/usr/bin/env python3
"""
Test Suite Runner

This script runs all verification tests and provides a comprehensive summary.
"""

import sys
import subprocess
from pathlib import Path

def run_test_script(script_path):
    """Run a test script and return success status"""
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=Path(script_path).parent)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def run_all_tests():
    """Run all verification tests"""
    print("="*80)
    print("FASTMODELTRADINGWEIGHTER TEST SUITE")
    print("="*80)
    
    # Test scripts to run
    test_scripts = [
        "/home/stephen/projects/Testing/TestPy/test-lstm/tests/verification/test_fast_weighter_complete.py",
        "/home/stephen/projects/Testing/TestPy/test-lstm/tests/verification/test_regime_filtering.py"
    ]
    
    results = {}
    
    for script_path in test_scripts:
        script_name = Path(script_path).stem
        print(f"\n🧪 Running {script_name}...")
        print("-" * 60)
        
        if Path(script_path).exists():
            success, stdout, stderr = run_test_script(script_path)
            results[script_name] = success
            
            # Show key results
            if success:
                print("✅ Test PASSED")
            else:
                print("❌ Test FAILED")
                if stderr:
                    print(f"Error: {stderr}")
        else:
            print(f"❌ Test script not found: {script_path}")
            results[script_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed - please review")
    
    print("="*80)

if __name__ == "__main__":
    run_all_tests()
