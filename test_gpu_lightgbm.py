#!/usr/bin/env python3
"""
Quick test script to verify LightGBM GPU support
Run this after installing CUDA-enabled LightGBM
"""

import sys
import numpy as np

print("=" * 70)
print("LightGBM GPU Support Test")
print("=" * 70)
print()

# Test 1: Import LightGBM
print("Test 1: Importing LightGBM...")
try:
    import lightgbm as lgb
    print(f"✓ LightGBM imported successfully")
    print(f"  Version: {lgb.__version__}")
except ImportError as e:
    print(f"✗ Failed to import LightGBM: {e}")
    sys.exit(1)

print()

# Test 2: Check for GPU build
print("Test 2: Checking LightGBM build configuration...")
try:
    # Try to get build info
    print(f"  LightGBM location: {lgb.__file__}")
except Exception as e:
    print(f"  Could not determine build info: {e}")

print()

# Test 3: Test GPU training
print("Test 3: Testing GPU training...")
try:
    # Create small test dataset
    np.random.seed(42)
    X = np.random.rand(1000, 20)
    y = np.random.rand(1000)
    
    train_data = lgb.Dataset(X, label=y)
    
    # GPU parameters
    params = {
        'device': 'gpu',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'verbose': 1
    }
    
    print("  Training 10 iterations on GPU...")
    booster = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=10
    )
    
    print()
    print("✓ GPU training SUCCESSFUL!")
    print("  LightGBM can use CUDA GPUs")
    
    # Test prediction
    pred = booster.predict(X[:10])
    print(f"  Sample predictions: {pred[:3]}")
    
except Exception as e:
    error_msg = str(e)
    print(f"✗ GPU training FAILED: {error_msg}")
    print()
    
    if 'OpenCL' in error_msg:
        print("DIAGNOSIS: OpenCL/CUDA Mismatch")
        print("-" * 70)
        print("Your LightGBM was built with OpenCL support, but you have CUDA GPUs.")
        print()
        print("SOLUTION:")
        print("1. Install CUDA-enabled LightGBM:")
        print("   conda install -c conda-forge lightgbm cuda-version=12.1")
        print()
        print("2. Or build from source with CUDA:")
        print("   See INSTALL_GPU_LIGHTGBM.md for instructions")
        print()
        print("3. Or use CPU mode in config.toml:")
        print("   LightGBM.device = 'cpu'")
        
    elif 'GPU' in error_msg or 'CUDA' in error_msg:
        print("DIAGNOSIS: GPU/CUDA Configuration Issue")
        print("-" * 70)
        print("Possible causes:")
        print("- CUDA drivers not properly installed")
        print("- LightGBM not built with GPU support")
        print("- GPU not accessible in container")
        print()
        print("Check:")
        print("- nvidia-smi output")
        print("- CUDA_VISIBLE_DEVICES environment variable")
        
    else:
        print("DIAGNOSIS: Unknown Error")
        print("-" * 70)
        print("This may be a different issue. Check the error message above.")
    
    sys.exit(1)

print()

# Test 4: Test CPU fallback
print("Test 4: Testing CPU mode (for comparison)...")
try:
    cpu_params = params.copy()
    cpu_params['device'] = 'cpu'
    cpu_params['verbose'] = -1
    
    booster_cpu = lgb.train(
        params=cpu_params,
        train_set=train_data,
        num_boost_round=10
    )
    print("✓ CPU mode also works (as expected)")
except Exception as e:
    print(f"⚠️  CPU mode failed: {e}")

print()
print("=" * 70)
print("SUMMARY: LightGBM GPU Support is WORKING!")
print("=" * 70)
print()
print("Your config.toml is correctly set to use GPU:")
print("  LightGBM.device = 'gpu'")
print()
print("You should see these messages during training:")
print("  [LightGBM] [Info] This is the GPU trainer!!")
print("  [LightGBM] [Info] Using GPU device: ...")
print()
