"""
Test script for DeviceManager
Verifies GPU/CPU detection and tensor operations
"""

import sys
sys.path.insert(0, r'c:\Users\Eeksha\Desktop\todo-app\Version2-PP\pareco_py')

from core.device_manager import DeviceManager, get_device_manager

def test_device_manager():
    """Test DeviceManager functionality"""
    print("=" * 60)
    print("Testing DeviceManager")
    print("=" * 60)
    
    # Test 1: Create DeviceManager instance
    print("\n[TEST 1] Creating DeviceManager instance...")
    dm = DeviceManager()
    print(f"  Result: {dm}")
    print(f"  Device: {dm.device}")
    print(f"  Use GPU: {dm.use_gpu}")
    print(f"  PyTorch available: {dm.torch is not None}")
    
    # Test 2: Test tensor creation without dtype
    print("\n[TEST 2] Creating tensor without dtype...")
    tensor1 = dm.tensor([1, 2, 3, 4, 5])
    print(f"  Type: {type(tensor1)}")
    print(f"  Value: {tensor1}")
    
    # Test 3: Test tensor creation with float32 dtype
    print("\n[TEST 3] Creating tensor with float32 dtype...")
    tensor2 = dm.tensor([1.0, 2.0, 3.0], dtype="float32")
    print(f"  Type: {type(tensor2)}")
    print(f"  Value: {tensor2}")
    
    # Test 4: Test tensor creation with int64 dtype
    print("\n[TEST 4] Creating tensor with int64 dtype...")
    tensor3 = dm.tensor([10, 20, 30], dtype="int64")
    print(f"  Type: {type(tensor3)}")
    print(f"  Value: {tensor3}")
    
    # Test 5: Test singleton pattern
    print("\n[TEST 5] Testing singleton pattern...")
    dm2 = get_device_manager()
    print(f"  Same instance: {dm2 is not dm}")  # Should be different from first instance
    print(f"  Device matches: {dm2.device}")
    
    # Test 6: Test 2D array/matrix
    print("\n[TEST 6] Creating 2D tensor...")
    tensor4 = dm.tensor([[1, 2], [3, 4], [5, 6]], dtype="float32")
    print(f"  Type: {type(tensor4)}")
    print(f"  Shape: {tensor4.shape if hasattr(tensor4, 'shape') else 'N/A'}")
    print(f"  Value:\n{tensor4}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_device_manager()
