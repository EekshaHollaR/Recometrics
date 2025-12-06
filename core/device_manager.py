"""
Device Manager Module
Handles GPU/CPU detection and manages parallel computing resources

This module provides automatic detection of available hardware (GPU via PyTorch CUDA
or CPU-only) and offers a unified interface for tensor operations across backends.
"""

from typing import Any, Optional
import numpy as np


class DeviceManager:
    """
    Manages computational device selection and tensor operations.
    
    Automatically detects whether PyTorch with CUDA support is available.
    Falls back to CPU-only NumPy operations if PyTorch is not installed
    or CUDA is unavailable.
    
    Attributes:
        use_gpu (bool): True if PyTorch with CUDA is available and will be used
        device (str): Device identifier - "cuda" for GPU, "cpu" for CPU-only
        np (module): NumPy module (always available)
        torch (module | None): PyTorch module if installed, None otherwise
    
    Example:
        >>> dm = DeviceManager()
        >>> tensor = dm.tensor([1, 2, 3], dtype="float32")
        >>> print(f"Using device: {dm.device}")
    """
    
    def __init__(self) -> None:
        """
        Initialize the DeviceManager and detect available computing backend.
        
        Automatically runs hardware detection and configures the appropriate
        computational backend (GPU or CPU).
        """
        self.use_gpu: bool = False
        self.device: str = "cpu"
        self.np = np  # NumPy is always available
        self.torch: Optional[Any] = None
        
        # Detect and initialize backend
        self._init_backend()
    
    def _init_backend(self) -> None:
        """
        Private method to detect GPU/CPU availability and initialize modules.
        
        Detection logic:
        1. Try to import PyTorch
        2. If successful, check if CUDA is available
        3. If both conditions are met, enable GPU mode
        4. Otherwise, fall back to CPU-only mode with NumPy
        
        Prints informative messages about the selected backend.
        """
        try:
            # Attempt to import PyTorch
            import torch
            
            self.torch = torch
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                self.use_gpu = True
                self.device = "cuda"
                print("[DeviceManager] Using GPU via PyTorch.")
                print(f"[DeviceManager] CUDA Device: {torch.cuda.get_device_name(0)}")
            else:
                # PyTorch installed but no CUDA
                self.use_gpu = False
                self.device = "cpu"
                print("[DeviceManager] PyTorch installed but CUDA not available. Using CPU only.")
        
        except ImportError:
            # PyTorch not installed
            self.torch = None
            self.use_gpu = False
            self.device = "cpu"
            print("[DeviceManager] PyTorch not installed. Using CPU only.")
    
    def tensor(self, data: Any, dtype: Optional[str] = None) -> Any:
        """
        Convert data to a tensor using the appropriate backend.
        
        If GPU/PyTorch is available, returns a PyTorch tensor on the selected device.
        Otherwise, returns a NumPy array.
        
        Args:
            data: Input data to convert (list, array, scalar, etc.)
            dtype: Optional data type string (e.g., "float32", "int64", "float64")
                  If None, uses default type inference
        
        Returns:
            torch.Tensor if GPU is available, otherwise np.ndarray
        
        Raises:
            ValueError: If an unsupported dtype string is provided
        
        Example:
            >>> dm = DeviceManager()
            >>> tensor = dm.tensor([1.0, 2.0, 3.0], dtype="float32")
            >>> # Returns torch.Tensor on GPU or np.ndarray on CPU
        """
        if self.use_gpu and self.torch is not None:
            # Use PyTorch with GPU
            try:
                if dtype is not None:
                    # Map string dtype to PyTorch dtype
                    torch_dtype = self._get_torch_dtype(dtype)
                    return self.torch.tensor(data, dtype=torch_dtype, device=self.device)
                else:
                    return self.torch.tensor(data, device=self.device)
            except Exception as e:
                print(f"[DeviceManager] Warning: Failed to create torch tensor: {e}")
                print("[DeviceManager] Falling back to NumPy array.")
                # Fallback to NumPy if torch tensor creation fails
                return self._create_numpy_array(data, dtype)
        else:
            # Use NumPy for CPU-only mode
            return self._create_numpy_array(data, dtype)
    
    def _get_torch_dtype(self, dtype_str: str) -> Any:
        """
        Convert string dtype to PyTorch dtype.
        
        Args:
            dtype_str: String representation of dtype (e.g., "float32", "int64")
        
        Returns:
            Corresponding PyTorch dtype
        
        Raises:
            ValueError: If dtype string is not recognized
        """
        if self.torch is None:
            raise ValueError("PyTorch is not available")
        
        dtype_mapping = {
            "float32": self.torch.float32,
            "float64": self.torch.float64,
            "float16": self.torch.float16,
            "int32": self.torch.int32,
            "int64": self.torch.int64,
            "int16": self.torch.int16,
            "int8": self.torch.int8,
            "uint8": self.torch.uint8,
            "bool": self.torch.bool,
        }
        
        if dtype_str not in dtype_mapping:
            raise ValueError(
                f"Unsupported dtype '{dtype_str}'. "
                f"Supported types: {list(dtype_mapping.keys())}"
            )
        
        return dtype_mapping[dtype_str]
    
    def _create_numpy_array(self, data: Any, dtype: Optional[str] = None) -> np.ndarray:
        """
        Create a NumPy array with optional dtype.
        
        Args:
            data: Input data to convert
            dtype: Optional NumPy dtype string
        
        Returns:
            NumPy ndarray
        """
        try:
            if dtype is not None:
                return np.array(data, dtype=dtype)
            else:
                return np.array(data)
        except Exception as e:
            raise ValueError(f"Failed to create NumPy array: {e}")
    
    def __repr__(self) -> str:
        """
        String representation of DeviceManager.
        
        Returns:
            Formatted string showing current configuration
        """
        return (
            f"DeviceManager(device='{self.device}', use_gpu={self.use_gpu}, "
            f"pytorch_available={self.torch is not None})"
        )


# Convenience function to get a global DeviceManager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """
    Get or create the global DeviceManager instance (singleton pattern).
    
    Returns:
        DeviceManager: The global DeviceManager instance
    
    Example:
        >>> dm = get_device_manager()
        >>> print(dm.device)
    """
    global _global_device_manager
    
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    
    return _global_device_manager
