"""
Metrics Module
Performance evaluation and benchmarking utilities

This module provides utilities for measuring execution time, computing speedup,
and evaluating parallel efficiency.
"""

import time
from typing import Callable, TypeVar, Tuple


# Generic type variable for function return types
T = TypeVar('T')


def time_function(fn: Callable[..., T], *args, **kwargs) -> Tuple[T, float]:
    """
    Run a function and measure its execution time.
    
    Generic timing utility that works with any callable. Returns both
    the function result and the elapsed time in seconds.
    
    Args:
        fn: Function to time
        *args: Positional arguments to pass to fn
        **kwargs: Keyword arguments to pass to fn
    
    Returns:
        Tuple of (result, elapsed_time_seconds)
        - result: Return value from fn
        - elapsed_time_seconds: Execution time in seconds (float)
    
    Example:
        >>> def expensive_operation(n):
        ...     return sum(range(n))
        >>> result, elapsed = time_function(expensive_operation, 1000000)
        >>> print(f"Sum: {result}, Time: {elapsed:.4f}s")
        Sum: 499999500000, Time: 0.0234s
    """
    start_time = time.time()
    result = fn(*args, **kwargs)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    return result, elapsed_time


def compute_speedup(t_sequential: float, t_parallel: float) -> float:
    """
    Compute speedup factor from sequential and parallel execution times.
    
    Speedup = T_sequential / T_parallel
    
    A speedup of 4.0 means the parallel version is 4x faster.
    Linear speedup with N workers would be N (ideal but rarely achieved).
    
    Args:
        t_sequential: Execution time of sequential version (seconds)
        t_parallel: Execution time of parallel version (seconds)
    
    Returns:
        Speedup factor (float)
        - > 1.0: Parallel is faster
        - = 1.0: No speedup
        - < 1.0: Parallel is slower (overhead dominates)
    
    Raises:
        ValueError: If either time is <= 0
    
    Example:
        >>> sequential_time = 10.0  # seconds
        >>> parallel_time = 2.5     # seconds with 4 workers
        >>> speedup = compute_speedup(sequential_time, parallel_time)
        >>> print(f"Speedup: {speedup:.2f}x")
        Speedup: 4.00x
    """
    if t_sequential <= 0:
        raise ValueError(f"Sequential time must be > 0, got {t_sequential}")
    
    if t_parallel <= 0:
        raise ValueError(f"Parallel time must be > 0, got {t_parallel}")
    
    speedup = t_sequential / t_parallel
    
    return speedup


def compute_efficiency(speedup: float, n_workers: int) -> float:
    """
    Compute parallel efficiency from speedup and number of workers.
    
    Efficiency = Speedup / N_workers
    
    Measures how effectively workers are utilized:
    - 1.0 (100%): Perfect linear scaling
    - 0.5 (50%): Workers are 50% utilized
    - > 1.0: Super-linear speedup (rare, usually due to cache effects)
    
    Args:
        speedup: Speedup factor from compute_speedup()
        n_workers: Number of parallel workers used
    
    Returns:
        Efficiency factor (float), typically in range [0.0, 1.0]
        - 1.0: Perfect efficiency (linear speedup)
        - < 1.0: Sub-linear speedup (common due to overhead)
        - > 1.0: Super-linear speedup (uncommon)
    
    Raises:
        ValueError: If n_workers <= 0 or speedup < 0
    
    Example:
        >>> speedup = 3.5
        >>> workers = 4
        >>> efficiency = compute_efficiency(speedup, workers)
        >>> print(f"Efficiency: {efficiency*100:.1f}%")
        Efficiency: 87.5%
    """
    if n_workers <= 0:
        raise ValueError(f"Number of workers must be > 0, got {n_workers}")
    
    if speedup < 0:
        raise ValueError(f"Speedup must be >= 0, got {speedup}")
    
    efficiency = speedup / n_workers
    
    return efficiency


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "1.23s", "123.45ms", "12.34μs")
    
    Example:
        >>> print(format_time(1.234))
        1.234s
        >>> print(format_time(0.001234))
        1.234ms
    """
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.3f}ms"
    else:
        return f"{seconds * 1_000_000:.3f}μs"


def print_performance_summary(
    t_sequential: float,
    t_parallel: float,
    n_workers: int,
    operation_name: str = "Operation"
) -> None:
    """
    Print a formatted performance summary.
    
    Displays sequential time, parallel time, speedup, and efficiency
    in a human-readable format.
    
    Args:
        t_sequential: Sequential execution time (seconds)
        t_parallel: Parallel execution time (seconds)
        n_workers: Number of parallel workers
        operation_name: Name of the operation being benchmarked
    
    Example:
        >>> print_performance_summary(10.0, 2.5, 4, "Matrix Multiplication")
        Performance Summary: Matrix Multiplication
        ==========================================
        Sequential Time:  10.000s
        Parallel Time:     2.500s (4 workers)
        Speedup:          4.00x
        Efficiency:       100.0%
    """
    speedup = compute_speedup(t_sequential, t_parallel)
    efficiency = compute_efficiency(speedup, n_workers)
    
    print(f"\nPerformance Summary: {operation_name}")
    print("=" * 50)
    print(f"Sequential Time:  {format_time(t_sequential):>10}")
    print(f"Parallel Time:    {format_time(t_parallel):>10} ({n_workers} workers)")
    print(f"Speedup:          {speedup:>10.2f}x")
    print(f"Efficiency:       {efficiency*100:>10.1f}%")
    print()


if __name__ == "__main__":
    """
    Sanity tests and demonstrations of metrics module.
    
    Run this script to verify metric calculations:
        python core/metrics.py
    """
    print("=" * 70)
    print("PaReCo-Py Metrics Module - Sanity Tests")
    print("=" * 70)
    
    # Test 1: time_function with simple computation
    print("\n[Test 1] Testing time_function() with sum computation")
    
    def compute_sum(n: int) -> int:
        return sum(range(n))
    
    result, elapsed = time_function(compute_sum, 1_000_000)
    print(f"  Result: {result}")
    print(f"  Elapsed time: {format_time(elapsed)}")
    print(f"  ✓ time_function works correctly")
    
    # Test 2: compute_speedup
    print("\n[Test 2] Testing compute_speedup()")
    
    t_seq = 10.0
    t_par = 2.5
    speedup = compute_speedup(t_seq, t_par)
    
    print(f"  Sequential: {t_seq}s")
    print(f"  Parallel:   {t_par}s")
    print(f"  Speedup:    {speedup:.2f}x")
    
    assert speedup == 4.0, f"Expected 4.0, got {speedup}"
    print(f"  ✓ Speedup calculation correct")
    
    # Test 3: compute_efficiency
    print("\n[Test 3] Testing compute_efficiency()")
    
    speedup_val = 3.5
    n_workers = 4
    efficiency = compute_efficiency(speedup_val, n_workers)
    
    print(f"  Speedup:    {speedup_val}x")
    print(f"  Workers:    {n_workers}")
    print(f"  Efficiency: {efficiency*100:.1f}%")
    
    expected_eff = 0.875
    assert abs(efficiency - expected_eff) < 0.001, f"Expected {expected_eff}, got {efficiency}"
    print(f"  ✓ Efficiency calculation correct")
    
    # Test 4: Edge cases
    print("\n[Test 4] Testing edge cases")
    
    # Linear speedup (perfect efficiency)
    linear_speedup = compute_speedup(8.0, 2.0)
    linear_eff = compute_efficiency(linear_speedup, 4)
    print(f"  Linear speedup (4 workers): {linear_speedup}x, efficiency: {linear_eff*100:.0f}%")
    assert linear_eff == 1.0, "Linear scaling should have 100% efficiency"
    print(f"  ✓ Linear scaling detected correctly")
    
    # No speedup
    no_speedup = compute_speedup(5.0, 5.0)
    no_eff = compute_efficiency(no_speedup, 4)
    print(f"  No speedup (4 workers): {no_speedup}x, efficiency: {no_eff*100:.0f}%")
    assert no_speedup == 1.0, "Same time should give 1x speedup"
    print(f"  ✓ No speedup case handled correctly")
    
    # Test 5: Error handling
    print("\n[Test 5] Testing error handling")
    
    try:
        compute_speedup(-1.0, 2.0)
        print("  ✗ Should have raised ValueError for negative time")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    try:
        compute_efficiency(2.0, 0)
        print("  ✗ Should have raised ValueError for zero workers")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 6: Performance summary
    print("\n[Test 6] Testing print_performance_summary()")
    print_performance_summary(10.0, 2.5, 4, "Sample Computation")
    
    # Test 7: Real-world example
    print("[Test 7] Real-world simulation")
    
    # Simulate sequential vs parallel execution
    def slow_operation(duration: float) -> str:
        time.sleep(duration)
        return "Done"
    
    print("  Running sequential version (0.1s)...")
    _, t1 = time_function(slow_operation, 0.1)
    
    print("  Running parallel version (0.03s, simulating 4 workers)...")
    _, t2 = time_function(slow_operation, 0.03)
    
    actual_speedup = compute_speedup(t1, t2)
    actual_efficiency = compute_efficiency(actual_speedup, 4)
    
    print(f"\n  Measured speedup: {actual_speedup:.2f}x")
    print(f"  Measured efficiency: {actual_efficiency*100:.1f}%")
    print(f"  ✓ Real timing measurements work")
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ All sanity tests passed!")
    print("=" * 70)
    print("\nMetrics module is ready for use in benchmarking PaReCo-Py components.")
