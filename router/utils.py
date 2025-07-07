import time
from contextlib import contextmanager
from typing import Optional, Callable
import functools


class Timer:
    """A utility class for measuring execution time in milliseconds."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None
    
    def start(self) -> 'Timer':
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds."""
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self.elapsed_ms
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds without stopping the timer."""
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        
        current_time = time.perf_counter()
        return (current_time - self.start_time) * 1000
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        if self.name:
            print(f"{self.name}: {self.elapsed_ms:.2f}ms")
        else:
            print(f"Execution time: {self.elapsed_ms:.2f}ms")


@contextmanager
def timer(name: Optional[str] = None):
    """
    Context manager for timing code execution.
    
    Args:
        name: Optional name for the timer (will be printed with the result)
    
    Example:
        with timer("Query execution"):
            # your code here
            pass
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if name:
            print(f"{name}: {elapsed_ms:.2f}ms")
        else:
            print(f"Execution time: {elapsed_ms:.2f}ms")


def time_function(func: Optional[Callable] = None, name: Optional[str] = None):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to decorate
        name: Optional name for the timer (defaults to function name)
    
    Example:
        @time_function
        def my_function():
            pass
        
        @time_function(name="Custom name")
        def another_function():
            pass
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            timer_name = name or f.__name__
            with timer(timer_name):
                return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


def measure_time(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure the execution time of a function call.
    
    Args:
        func: Function to measure
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Tuple of (result, elapsed_time_ms)
    
    Example:
        result, elapsed_ms = measure_time(my_function, arg1, arg2)
        print(f"Function took {elapsed_ms:.2f}ms")
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return result, elapsed_ms
