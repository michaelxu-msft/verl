import signal
import sys
import threading

# Monkey-patch signal.signal to ignore errors in non-main threads (Ray actors)
# This must happen before any rllm imports
_original_signal = signal.signal

def _is_main_thread() -> bool:
    """Return True if the current thread is the Python main thread."""
    return threading.current_thread() is threading.main_thread()

def _patched_signal(signalnum, handler):
    if _is_main_thread():
        try:
            return _original_signal(signalnum, handler)
        except Exception:
            # Fall back to ignoring errors in case other edge cases surface
            return None

    # Signal handlers can only be set in main thread.
    # Silently ignore requests from Ray actor threads while mimicking stdlib return type.
    return None

signal.signal = _patched_signal

# Also patch it in sys.modules to ensure all imports see the patched version
sys.modules['signal'].signal = _patched_signal

from rllm.rewards.code_reward import rllm_reward_fn_code

# Adapter wrapper to map VERL interface to rllm interface
def compute_score(data_source, solution_str, ground_truth, **kwargs):
    """Adapter wrapper that maps VERL's interface to rllm's interface.
    
    VERL calls with: data_source, solution_str, ground_truth
    rllm expects: data_source, llm_solution, ground_truth
    """
    return rllm_reward_fn_code(
        data_source=data_source,
        llm_solution=solution_str,  # Map solution_str -> llm_solution
        ground_truth=ground_truth,
        **kwargs
    )

# Re-export it
__all__ = ['compute_score']

def _is_main_thread() -> bool:
    """Return True if the current thread is the Python main thread."""
    return threading.current_thread() is threading.main_thread()
