import signal

# Monkey-patch signal.signal to ignore errors in non-main threads (Ray actors)
_original_signal = signal.signal

def _patched_signal(signalnum, handler):
    try:
        return _original_signal(signalnum, handler)
    except ValueError:
        # Signal handlers can only be set in main thread
        # Silently ignore this in Ray actor threads
        pass

signal.signal = _patched_signal

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
