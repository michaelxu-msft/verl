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

from rllm.rewards.code_reward import rllm_reward_fn_code as compute_score

# Re-export it
__all__ = ['compute_score']
