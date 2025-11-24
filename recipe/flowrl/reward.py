# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import signal
import sys
from functools import partial
import json

import ray

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward, get_custom_reward_fn
from verl.utils.reward_score import default_compute_score

import logging
import signal as _signal_module

logger = logging.getLogger(__name__)

# Global flag to prevent duplicate patch application
_patch_applied = False

# Save original function references
_original_functions = {}


def apply_signal_patch(verbose: bool = True):
    """
    Apply signal patch to allow AppWorld to run in a multithreading environment.

    Args:
        verbose: Whether to print patch application information

    Returns:
        bool: Whether the patch was successfully applied (returns False if already applied)
    """
    global _patch_applied, _original_functions

    if _patch_applied:
        if verbose:
            logger.info("Signal patch already applied, skipping")
        return False

    # Save original functions (for possible restoration)
    _original_functions["signal"] = _signal_module.signal
    _original_functions["getsignal"] = _signal_module.getsignal
    _original_functions["alarm"] = _signal_module.alarm
    if hasattr(_signal_module, "setitimer"):
        _original_functions["setitimer"] = _signal_module.setitimer

    # Define thread-safe alternative functions
    def _thread_safe_signal(signum, handler):
        """
        Thread-safe signal.signal() replacement.

        For SIGALRM, return None without setting a signal handler.
        For other signals, try using the original function, and if called in a non-main thread, catch the exception.
        """
        if signum == _signal_module.SIGALRM:
            return None
        try:
            return _original_functions["signal"](signum, handler)
        except ValueError as e:
            # ValueError: signal only works in main thread
            logger.debug(f"signal.signal() called in a non-main thread, ignored: {e}")
            return None

    def _thread_safe_getsignal(signum):
        """
        Thread-safe signal.getsignal() replacement.

        For SIGALRM, return None.
        For other signals, use the original function.
        """
        if signum == _signal_module.SIGALRM:
            return None
        return _original_functions["getsignal"](signum)

    def _thread_safe_alarm(seconds):
        """
        Thread-safe signal.alarm() replacement.

        Always return 0, indicating no previous alarm.
        Actually does not set any alarm.
        """
        return 0

    def _thread_safe_setitimer(which, seconds, interval=0):
        """
        Thread-safe signal.setitimer() replacement.

        Always return (0, 0), indicating no previous timer.
        Actually does not set any timer.
        """
        return (0, 0)

    # Apply the patch
    _signal_module.signal = _thread_safe_signal
    _signal_module.getsignal = _thread_safe_getsignal
    _signal_module.alarm = _thread_safe_alarm
    if hasattr(_signal_module, "setitimer"):
        _signal_module.setitimer = _thread_safe_setitimer

    _patch_applied = True

    if verbose:
        logger.info("Signal patch applied - AppWorld can now run in a multithreading environment")
        logger.warning("Warning: signal timeout protection is disabled, please set timeout control at a higher level")

    return True


def restore_signal_functions():
    """
    Restore the original signal functions.

    Warning: Only use this for testing or debugging. After restoration, AppWorld will no longer work in a multithreading environment.

    Returns:
        bool: Whether the signal functions were successfully restored (returns False if the patch was not applied)
    """
    global _patch_applied, _original_functions

    if not _patch_applied:
        logger.warning("Signal patch not applied, no need to restore")
        return False

    # Restore original functions
    _signal_module.signal = _original_functions["signal"]
    _signal_module.getsignal = _original_functions["getsignal"]
    _signal_module.alarm = _original_functions["alarm"]
    if "setitimer" in _original_functions:
        _signal_module.setitimer = _original_functions["setitimer"]

    _patch_applied = False
    _original_functions.clear()

    logger.info("Signal functions restored to original implementation")
    logger.warning("Warning: AppWorld will no longer work in a non-main thread")

    return True


def is_patch_applied():
    """
    Check if the signal patch has been applied.

    Returns:
        bool: Whether the patch has been applied
    """
    return _patch_applied


# Convenient context manager
class SignalPatch:
    """
    Context manager for signal patch.

    Usage example:
        with SignalPatch():
            from appworld import AppWorld
            world = AppWorld(task_id="test")
            # In this code block, the signal patch is activated
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.applied_by_me = False

    def __enter__(self):
        self.applied_by_me = apply_signal_patch(verbose=self.verbose)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.applied_by_me:
            restore_signal_functions()
        return False


apply_signal_patch()
from rllm.rewards.code_reward import rllm_reward_fn_code


def _default_compute_score(data_source, solution_str, ground_truth, **kwargs):
    """Adapter wrapper that maps VERL's interface to rllm's interface.
    
    VERL calls with: data_source, solution_str, ground_truth
    rllm expects: data_source, llm_solution, ground_truth
    
    This is used when sandbox_fusion is NOT configured.
    """
    try:
        # Parse ground truth (test cases)
        if isinstance(ground_truth, str):
            test_cases = json.loads(ground_truth)
        else:
            test_cases = ground_truth
        
        # Call rllm reward function
        result = rllm_reward_fn_code(
            data_source=data_source,
            llm_solution=solution_str,
            ground_truth=test_cases,
            **kwargs
        )
        
        return float(result) if isinstance(result, (bool, int, float)) else float(result[0])
    
    except Exception as e:
        print(f"Error computing code reward: {e}")
        return 0.0


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """
    from verl.workers.reward_manager import get_reward_manager_cls

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            # Use VERL's default_compute_score with sandbox_fusion parameters
            sandbox_manager = multiprocessing.Manager()
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb
            )
        else:
            # Use rllm_reward_fn_code when no sandbox is configured
            final_compute_score = _default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
