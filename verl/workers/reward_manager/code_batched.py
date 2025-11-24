# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
import logging

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)


@register("code_batched")
class CodeBatchedRewardManager(AbstractRewardManager):
    """
    A batched code reward manager that processes sandbox requests in batches
    instead of sending them concurrently to avoid overwhelming the sandbox service.
    
    This manager groups all code execution requests and sends them to the sandbox
    in controlled batches with a configurable batch size and delay between batches.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        batch_size=16,
        batch_delay=0.1,
    ) -> None:
        """
        Args:
            tokenizer: The tokenizer to use for decoding responses
            num_examine: Number of batches of decoded responses to print
            compute_score: Custom score computation function (defaults to default_compute_score)
            reward_fn_key: Key to use for identifying data source
            max_resp_len: Maximum response length
            overlong_buffer_cfg: Configuration for overlong response penalties
            batch_size: Number of code execution requests to send in each batch (default: 16)
            batch_delay: Delay in seconds between batches to avoid overwhelming sandbox (default: 0.1)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.batch_size = batch_size
        self.batch_delay = batch_delay

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict=False):
        """Process data in batches to avoid overwhelming the sandbox service"""

        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # Prepare all items first
        items_to_process = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            items_to_process.append({
                'index': i,
                'prompt_str': prompt_str,
                'response_str': response_str,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'extra_info': extra_info,
                'valid_response_length': valid_response_length,
            })

        # Process items in batches
        import time
        total_items = len(items_to_process)
        logger.info(f"Processing {total_items} code execution requests in batches of {self.batch_size}")
        
        for batch_start in range(0, total_items, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_items)
            batch_items = items_to_process[batch_start:batch_end]
            
            logger.debug(f"Processing batch {batch_start//self.batch_size + 1}/{(total_items + self.batch_size - 1)//self.batch_size} "
                        f"(items {batch_start}-{batch_end-1})")
            
            # Process each item in the current batch sequentially
            for item in batch_items:
                i = item['index']
                result = self.compute_score(
                    data_source=item['data_source'],
                    solution_str=item['response_str'],
                    ground_truth=item['ground_truth'],
                    extra_info=item['extra_info'],
                )

                score: float
                if isinstance(result, dict):
                    score = result["score"]
                    extra_result_info = {key: value for key, value in result.items()}
                else:
                    score = result
                    extra_result_info = {"acc": score}

                # Store the extra info
                for key, value in extra_result_info.items():
                    reward_extra_info[key].append(value)

                reward = score

                # Apply overlong penalty if configured
                if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                    overlong_buffer_len = self.overlong_buffer_cfg.len
                    expected_len = self.max_resp_len - overlong_buffer_len
                    exceed_len = item['valid_response_length'] - expected_len
                    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                    reward += overlong_reward
                    if self.overlong_buffer_cfg.log:
                        reward_extra_info["overlong_reward"].append(overlong_reward)
                        reward_extra_info["overlong"].append(overlong_reward < 0)

                reward_tensor[i, item['valid_response_length'] - 1] = reward

                # Print examined samples
                data_source = item['data_source']
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", item['prompt_str'])
                    print("[response]", item['response_str'])
                    print("[ground_truth]", item['ground_truth'])
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)
            
            # Add delay between batches to avoid overwhelming the sandbox service
            if batch_end < total_items:
                time.sleep(self.batch_delay)

        logger.info(f"Completed processing {total_items} code execution requests")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
