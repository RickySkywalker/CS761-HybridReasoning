from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import torch
import re
import random
from tqdm.auto import tqdm
from utils import utils, PromptWriter
from utils.PromptWriter import Lean4_HEADER
from typing import List, Dict, Optional, Union, Tuple
from vllm.inputs import TokensPrompt
from vllm import LLM, SamplingParams


class LLM_rollout:
    """
    The inference component for Hybird reasoning
    """

    def __init__(
            self,
            model: LLM,
            tokenizer: PreTrainedTokenizer
    ):
        """
        The constructor for the LLM_rollout class, parameters list are as follows
        :param model: LLM model in the vllm package, we assume everything is set for the initialization
        :param tokenizer: tokenizer for the corresponding LLM for input
        :param terminators: the terminators for the model
        """
        self.model = model
        self.tokenizer = tokenizer


    def batched_query_model(
            self,
            prompts: List[str],
            max_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.0,
            sampling_num=2,
            stop_strs: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Function to query model with given prompts and return the result string, we assume all the instruction (or other) formats
        is provided in the prompt. The function will return the format for prompt + response
        :param prompts: List of prompts to query
        :param max_tokens: The max tokens for the whole sequence (prompt + generated text)
        :param temperature: temperature for sampling, larger means the result will be more random
        :param top_p: top_p for sampling. It means only sampling from the top p portion tokens.
        :param repetition_penalty: The penalty for repetition during sampling, larger means the model will prefer not to sampling tokens pre-existed in the sentence
        :param sampling_num: The number to sample for each prompt, when prompt is [a, b, c] and `sampling_num=2`, the sampled result will be [a_1, a_2, b_1, b_2, c_1, c_2]
        :param stop_strs: List of strings to indicate the LLM to stop sampling
        :return: List of strings of the completed sentence (prompt + response format)
        """
        tokenized_prompts = []
        for curr in prompts:
            curr_tokenized_prompt = self.tokenizer(
                [curr],
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"][0].tolist()

            tokenized_prompts += [curr_tokenized_prompt]

        if stop_strs == None:
            sampling_para = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                n=sampling_num,
                skip_special_tokens=False
            )
        else:
            sampling_para = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop_strs,
                max_tokens=max_tokens,
                n=sampling_num,
                skip_special_tokens=False
            )

        outputs = self.model.generate(
            prompts=None,
            prompt_token_ids=tokenized_prompts,
            sampling_params=sampling_para
        )
        output_strings = []
        for curr in outputs:
            for curr_response in curr.outputs:
                output_strings += [curr_response.text]

        return_strings = [x for x in prompts for _ in range(sampling_num)]
        assert len(output_strings) == len(return_strings), \
            (f"Output strings(len: {len(output_strings)}) and prompts (len: {len(return_strings)}) "
             f"duplicated number not identical, maybe some unknown bug")

        for i in range(len(return_strings)):
            return_strings[i] += output_strings[i]

        return return_strings