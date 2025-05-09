from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import torch
import re
import random
from tqdm.auto import tqdm
from utils import utils, PromptWriter
from utils.PromptWriter import Lean4_HEADER
from typing import List, Dict, Optional, Union, Tuple
from vllm import LLM, SamplingParams
from LLM_inference import LLM_rollout

class Autoformalizer:
    """
    This is the helper class for writing autoformalization of existence problem
    """
    def __init__(
            self,
            model_id: str
    ):
        self.model = LLM(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = LLM_rollout(self.model, self.tokenizer)

    def write_autoformalization_single_data(
            self, 
            data_record_to_autoformalize: Dict,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True
    ) -> Tuple[List[str], List[str]]:
        """
        This is the function to write autoformalization for a single theorem
        Args:
            data_record_to_autoformalize: The data record should be a dict of the following format:
                {
                    "Name": str for the name of the theorem
                    "problem": str for problem
                    "solution": str for step-by-step solution (optional)
                    "answer": str for the answer (optional)
                    "subject": str indicating the subject for the problem,
                    "level": int to indicate the level,
                    "translation_generated_results": List[Dict] of generated results
                        [{
                            "translated_existence_problem": List[str], the question translated to existence content
                            "response_existence_generation": List[str], total generation results for translation
                        }, ...]
                    ...
                }
            autoformalization_num: total number of autoformalization to be made
            max_tokens: the max_tokens for query

        Returns:
            generated_autoformalization_content: List[str], a list with size no larger than `autoformalization_num`, representing the
                extracted generated existence content
            response_ls: List[str], a list with size equal to generated_existence_content, representing the total
                generation results for record
        """
        generated_autoformalization_content = []
        response_ls = []

        curr_existence_problems = data_record_to_autoformalize["translation_generated_results"]
        batched_existence_problems = utils.batch_split(curr_existence_problems, batch_size)

        thm_idx = 0
        for curr_batch in batched_existence_problems:
            prompt_ls = []
            for curr_data_record_to_autoformalize in curr_batch:
                curr_existence_problem = curr_data_record_to_autoformalize["translated_existence_problem"]
                curr_name = f'{data_record_to_autoformalize["Name"]}_{thm_idx}'
                thm_idx += 1
                curr_prompt = PromptWriter.formulate_prompt_autoformalization(
                    name=curr_name,
                    problem_to_autoformalize=curr_existence_problem,
                    tokenizer=self.tokenizer
                )
                prompt_ls.append(curr_prompt)
                if print_result:
                    print("current prompt is:")
                    print(curr_prompt)
            curr_responses = self.generator.batched_query_model(
                prompt_ls,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                sampling_num=1
            )

            for i, curr_response in enumerate(curr_responses):
                curr_generated_content = curr_response[len(prompt_ls[i]):]
                match = re.search(r'theorem\b[\s\S]*?:=\s*by\b[\s\S]*', curr_generated_content)
                if match:
                    theorem_code = match.group(0)
                    generated_autoformalization_content.append(theorem_code)
                    response_ls.append(curr_response)
                if print_result:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")
        return generated_autoformalization_content, response_ls
    
    def write_autoformalization_dataset(
            self, 
            set_to_autoformalize: List[Dict],
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True,
            ckpt_path: str = "./Generated_autoformalization_ckpt"
    ) -> List[Dict]:
        """
        This is the function to write autoformalization for a dataset
        Args:
            set_to_autoformalize: The data record should be a dict of the following format:
                {
                    "Name": str for the name of the theorem
                    "problem": str for problem
                    "solution": str for step-by-step solution (optional)
                    "answer": str for the answer (optional)
                    "subject": str indicating the subject for the problem,
                    "level": int to indicate the level,
                    "translation_generated_results": List[Dict] of generated results
                        [{
                            "translated_existence_problem": List[str], the question translated to existence content
                            "response_existence_generation": List[str], total generation results for translation
                        }, ...]
                    ...
                }
            autoformalization_num: total number of autoformalization to be made
            max_tokens: the max_tokens for query

        Returns:
            updated `set_to_autoformalize` of the following format:
                {
                    "Name": str for the name of the theorem
                    "problem": str for problem
                    "solution": str for step-by-step solution (optional)
                    "answer": str for the answer (optional)
                    "subject": str indicating the subject for the problem,
                    "level": int to indicate the level,
                    "translation_generated_results": List[Dict] of generated results
                        [{
                            "translated_existence_problem": List[str], the question translated to existence content
                            "response_existence_generation": List[str], total generation results for translation
                        }, ...],
                    "autoformalization_generated_results": List[Dict] of generated results in autoformalization
                        [{
                            "autoformalization_generated_content": List[str], the autoformalization content
                            "response_autoformalization_generation": List[str], total generation results for autoformalization
                        }, ...],
                    ...
                }
        """
        for i in tqdm(range(len(set_to_autoformalize))):
            curr_data_record = set_to_autoformalize[i]
            curr_autoformalization_generated_content, curr_response_ls = self.write_autoformalization_single_data(
                curr_data_record,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                print_result=print_result
            )
            assert len(curr_autoformalization_generated_content) == len(curr_response_ls), \
                (f"The length of curr_autoformalization_generated_content (len: {len(curr_autoformalization_generated_content)}) and "
                 f"curr_response_ls (len: {len(curr_response_ls)}) should be the same")
            curr_data_record["autoformalization_generated_results"] = []
            for j in range(len(curr_response_ls)):
                curr_data_record["autoformalization_generated_results"] += [{
                    "autoformalization_generated_content": curr_autoformalization_generated_content[j],
                    "response_autoformalization_generation": curr_response_ls[j]
                }]
            set_to_autoformalize[i] = curr_data_record
            utils.write_to_json(f"{ckpt_path}/{curr_data_record['Name']}.json", curr_data_record)
        return set_to_autoformalize
        





