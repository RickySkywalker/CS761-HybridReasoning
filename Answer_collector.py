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

class AnswerCollector:
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

    def collect_answer_single_data(
            self, 
            data_record_to_collect: Dict,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True
    ) -> Tuple[List[str], List[str]]:
        """
        This is the function to collect answer for a single theorem
        Args:
            data_record_to_collect: The data record should be a dict of the following format:
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
                    "generated_proof_results": List[Dict] of generated results in proof
                        [{
                            "generated_proof": List[str], the proof content
                            "response_proof_generation": List[str], total generation results for proof
                        }, ...],
                    ...
                }
            autoformalization_num: total number of autoformalization to be made
            max_tokens: the max_tokens for query

        Returns:
            collected_answer_content: List[str], a list with size no larger than `generated_proof_results`, representing the
                collected answer in \\boxed\{\} format
            response_ls: List[str], a list with size equal to collected_answer_content, representing the total
                generation results for record
        """

        collected_answer_content = []
        response_ls = []

        curr_proof_results = data_record_to_collect["generated_proof_results"]
        batched_proof_results = utils.batch_split(curr_proof_results, batch_size)

        for curr_batch in batched_proof_results:
            prompt_ls = []
            for curr_data_record_to_collect in curr_batch:
                curr_generation_result = curr_data_record_to_collect["response_proof_generation"]
                curr_problem = data_record_to_collect["problem"]
                curr_prompt = PromptWriter.formulate_prompt_answer_collector(
                    prover_output=curr_generation_result, 
                    question_to_solve=curr_problem,
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
                curr_answer = utils.extract_boxed(curr_response)
                if curr_answer != None:
                    collected_answer_content.append(curr_answer[-1])
                    response_ls.append(curr_response)

                if print_result:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")

        return collected_answer_content, response_ls
    
    def collect_answer_dataset(
            self, 
            set_to_collect: List[Dict],
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True,
            ckpt_path : str = "./Collected_answer_ckpt"
    ) -> List[Dict]:    
        """
        This is the function to collect answer for a dataset of records

        Args:
            set_to_collect: The data record should be a list of dict with the following format:
                [{
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
                    "generated_proof_results": List[Dict] of generated results in proof
                        [{
                            "generated_proof": List[str], the proof content
                            "response_proof_generation": List[str], total generation results for proof
                        }, ...],
                    ...
                }]
        Response
            updated `set_to_collect` of the following format:
                [{
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
                    "generated_proof_results": List[Dict] of generated results in proof
                        [{
                            "generated_proof": List[str], the proof content
                            "response_proof_generation": List[str], total generation results for proof
                        }, ...],
                    "collected_answer_results": List[Dict] of generated results in proof
                        [{
                            "collected_answer": List[str], the proof content
                            "response_collected_answer_generation": List[str], total generation results for proof
                        }, ...],
                    ...
                }]
        """

        for i in tqdm(range(len(set_to_collect))):
            curr_data_record = set_to_collect[i]
            curr_collected_answer_content, curr_response_ls = self.collect_answer_single_data(
                curr_data_record, 
                max_tokens=max_tokens,
                temperature=temperature,   
                top_p=top_p,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                print_result=print_result
            )

            assert len(curr_collected_answer_content) == len(curr_response_ls), \
                (f"The length of curr_collected_answer_content (len: {len(curr_collected_answer_content)}) and "
                 f"curr_response_ls (len: {len(curr_response_ls)}) should be the same")
            
            curr_data_record["collected_answer_results"] = []
            for j in range(len(curr_response_ls)):
                curr_data_record["collected_answer_results"] += [{
                    "collected_answer": curr_collected_answer_content[j],
                    "response_collected_answer_generation": curr_response_ls[j]
                }]
            set_to_collect[i] = curr_data_record
            utils.write_to_json(f"{ckpt_path}/{curr_data_record['Name']}.json", curr_data_record)
        return set_to_collect

            