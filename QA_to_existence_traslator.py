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

class QA_to_exsitence_translator:
    """
    This is the helper class for writing QA problem to existence problem translation
    """

    def __init__(
            self,
            model_id: str
    ):
        self.model = LLM(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = LLM_rollout(self.model, self.tokenizer)

    def write_translation_single_data(
            self,
            data_record_to_translate: Dict,
            translation_num=8,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True
    ) -> Tuple[List[str], List[str]]:
        """
        This is the function to write translation for a single theorem
        Args:
            data_record_to_translate: The data record should be a dict of the following format:
                {
                    "problem": str for problem
                    "solution": str for step-by-step solution (optional)
                    "answer": str for the answer (optional)
                    "subject": str indicating the subject for the problem,
                    "level": int to indicate the level,
                    ...
                }
            translation_num: total number of translation to be made
            max_tokens: the max_tokens for query

        Returns:
            generated_existence_content: List[str], a list with size no larger than `translation_num`, representing the
                extracted generated existence content
            response_ls: List[str], a list with size equal to generated_existence_content, representing the total
                generation results for record
        """
        generated_existence_content = []
        response_ls = []

        for i in range(int(translation_num/batch_size)):
            prompt = PromptWriter.formulate_prompt_existence_translation(
                data_record_to_translate["problem"],
                tokenizer=self.tokenizer
            )

            if print_result:
                print("current prompt is:")
                print(prompt)

            curr_responses = self.generator.batched_query_model(
                [prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                sampling_num=batch_size
            )

            # extract the code block for question
            for curr_response in curr_responses:
                if (utils.contains_code_block(curr_response, "md") or
                        utils.contains_code_block(curr_response, "Markdown") or
                        utils.contains_code_block(curr_response, "markdown")):
                    if utils.contains_code_block(curr_response, "md"):
                        curr_translated_question = utils.extract_code_blocks_as_list(curr_response, "md")
                    elif utils.contains_code_block(curr_response, "Markdown"):
                        curr_translated_question = utils.extract_code_blocks_as_list(curr_response, "Markdown")
                    elif utils.contains_code_block(curr_response, "markdown"):
                        curr_translated_question = utils.extract_code_blocks_as_list(curr_response, "markdown")
                    generated_existence_content += [curr_translated_question]
                    response_ls += [curr_response]

                if print_result:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")

        return generated_existence_content, response_ls



    def write_translation_dataset(
            self,
            set_to_translate: List[Dict],
            translation_num=8,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True,
            ckpt_path="./Generated_proof_translation_ckpt",
    ) -> List[Dict]:
        """
        This it the function to write translation for the dataset
        Args:
            set_to_translate: The list of dict of the following format:
                [{
                    "problem": str for problem
                    "solution": str for step-by-step solution (optional)
                    "answer": str for the answer (optional)
                    "subject": str indicating the subject for the problem,
                    "level": int to indicate the level,
                    ...
                }, ...]
            translation_num: total number of translation to be made
            batch_size: the batch_size for each single query
        Returns:
            updated `set_to_translate` of the following format:
                [{
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
                }, ...]
        """
        for i in tqdm(range(len(set_to_translate))):
            curr_data_record = set_to_translate[i]
            curr_translated_question, curr_response_ls = self.write_translation_single_data(
                curr_data_record,
                translation_num=translation_num,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                print_result=print_result
            )

            assert len(curr_translated_question) == len(curr_response_ls), \
                (f"The length of curr_translated_question (len: {len(curr_translated_question)}) and "
                 f"curr_response_ls (len: {len(curr_response_ls)}) should be the same")
            curr_data_record["translation_generated_results"] = []
            for j in range(len(curr_response_ls)):
                curr_data_record["translation_generated_results"] += [{
                    "translated_existence_problem": curr_translated_question[j],
                    "response_existence_generation": curr_response_ls
                }]
            set_to_translate[i] = curr_data_record

            if ckpt_path != None:
                curr_ckpt_filename = utils.hash_dict(curr_data_record) + ".json"
                utils.write_to_json(f"{ckpt_path}/{curr_ckpt_filename}.json", curr_data_record)