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

class Prover:
    """
    This is the helper class for writing prove for the lean4 theorem
    """

    def __init__(
            self,
            model_id: str
    ):
        self.model = LLM(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = LLM_rollout(self.model, self.tokenizer)

    def write_proof_single_data(
            self, 
            data_record_to_prove: Dict,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True
    ) -> Tuple[List[str], List[str]]:
        """
        This is the function to write proof for a single theorem
        Args:
            data_record_to_prove: The data record should be a dict of the following format:
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
            max_tokens: the max_tokens for query

        Returns:
            generated_proof_content: List[str], a list with size no larger than `autoformalization_num`, representing the
                extracted generated existence content
            response_ls: List[str], a list with size equal to generated_proof_content, representing the total
                generation results for record
        """
        generated_proof_content = []
        response_ls = []

        curr_statements = data_record_to_prove["autoformalization_generated_results"]
        batched_statements = utils.batch_split(curr_statements, batch_size)

        for curr_batch in batched_statements:
            prompt_ls = []
            for curr_data_record_to_prove in curr_batch:
                prompt = PromptWriter.formulate_prompt_prover(
                    theorem_to_prove=curr_data_record_to_prove["autoformalization_generated_content"],
                    NL_question=data_record_to_prove["problem"],
                    tokenizer=self.tokenizer,
                )
                prompt_ls.append(prompt)
                if print_result:
                    print("current prompt is:")
                    print(prompt)
            curr_response = self.generator.batched_query_model(
                prompt_ls,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                sampling_num=1
            )

            for i, curr_response in enumerate(curr_response):
                if utils.contains_code_block(curr_response, "lean4"):
                    curr_proof = utils.extract_code_blocks_as_list(curr_response, "lean4")[-1]
                    generated_proof_content.append(curr_proof)
                    response_ls.append(curr_response)

                if print_result:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")
        return generated_proof_content, response_ls
    
    def write_proof_dataset(
            self, 
            set_to_prove: List[Dict],
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True,
            ckpt_path: str = "./Generated_proof_ckpt"
    ) -> List[Dict]:
        """
        This is the function to write proof for a dataset
        Args:
            set_to_prove: List[Dict], the set should follow the following format:
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
                    ...
                }]
            max_tokens: the max_tokens for query

        Returns:
            Updated set_to_prove: List[Dict], the updated set with proof generation:
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
        """

        for i in tqdm(range(len(set_to_prove))):
            curr_data_record = set_to_prove[i]
            curr_generated_proofs, curr_response_ls = self.write_proof_single_data(
                curr_data_record, 
                max_tokens=max_tokens, 
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                print_result=print_result
            )
            assert len(curr_generated_proofs) == len(curr_response_ls), \
                (f"The length of curr_generated_proofs (len: {len(curr_generated_proofs)}) and "
                 f"curr_response_ls (len: {len(curr_response_ls)}) should be the same")
            
            curr_data_record["generated_proof_results"] = []

            for j in range(len(curr_response_ls)):
                curr_data_record["generated_proof_results"] += [{
                    "generated_proof": curr_generated_proofs[j],
                    "response_proof_generation": curr_response_ls[j]
                }]
            set_to_prove[i] = curr_data_record
            utils.write_to_json(f"{ckpt_path}/{curr_data_record['Name']}.json", curr_data_record)
        return set_to_prove
        