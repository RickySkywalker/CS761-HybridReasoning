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

class NL_solver:
    """
    This is the helper class for writing the NL answer for the Math dataset.
    """

    def __init__(
            self, 
            model_id: str,
    ):
        self.model = LLM(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = LLM_rollout(self.model, self.tokenizer)

    def solve_question_single_data(
            self, 
            data_record_to_solve: Dict, 
            max_tokens: int = 8192, 
            solve_num: int = 16,
            temperature=0.6,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True
    ) -> Tuple[List[str], List[str]]:
        """
        This is the function to solve the question for a single data record in the Math dataset.

        Args:
            data_record_to_solve (Dict): The data record should be a dict with the following format:
                {
                    "Name": str for the name of the theorem
                    "problem": str for problem
                    "solution": str for step-by-step solution (optional)
                    "answer": str for the answer (optional)
                    "subject": str indicating the subject for the problem,
                    "level": int to indicate the level,
                }
            solve_num (int): The number of solutions to generate for each problem.
            max_tokens (int): The maximum number of tokens to generate.
        
        Returns:
            solved_answers: List[str], a list with length no larger than the `solve_num`, representing the generated answer 
                in the \\boxed{} format.
            response_ls: List[str], a list with size equal to solved_answers, representing the total generation 
                results for the record
        """

        solved_answers = []
        response_ls = []

        curr_question = data_record_to_solve["problem"]
        
        for i in range(int(solve_num / batch_size)):
            prompt = PromptWriter.formulate_prompt_direct_answer(
                curr_question, 
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

            for curr_response in curr_responses:
                curr_answer = utils.extract_boxed(curr_response)
                if curr_answer != None:
                    solved_answers.append(curr_answer)
                    response_ls.append(curr_response)
                
                if print_result:
                    print(f"{'#' * 20}\ncurrent response is:\n{curr_response}")
        return solved_answers, response_ls
    
    def solve_question_dataset(
            self, 
            set_to_solve: List[Dict], 
            max_tokens: int = 8192, 
            solve_num: int = 16,
            temperature=0.8,
            top_p=0.95,
            batch_size=4,
            repetition_penalty=1.0,
            print_result=True,
            ckpt_path : str = "./Solved_answer_ckpt"
    ) -> List[Dict]:
        """
        This is the function to solve the Math500 dataset.
        
        Args:
            set_to_solve (List[Dict]): The dataset to solve, should follows the format as below:
                [{
                    "Name": str for the name of the theorem
                    "problem": str for problem
                    "solution": str for step-by-step solution (optional)
                    "answer": str for the answer (optional)
                    "subject": str indicating the subject for the problem,
                    "level": int to indicate the level,
                }, ...]
            solve_num (int): The number of solutions to generate for each problem.
            max_tokens (int): The maximum number of tokens to generate.
        
        Returns:
            updated `set_to_solve` with the following format:
                [{
                    "Name": str for the name of the theorem
                    "problem": str for problem
                    "solution": str for step-by-step solution (optional)
                    "answer": str for the answer (optional)
                    "subject": str indicating the subject for the problem,
                    "level": int to indicate the level,
                    "solved_answer_results" (List[Dict]) of the generated results in solving answers:
                        [{
                            "solved_answer": str for the generated answer in the \\boxed{} format
                            "response": str for the total generation results for the record
                        }]
                }, ...]
        """

        for i in tqdm(range(len(set_to_solve))):
            curr_data_record = set_to_solve[i]
            curr_solved_answer, curr_response_ls = self.solve_question_single_data(
                curr_data_record,
                max_tokens=max_tokens,
                solve_num=solve_num,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                repetition_penalty=repetition_penalty,
                print_result=print_result
            )
            assert len(curr_solved_answer) == len(curr_response_ls), \
                (f"The length of curr_solved_answer (len: {len(curr_solved_answer)}) and "
                 f"curr_response_ls (len: {len(curr_response_ls)}) should be the same")
            
            curr_data_record["solved_answer_results"] = []
            for j in range(len(curr_solved_answer)):
                curr_data_record["solved_answer_results"].append({
                    "solved_answer": curr_solved_answer[j],
                    "response": curr_response_ls[j]
                })
            set_to_solve[i] = curr_data_record
            utils.write_to_json(f"{ckpt_path}/{curr_data_record['Name']}.json", curr_data_record)
        return set_to_solve


