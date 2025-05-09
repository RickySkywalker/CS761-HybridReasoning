from NL_solver import NL_solver
from utils import utils
import os

DATASET_TO_GEN = "./data/Math-500.json"
CKPT_PATH = "./NL_solver_results_ckpt/MATH-500"
SAVE_PATH = "./NL_solver_results/MATH-500"
MODEL_ID = "./data/models/Qwen3-8B"

CUDA_DEVICE_ID = 8
BEGIN_IDX = 200 + 39
END_IDX = 300
SOLVE_NUM = 16
BATCH_SIZE = 4
MAX_TOKENS = 8192

def main():
    utils.check_folder_exit(CKPT_PATH)
    utils.check_folder_exit(SAVE_PATH)

    dataset_to_solve = utils.read_from_json(DATASET_TO_GEN)[BEGIN_IDX : END_IDX]
    nl_solver = NL_solver(MODEL_ID)

    NL_solved_data_ls = nl_solver.solve_question_dataset(
        set_to_solve=dataset_to_solve,
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        solve_num=SOLVE_NUM,
        batch_size=BATCH_SIZE,
        ckpt_path=CKPT_PATH
    )

    utils.write_to_json(f"./{SAVE_PATH}/Math-500_solved_{BEGIN_IDX}-{END_IDX}.json", NL_solved_data_ls)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_ID)
    main()