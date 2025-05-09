from Prover import Prover
from utils import utils
import os

DATASET_TO_GEN = "./data/formalized_Math-500.json"
CKPT_PATH = "./Proof_generation_results_ckpt/MATH-500"
SAVE_PATH = "./Proof_generation_results/MATH-500"
MODEL_ID = "data/models/Kimina-Prover-Preview-Distill-7B"

CUDA_DEVICE_ID = 9
BEGIN_IDX = 0
END_IDX = 100
BATCH_SIZE = 4
MAX_TOKENS = 8192

def main():
    utils.check_folder_exit(CKPT_PATH)
    utils.check_folder_exit(SAVE_PATH)

    dataset_to_prove = utils.read_from_json(DATASET_TO_GEN)[BEGIN_IDX : END_IDX]
    prover = Prover(MODEL_ID)

    proof_data_ls = prover.write_proof_dataset(
        set_to_prove=dataset_to_prove,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        batch_size=BATCH_SIZE,
        ckpt_path=CKPT_PATH
    )

    utils.write_to_json(f"./{SAVE_PATH}/Math-500_proof_{BEGIN_IDX}-{END_IDX}.json", proof_data_ls)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_ID)
    main()