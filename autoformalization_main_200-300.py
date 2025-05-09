from Autoformalizer import Autoformalizer
from utils import utils
import os
import torch

DATASET_TO_GEN = "./data/translated_Math-500.json"
CKPT_PATH = "./Autoformalization_results_ckpt/MATH-500"
SAVE_PATH = "./Autoformalization_results/MATH-500"
MODEL_ID = "./data/models/Kimina-Autoformalizer-7B"

CUDA_DEVICE_ID = 2
BEGIN_IDX = 200
END_IDX = 300
BATCH_SIZE = 4
MAX_TOKENS = 8192


def main():
    utils.check_folder_exit(CKPT_PATH)
    utils.check_folder_exit(SAVE_PATH)

    dataset_to_autoformalize = utils.read_from_json(DATASET_TO_GEN)[BEGIN_IDX : END_IDX]
    autoformalizer = Autoformalizer(MODEL_ID)

    autoformalized_data_ls = autoformalizer.write_autoformalization_dataset(
        set_to_autoformalize=dataset_to_autoformalize,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        batch_size=BATCH_SIZE,
        ckpt_path=CKPT_PATH
    )

    utils.write_to_json(f"./{SAVE_PATH}/Math-500_autoformalization_{BEGIN_IDX}-{END_IDX}.json", autoformalized_data_ls)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_ID)
    main()