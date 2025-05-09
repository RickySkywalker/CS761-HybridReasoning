from QA_to_existence_traslator import QA_to_exsitence_translator
from utils import utils
import os
import torch

DATASET_TO_GEN = "./data/Math-500.json"
CKPT_PATH = "./QA-existence_problem_ckpt/MATH-500"
SAVE_PATH = "./QA-existence_problem/MATH-500"
MODEL_ID = "./data/models/Qwen3-8B"

CUDA_DEVICE_ID = 0
BEGIN_IDX = 0
END_IDX = 100
TRANSLATION_NUM = 8
BATCH_SIZE = 4
MAX_TOKENS = 8192


def main():
    utils.check_folder_exit(CKPT_PATH)
    utils.check_folder_exit(SAVE_PATH)

    dataset_to_translate = utils.read_from_json(DATASET_TO_GEN)[BEGIN_IDX : END_IDX]
    translator = QA_to_exsitence_translator(MODEL_ID)

    translated_dataset_ls = translator.write_translation_dataset(
        set_to_translate=dataset_to_translate,
        translation_num=TRANSLATION_NUM,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        batch_size=BATCH_SIZE,
        ckpt_path=CKPT_PATH
    )

    utils.write_to_json(f"./{SAVE_PATH}/Math-500_existence_translation_{BEGIN_IDX}-{END_IDX}.json", translated_dataset_ls)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_ID)
    main()