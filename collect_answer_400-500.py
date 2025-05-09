from Answer_collector import AnswerCollector
from utils import utils
import os

DATASET_TO_GEN = "./data/proved_Math-500.json"
CKPT_PATH = "./Answer_collection_results_ckpt/MATH-500"
SAVE_PATH = "./Answer_collection_results/MATH-500"
MODEL_ID = "./data/models/Qwen3-8B"

CUDA_DEVICE_ID = 4
BEGIN_IDX = 400
END_IDX = 500
BATCH_SIZE = 4
MAX_TOKENS = 8192

def main():
    utils.check_folder_exit(CKPT_PATH)
    utils.check_folder_exit(SAVE_PATH)

    dataset_to_collect = utils.read_from_json(DATASET_TO_GEN)[BEGIN_IDX : END_IDX]
    answer_collector = AnswerCollector(MODEL_ID)

    answer_data_ls = answer_collector.collect_answer_dataset(
        set_to_collect=dataset_to_collect,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        batch_size=BATCH_SIZE,
        ckpt_path=CKPT_PATH
    )

    utils.write_to_json(f"./{SAVE_PATH}/Math-500_answer_{BEGIN_IDX}-{END_IDX}.json", answer_data_ls)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_ID)
    main()