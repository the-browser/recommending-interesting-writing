import copy
import jobs
import pathlib
import addict


def get_slurm_script_gpu(output_dir, command):
    """Returns contents of SLURM script for a gpu job."""
    return """#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:tesla_p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --output={}/slurm_%j.out
#SBATCH -t 05:59:00
#module load anaconda3 cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1
#source activate yumi
{}
""".format(
        output_dir, command
    )


if __name__ == "__main__":
    commands = [
        "PYTHONPATH=. python train-rankfromsets.py  --train_path /scratch/gpfs/altosaar/dat/longform-data/main/combined-data/train.json --test_path /scratch/gpfs/altosaar/dat/longform-data/main/combined-data/test.json --eval_path /scratch/gpfs/altosaar/dat/longform-data/main/combined-data/evaluation.json "
    ]

    experiment_name = "news-rankfromsets"
    log_dir = (
        pathlib.Path(pathlib.os.environ["LOG"]) / "news-classification-inner-product"
    )

    base_grid = addict.Dict()
    base_grid.create_dicts = False
    base_grid.map_items = False
    base_grid.emb_size = [10, 25, 50, 100]
    base_grid.recall_max = 100
    base_grid.tokenize = False
    base_grid.target_publication = 0
    base_grid.batch_size = 2000
    base_grid.training_steps = 1500
    base_grid.momentum = 0.9
    base_grid.use_sparse = False
    base_grid.use_gpu = True
    base_grid.frequency = 50
    base_grid.dict_dir = pathlib.Path(
        "/scratch/gpfs/altosaar/dat/longform-data/main/dictionaries"
    )
    base_grid.tokenizer_file = (
        "/scratch/gpfs/altosaar/dat/longform-data/main/bert-base-uncased.txt"
    )
    base_grid.index_file_path = (
        "/scratch/gpfs/altosaar/dat/longform-data/BERT/eval_indices_list.txt"
    )

    # RMS with all words
    grid = copy.deepcopy(base_grid)
    grid["optimizer_type"] = "RMS"
    grid["use_all_words"] = True
    grid["learning_rate"] = [1e-1, 1e-3, 1e-4, 1e-5]
    grid["word_embedding_type"] = ["sum", "mean"]
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    keys_for_dir_name.insert(0, "optimizer_type")
    keys_for_dir_name.insert(1, "use_all_words")
    for cfg in jobs.param_grid(grid):
        cfg["output_dir"] = jobs.make_output_dir(
            log_dir, experiment_name, cfg, keys_for_dir_name
        )
        jobs.submit(commands, cfg, get_slurm_script_gpu)

    # RMS with only unique from first 500 words
    grid = copy.deepcopy(base_grid)
    grid["optimizer_type"] = "RMS"
    grid["use_all_words"] = False
    grid["words_to_use"] = 500
    grid["learning_rate"] = [1e-1, 1e-3, 1e-4, 1e-5]
    grid["word_embedding_type"] = ["sum", "mean"]
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    keys_for_dir_name.insert(0, "optimizer_type")
    keys_for_dir_name.insert(1, "use_all_words")
    for cfg in jobs.param_grid(grid):
        cfg["output_dir"] = jobs.make_output_dir(
            log_dir, experiment_name, cfg, keys_for_dir_name
        )
        jobs.submit(commands, cfg, get_slurm_script_gpu)

    # SGD with all words and sum
    grid = copy.deepcopy(base_grid)
    grid["optimizer_type"] = "SGD"
    grid["use_all_words"] = True
    grid["learning_rate"] = [0.1, 1, 5, 10, 15]
    grid["word_embedding_type"] = "sum"
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    keys_for_dir_name.insert(0, "optimizer_type")
    keys_for_dir_name.insert(1, "use_all_words")
    keys_for_dir_name.insert(2, "word_embedding_type")
    for cfg in jobs.param_grid(grid):
        cfg["output_dir"] = jobs.make_output_dir(
            log_dir, experiment_name, cfg, keys_for_dir_name
        )
        jobs.submit(commands, cfg, get_slurm_script_gpu)

    # SGD with all words and mean
    grid = copy.deepcopy(base_grid)
    grid["optimizer_type"] = "SGD"
    grid["use_all_words"] = True
    grid["learning_rate"] = [60, 600, 3000, 6000, 9000]
    grid["word_embedding_type"] = "mean"
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    keys_for_dir_name.insert(0, "optimizer_type")
    keys_for_dir_name.insert(1, "use_all_words")
    keys_for_dir_name.insert(2, "word_embedding_type")
    for cfg in jobs.param_grid(grid):
        cfg["output_dir"] = jobs.make_output_dir(
            log_dir, experiment_name, cfg, keys_for_dir_name
        )
        jobs.submit(commands, cfg, get_slurm_script_gpu)

    # SGD with only unique from first 500 words and sum
    grid = copy.deepcopy(base_grid)
    grid["optimizer_type"] = "SGD"
    grid["use_all_words"] = False
    grid["words_to_use"] = 500
    grid["learning_rate"] = [0.1, 1, 5, 10, 15]
    grid["word_embedding_type"] = "mean"
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    keys_for_dir_name.insert(0, "optimizer_type")
    keys_for_dir_name.insert(1, "use_all_words")
    keys_for_dir_name.insert(2, "word_embedding_type")
    for cfg in jobs.param_grid(grid):
        cfg["output_dir"] = jobs.make_output_dir(
            log_dir, experiment_name, cfg, keys_for_dir_name
        )
        jobs.submit(commands, cfg, get_slurm_script_gpu)

    # SGD with only unique from first 500 words and sum
    grid = copy.deepcopy(base_grid)
    grid["optimizer_type"] = "SGD"
    grid["use_all_words"] = False
    grid["words_to_use"] = 500
    grid["learning_rate"] = [30, 300, 1500, 3000, 4500]
    grid["word_embedding_type"] = "mean"
    keys_for_dir_name = jobs.get_keys_for_dir_name(grid)
    keys_for_dir_name.insert(0, "optimizer_type")
    keys_for_dir_name.insert(1, "use_all_words")
    keys_for_dir_name.insert(2, "word_embedding_type")
    for cfg in jobs.param_grid(grid):
        cfg["output_dir"] = jobs.make_output_dir(
            log_dir, experiment_name, cfg, keys_for_dir_name
        )
        jobs.submit(commands, cfg, get_slurm_script_gpu)
