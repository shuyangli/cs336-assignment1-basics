# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### To profile a test:
`uv run scalene -m pytest tests/test_train_bpe.py::test_train_bpe_speed --profile-all`


### Commands to launch
```
uv run ./cs336_basics/training/training_loop.py --train-dataset data/TinystoriesV2-train.npy --val-dataset data/TinystoriesV2-valid.npy --num-layers 4 --vocab-size 10000 --context-length 256 --d-model 512 --num-heads 16 --d-ff 1344 --rope-theta 10000 --beta2 0.95 --learning-rate 0.001 --weight-decay 0.0001 --batch-size 32 --epochs 400 --device mps --save-every 500 --save-path ./checkpoints --enable-wandb --num-warmup-iterations 100
```

```
uv run ./cs336_basics/inference/inference_main.py --checkpoint checkpoints/epoch-5000.pt --num-layers 4 --vocab-size 10000 --context-length 256 --d-model 512 --num-heads 16 --d-ff 1344 --rope-theta 10000 --device mps --temperature 0.0 --top-p 0.9 --max-length 256
```