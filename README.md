# EureQA: Extractive Question Answering model

## Overview

EureQA is an Extractive Question Answering model based on the [ALBERT Base v2](https://huggingface.co/albert/albert-base-v2) architecture and finetuned on [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) dataset.

## Install

1. Install [miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).

2. Create conda environment.

```bash
conda env create -f environment.yml
```

3. Activate the `eureqa` environment.

```bash
conda activate eureqa 
```

## Run gradio demo

1. Run the following command in the `eureqa` environment.

```bash
python app.py
```

2. Open the address in the console.

Alternatively, you can try online demo at [Huggingface Spaces](https://huggingface.co/spaces/ThreeBlessings/eureqa).


## Model description

EureQA is a transformers model, based on on ALBERT and fine-tuned on an extractive question answering dataset SQuAD 2.0. This means it was fine-tuned to answer questions given some context. It can also detect if there is no answer in the context to the provided question.

Since EureQA is based on ALBERT, it shares its layers across its Transformer, meaning all layers have the same weights. Using repeating layers results in a small memory footprint, however, the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.

This model has the following configuration:

- 12 repeating layers
- 128 embedding dimension
- 768 hidden dimension
- 12 attention heads
- 11M parameters

## Intended uses & limitations

You can use this model for the extractive question answering task on your data. Since the model was finetuned on SQuAD 2.0 dataset, it might perform worse on OOD data.

## Training and evaluation data

The model was trained and evaluated on the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

## Training procedure

Training is performed using the `run_qa.py` script from [huggingface examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering):

```bash
python run_qa.py \
    --model_name_or_path albert/albert-base-v2 \    # Pretrained model identifier on the Huggingface model hub
    --dataset_name squad_v2 \                       # Dataset name on the Huggingface datasets hub
    --version_2_with_negative \                     # Indicates the answer might be absent in the context
    --do_train \                                    # Perform model training
    --num_train_epochs 2 \                          # Number of training epochs
    --per_device_train_batch_size 12 \              # Train batch size
    --learning_rate 3e-5 \                          # Learning rate
    --max_seq_length 384 \                          # Maximum sequence length (in tokens) to pass to the model. This includes both question and context. Longer sequences are truncated with the overflow becoming the context of another sequence
    --doc_stride 128 \                              # Stride to use when splitting long sequences into several ones
    --bf16 True \                                   # Use bf16 during training. This can speedup the training when using some GPUs
    --torch_compile \                               # Compile the model with `torch.compile`. This can speedup the training and inference of the model
    --do_eval \                                     # Perform model evaluation
    --per_device_eval_batch_size 64 \               # Evaluation batch size. Since we don't store the gradients during evaluation, it can be larger that train batch size
    --eval_strategy "steps" \                       # Evaluate the model each N steps
    --eval_steps 2000 \                             # Steps at which the model should be evaluated
    --eval_on_start True \                          # Evaluate the model before the training (evaluates the pretrained model with initialized head)
    --bf16_full_eval \                              # Use bf16 during evaluation
    --logging_steps 2000 \                          # Steps at which the logging should be performed
    --save_steps 2000                               # Steps at which the model should be saved
    --output_dir ./tmp/debug_squad2/                # Directory to save the model and model info to
```

### Training hyperparameters

The following hyperparameters were used during training:
- Learning rate: 3e-05
- Train batch size: 12
- Eval batch size: 64
- Seed: 42
- Optimized: Adam with betas=(0.9,0.999) and epsilon=1e-08
- Learning rate scheduler: linear
- Epochs: 2.0

### Training results

Training loss is 0.9652.

| Metric category          | Exact Match | F1 score |
|--------------------------|-------------|----------|
| Answer in the context    | 75.0        | 81.4     |
| No answer in the context | 82.1        | 82.1     |
| Overall                  | 78.6        | 81.8     |

Comparison with the current SOTA single models (base variants)

| Model | Exact Match | F1 score |
|-|-|-|
| [BERT](https://github.com/google-research/BERT) | 73.7 | 76.3 |
| [RoBERTa](https://arxiv.org/pdf/1907.11692) | 80.5 | 83.7 |
| [ELECTRA](https://huggingface.co/deepset/electra-base-squad2) | 80.5 | 83.3 |
| [DeBERTa v3](https://github.com/microsoft/DeBERTa) | 85.4 | 88.4 |
| [ALBERT](https://github.com/google-research/ALBERT) (original) | 79.3 | 82.1 |
| **EureQA (ours)** | **78.6** | **81.8** |

### Training log

You can find training log on [ClearML](https://app.clear.ml/projects/cd2f4008afa34a68bd085588fe8f44e1/experiments/a971b54e499a4dbe8b90faf9b6969608/output/execution) (you must be registered on clear.ml).
