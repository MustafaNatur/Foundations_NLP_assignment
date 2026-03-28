# Assignment: Training and Evaluating a Small Transformer Language Model

**Course:** Foundations of NLP

---

## Overview

In this assignment you will train a small character-level Transformer language model from scratch using [nanoGPT](https://github.com/karpathy/nanoGPT), a minimal and readable GPT implementation. You will experiment with different hyperparameters, evaluate the model both quantitatively and qualitatively, and report your findings in a structured write-up with plots and tables.

### Learning Objectives

- Understand the training loop of a Transformer language model in practice.
- Develop intuition for the effect of hyperparameters (learning rate, model size, context length, etc.) on training dynamics and generation quality.
- Practice systematic experimental methodology: controlled comparisons, proper logging, and clear reporting.

---

## 1. Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
```

### 1.2 Install Dependencies

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib
```

> This works on Windows, macOS, and Linux. A GPU is not required — all experiments in this assignment are designed to run on a laptop CPU in reasonable time. If you have access to a GPU (including free-tier Google Colab or Kaggle), training will be faster but is not necessary.

### 1.3 Prepare the Dataset

We will use the **Tiny Shakespeare** dataset (a ~1MB corpus of Shakespeare's works):

```bash
cd data/shakespeare_char
python prepare.py
cd ../..
```

This creates `train.bin` and `val.bin` in the data directory.

---

## 2. Baseline Training Run

Train a small character-level GPT with the following baseline configuration. Create a file called `config/train_shakespeare_char_baseline.py`:

```python
# Baseline configuration
out_dir = 'out-shakespeare-baseline'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
wandb_log = False
wandb_project = 'nanoGPT-assignment'
wandb_run_name = 'baseline'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
weight_decay = 1e-1

device = 'cpu'   # change to 'cuda' if you have a GPU
compile = False  # set True only on Linux with GPU
```

**Run the baseline:**

```bash
python train.py config/train_shakespeare_char_baseline.py
```

> Training should take approximately 10–20 minutes on a modern laptop CPU.

**Generate text from the trained model**

```bash
python sample.py --out_dir=out-shakespeare-baseline \
    --device=cpu --num_samples=3 --max_new_tokens=500
```

Save the generated samples to a text file for your report.

---

## 3. Hyperparameter Experiments

Design and run **at least four additional experiments**, varying one hyperparameter at a time relative to the baseline. You must cover at least **three** of the following categories:

| Category | Hyperparameter | Suggested Values to Try |
|---|---|---|
| Learning rate | `learning_rate` | `1e-4`, `5e-4`, `1e-3` (baseline), `5e-3`, `1e-2` |
| Model depth | `n_layer` | `2`, `4`, `6` (baseline), `8` |
| Model width | `n_embd` (`n_head`) | `128` (4 heads), `256` (4 heads), `384` (6 heads, baseline) |
| Context length | `block_size` | `64`, `128`, `256` (baseline) |
| Regularisation | `dropout` | `0.0`, `0.1`, `0.2` (baseline), `0.4` |
| Training duration | `max_iters` | `1000`, `2500`, `5000` (baseline), `10000` |

For each experiment, create a separate config file (e.g., `config/train_shakespeare_char_lr5e4.py`) and a separate `out_dir` so that results are not overwritten.

> **Important:** Keep all other hyperparameters fixed when varying one. Record the exact configuration for every run.

---

## 4. Evaluation and Reporting

### 4.1 Quantitative Evaluation

For each run, record:

- Final training loss
- Final validation loss
- Training time (wall-clock)

Present these in a summary table like the one below:

| Experiment | LR | Layers | Embd | Block | Dropout | Iters | Train Loss | Val Loss | Time (min) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 1e-3 | 6 | 384 | 256 | 0.2 | 5000 | ... | ... | ... |
| Low LR | 1e-4 | 6 | 384 | 256 | 0.2 | 5000 | ... | ... | ... |
| ... | | | | | | | | | |

### 4.2 Loss Curves

Create a Python script that:

- Parses the training logs for each run (nanoGPT prints step, train loss, val loss to stdout).
- Plots **training loss vs. step** for all runs in a single figure (one line per run, with a legend).
- Plots **validation loss vs. step** in a second figure.

Save these as `.png` or `.pdf` files.

**Tip:** Redirect output to a log file during training:

```bash
python train.py config/train_shakespeare_char_baseline.py | tee logs/baseline.log
```

### 4.3 Qualitative Evaluation

For each experiment, generate at least 3 text samples (500 tokens each) and include them in an appendix. In the main body of your report, discuss:

- Which configurations produce the most coherent or "Shakespeare-like" text?
- At what point do you see clear signs of **underfitting** (e.g., gibberish output)?
- Do you observe any signs of **overfitting** (low train loss, high val loss, repetitive output)?
- Is there a noticeable difference between models with different context lengths?

### 4.4 Analysis Questions

Answer the following in your write-up (2–4 sentences each):

1. What happens when the learning rate is too high? Too low? How does this manifest in the loss curve and the generated text?
2. What is the relationship between model size (layers/embedding) and validation loss? Is bigger always better at this scale and dataset size?
3. What role does dropout play? Compare the train/val loss gap across dropout values.
4. Based on your experiments, what configuration would you recommend for this dataset, and why?

---

## 5. Bonus (Optional)

Choose one or both of the following:

**A. Custom Dataset**
Train on a custom dataset of your choice (minimum 500KB of text). Describe the dataset, any preprocessing you did, and compare sample quality with Shakespeare.

**B. BPE Tokeniser**
Replace the character-level tokeniser with your own Byte Pair Encoding (BPE) tokeniser. Train it on the Shakespeare corpus (or your custom dataset), integrate it into the nanoGPT data pipeline, and compare training dynamics and generation quality against the character-level baseline.

---

## Deliverables

Submit a **single ZIP file** containing:

- **Report** (PDF, max 6 pages excluding appendix): summary table, loss curve plots, qualitative analysis, and answers to the analysis questions.
- **Notebook** (`.ipynb`): a single Jupyter/IPython notebook containing all code for training runs, log parsing, plotting, and sample generation. The notebook should be runnable and clearly organised with markdown cells explaining each section.

---

## Grading Rubric

| Component | Points |
|---|---|
| Setup and baseline run completed correctly | 10 |
| At least 4 well-designed experiments (controlled, one variable at a time) | 20 |
| Summary table with complete results | 10 |
| Loss curve plots (clear, labelled, readable) | 15 |
| Qualitative analysis of generated text | 15 |
| Answers to analysis questions (depth and correctness) | 20 |
| Code quality and reproducibility | 10 |
| **Total** | **100** |
| Bonus | up to +10 |

---

## Tips

- **Redirect logs.** nanoGPT prints to stdout. Use `| tee logs/run_name.log` (Linux/Mac) or `> logs\run_name.log 2>&1` (Windows CMD) to save output.
- **Start early.** Each run takes 10–20 min on CPU. With 5+ runs, you need at least 1–2 hours of compute time.
- **Use Colab/Kaggle as a fallback.** If your laptop is too slow, both offer free GPU access. Set `device = 'cuda'` and `compile = True` in your config.
- **Do not change the tokeniser** (unless attempting Bonus B). The character-level tokeniser is part of the `prepare.py` script and should stay fixed across experiments.
