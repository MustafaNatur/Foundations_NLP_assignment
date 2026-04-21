# Pride and Prejudice baseline configuration
out_dir = '../Weights/out-pride-prejudice-baseline'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanoGPT-assignment'
wandb_run_name = 'pride_prejudice_baseline'

dataset = 'pride_prejudice_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 1500
lr_decay_iters = 1500
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
weight_decay = 1e-1

device = 'mps'    # Apple Silicon MPS backend for the reduced bonus rerun
compile = False   # keep False for MPS
