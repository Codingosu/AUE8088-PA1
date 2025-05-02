import os

# Training Hyperparameters
NUM_CLASSES         = 200
# BATCH_SIZE          = 512
BATCH_SIZE = 256
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'resnet18'
# MODEL_NAME = 'AlexNet_copy'
# MODEL_NAME = 'Mini_AlexNet'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'

'''
python test.py --ckpt_file wandb/aue8088-pa1/파일명/checkpoints/epoch=숫자-step=숫자.ckpt
[TODO] 2. try different setting
default model : zi794byi
python test.py --ckpt_file wandb/aue8088-pa1/zi794byi/checkpoints/epoch=38-step=7644.ckpt

default model + batch size (512 -> 128) : bb65nh7v
python test.py --ckpt_file wandb/aue8088-pa1/bb65nh7v/checkpoints/epoch=37-step=59394.ckpt

default model + optimizer momentum (0.9 -> 0) : of9viqy1
python test.py --ckpt_file wandb/aue8088-pa1/of9viqy1/checkpoints/epoch=35-step=7056.ckpt

[TODO] 3. Alexnet
Alexnet defualt model (Alexnet_copy) : xhsw8o72
python test.py --ckpt_file wandb/aue8088-pa1/xhsw8o72/checkpoints/epoch=39-step=7840.ckpt

Mini_AlexNet : 5eqm1b1h
python test.py --ckpt_file wandb/aue8088-pa1/5eqm1b1h/checkpoints/epoch=38-step=7644.ckpt

[TODO] 4. size-accuracy trade-off plot
resnet family (18, 34는 BATCH_SIZE = 512, 50부터는 BATCH_SIZE = 256)
resnet18 (11.3 M) : gn4tz3uc
python test.py --ckpt_file wandb/aue8088-pa1/gn4tz3uc/checkpoints/epoch=39-step=7840.ckpt
Accuracy/val : 0.370  / Flop : 296.296 M 

resnet34 (21.4 M) : zr7el75n
python test.py --ckpt_file wandb/aue8088-pa1/zr7el75n/checkpoints/epoch=30-step=6045.ckpt
Accuracy/val : 0.339  / Flop : 598.286 M 

resnet50 (23.9 M) : a0wesj4u
python test.py --ckpt_file wandb/aue8088-pa1/a0wesj4u/checkpoints/epoch=30-step=12121.ckpt
Accuracy/val : 0.317  / Flop : 668.107 M 

resnet101 

resnet152

[TODO] 5. state-of-the-art
resnet18 (label [141, 138, 24, 120, 65, 160] 제외) : 2eybua6a
python test.py --ckpt_file wandb/aue8088-pa1/2eybua6a/checkpoints/epoch=36-step=7030.ckpt

resnet18 (label [141] 제외) : o2h5su92
python test.py --ckpt_file wandb/aue8088-pa1/o2h5su92/checkpoints/epoch=35-step=7020.ckpt

'''