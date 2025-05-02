"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python train.py
        - For better flexibility, consider using LightningCLI in PyTorch Lightning
"""
# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule # import image, if image exist X -> down from site (prof.hwang)
from src.network import SimpleClassifier # 
import src.config as cfg # hyper parameter setting

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":

    # default : resnet18 in network.py
    model = SimpleClassifier(
        model_name = cfg.MODEL_NAME,
        num_classes = cfg.NUM_CLASSES,
        optimizer_params = cfg.OPTIMIZER_PARAMS,
        scheduler_params = cfg.SCHEDULER_PARAMS,
    )
    
    total_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        shape_str = str(tuple(param.shape))           # tuple → str
        print(f"{name:40s} {shape_str:15s}  params={n}")
        
        total_params += n
    print(f"\nTotal parameters: {total_params:,}")

    datamodule = TinyImageNetDatasetModule(
        batch_size = cfg.BATCH_SIZE,
    )

    wandb_logger = WandbLogger(
        project = cfg.WANDB_PROJECT,
        save_dir = cfg.WANDB_SAVE_DIR,
        entity = cfg.WANDB_ENTITY,
        name = cfg.WANDB_NAME,
    )

    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        max_epochs = cfg.NUM_EPOCHS,
        check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
        logger = wandb_logger,
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
            # 가장 model이 좋을 때 보관
        ],
    )

    # NN train & save the best NN model that makes the best accuracy from validation
    trainer.fit(model, datamodule=datamodule)
    
    # validate at tge best NN model
    trainer.validate(ckpt_path='best', datamodule=datamodule)


