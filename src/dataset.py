# Python packages
from termcolor import colored
from tqdm import tqdm
import os
import tarfile
import wget # for downing data from site

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Custom packages
import src.config as cfg


class TinyImageNetDatasetModule(LightningDataModule):
    __DATASET_NAME__ = 'tiny-imagenet-200'

    def __init__(self, batch_size: int = cfg.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        if not os.path.exists(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__)):
            # download data
            print(colored("\nDownloading dataset...", color='green', attrs=('bold',)))
            filename = self.__DATASET_NAME__ + '.tar'
            wget.download(f'https://hyu-aue8088.s3.ap-northeast-2.amazonaws.com/{filename}')

            # extract data
            print(colored("\nExtract dataset...", color='green', attrs=('bold',)))
            with tarfile.open(name=filename) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(path=cfg.DATASET_ROOT_PATH, member=member)
            os.remove(filename)

    def train_dataloader(self):
        from collections import Counter
        from torch.utils.data import Subset

        tf_train = transforms.Compose([
            # augmented
            transforms.RandomRotation(cfg.IMAGE_ROTATION), 
            transforms.RandomHorizontalFlip(cfg.IMAGE_FLIP_PROB),
            transforms.RandomCrop(cfg.IMAGE_NUM_CROPS, padding=cfg.IMAGE_PAD_CROPS),
            # 
            transforms.ToTensor(),
            # normalization for good optimization
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'train'), tf_train)
        
        # # 수정 시작
        # # Step 1. 제외할 label 지정
        # # excluded_labels = [141, 138, 24, 120, 65, 160]
        # excluded_labels = [141]

        # # Step 2. 제거 전 전체 라벨 분포 확인
        # full_label_count = Counter(dataset.targets)
        # print(colored("\n[Before Filtering] Label Distribution:", 'cyan', attrs=('bold',)))
        # for label in sorted(full_label_count.keys()):
        #     print(f"Label {label:3d}: {full_label_count[label]:5d} samples")

        # print(colored("\n[Labels to Exclude]", 'magenta', attrs=('bold',)))
        # for label in excluded_labels:
        #     print(f"Label {label:3d}: {full_label_count.get(label, 0)} samples")

        # # Step 3. 제외하지 않을 인덱스 리스트 구성
        # included_indices = [i for i, label in enumerate(dataset.targets) if label not in excluded_labels]

        # # Step 4. Subset 구성
        # dataset = Subset(dataset, included_indices)

        # # Step 5. 제거 후 라벨 분포 확인
        # targets_in_subset = [dataset.dataset.targets[i] for i in dataset.indices]
        # subset_label_count = Counter(targets_in_subset)

        # print(colored("\n[After Filtering] Label Distribution:", 'green', attrs=('bold',)))
        # for label in sorted(subset_label_count.keys()):
        #     print(f"Label {label:3d}: {subset_label_count[label]:5d} samples")

        # print(colored("\n[Check Excluded Labels]", 'yellow', attrs=('bold',)))
        # for label in excluded_labels:
        #     if label in subset_label_count:
        #         print(f"⚠️ Label {label} still present with {subset_label_count[label]} samples")
        #     else:
        #         print(f"✅ Label {label} successfully excluded.")
        # msg = f"[Train] root dir: {dataset.dataset.root} | # of samples: {len(dataset):,}"
        # # 수정 끝
        
        msg = f"[Train]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader( 
            dataset,
            shuffle=True, # mix every epoch
            pin_memory=True, # for efficiency
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size, 
        )

    def val_dataloader(self):
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'val'), tf_val)
        msg = f"[Val]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'test'), tf_test)
        msg = f"[Test]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )
