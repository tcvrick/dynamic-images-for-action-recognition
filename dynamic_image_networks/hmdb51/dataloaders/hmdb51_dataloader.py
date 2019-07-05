import cv2
import torch
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa

from pathlib import Path
from datetime import datetime
from typing import Tuple
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


""" 
Wrapper function used to initialize the data-loaders needed for training, validation and testing.
"""


def get_train_loader(fold_id: int,
                     batch_size: int,
                     num_workers: int,
                     image_augmenation: bool,
                     segment_size: int) -> Tuple['DataLoader', 'DataLoader']:
    # --------------------------------------------------------------------------------
    # Load information from data-frame
    # --------------------------------------------------------------------------------
    metadata = pd.read_csv(f'../preprocessing/hmdb51_metadata_split_{fold_id}.csv')

    # Create an additional row which contains the image paths of the dynamic images.
    image_paths = [str(Path(r'E:\hmdb51_org\multiple_dynamic_images') / row['category_name'] / row['name'])
                   for _, row in metadata.iterrows()]
    metadata['image_path'] = image_paths

    # Create the dataframes for the training and validation sets.
    train_df = metadata[metadata['training_split'] == True].reset_index(drop=True)
    val_df = metadata[metadata['training_split'] == False].reset_index(drop=True)

    # Random samplers.
    train_sampler = SubsetRandomSampler(range(len(train_df)))
    valid_sampler = SubsetRandomSampler(range(len(val_df)))

    # Split the videos into training and validation.
    train_imgs = train_df['image_path'].copy()
    train_labels = train_df['category'].copy()
    val_imgs = val_df['image_path'].copy()
    val_labels = val_df['category'].copy()

    # Form the training / validation set accordingly.
    train_dataset = ImageDataset(train_imgs, train_labels, segment_size, data_aug=image_augmenation)
    validation_dataset = ImageDataset(val_imgs, val_labels, segment_size, data_aug=False)

    # Form the training / validation data loaders.
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=False,
                              worker_init_fn=worker_init_fn,)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   sampler=valid_sampler,
                                   num_workers=0,
                                   pin_memory=False,)

    return train_loader, validation_loader


def worker_init_fn(worker_id):
    random_seed = datetime.now().microsecond + datetime.now().second + worker_id
    print('WORKER [{}] START'.format(worker_id), random_seed)
    np.random.seed(random_seed)


# --------------------------------------------------------------------------------
# Dataset Definition
# --------------------------------------------------------------------------------


class ImageDataset(data.Dataset):

    def __init__(self, image_paths, labels, segment_size, data_aug=False):
        # Settings
        self.image_paths = image_paths
        self.labels = torch.tensor(labels).float()
        self.data_aug = data_aug
        self.segment_size = segment_size

        # Data augmentation.
        self.data_aug_pipeline = iaa.Sequential([
            iaa.Sometimes(1.00, [
                iaa.Affine(
                    scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
                    translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                    rotate=(-15, 15),
                    shear=(-10, 10)
                ),
            ]),
            iaa.Fliplr(0.5),

        ])

        assert len(image_paths) == len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Load the first N dynamic images (precomputed using window size 10 and stride 6).
        frames = list(Path(self.image_paths[index]).glob('*.jpg'))
        frames = frames[:self.segment_size]
        frames = np.array([cv2.imread(str(x)) for x in frames])

        # Process each dynamic image independently.
        processed_images = []
        for image in frames:
            # Data augmentation.
            if self.data_aug is True:
                self.data_aug_pipeline.augment(images=image)

            # Convert BGR -> RGB, HWC -> CHW, and from NumPy to Torch tensors.
            image = image[..., ::-1] - np.zeros_like(image)
            image = image.transpose((2, 0, 1))
            image = torch.tensor(image).float() / 255.0
            processed_images.append(image)

        # Zero pad the sequence of images.
        while len(processed_images) < self.segment_size:
            processed_images.append(torch.zeros_like(processed_images[0]))

        # Collate the sequence of images.
        processed_images = torch.stack(processed_images)
        return processed_images.float(), self.labels[index].long()


def main():
    # Visualize the dataloader for debug purposes.
    train_dataloader, val_dataloader = get_train_loader(fold_id=1, batch_size=5, num_workers=0, image_augmenation=False,
                                                        segment_size=10)
    for j, data in enumerate(train_dataloader):
        for sample in data:
            for n in range(sample[0].size(0)):
                img = sample[0][n].numpy() * 255
                img = img.transpose((1, 2, 0))
                img = img[..., ::-1]
                label = sample[1].numpy()

                print('Label:', label)
                cv2.imshow('', img.astype(np.uint8))
                cv2.waitKey()


if __name__ == '__main__':
    main()
