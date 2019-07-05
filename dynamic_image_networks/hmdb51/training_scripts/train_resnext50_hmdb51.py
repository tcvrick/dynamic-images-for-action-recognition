# import apex - !!!! INCLUDE THIS IMPORT IF YOU WANT TO USE MIXED PRECISION TRAINING !!!!
import torch
import os
import sys
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# Make sure that the project root is in your PATH (i.e., the parent folder containing 'dynamic_image_networks').
sys.path.append(str(Path('../../..').resolve()))

# ---------------------------------------------------------------
# Model / dataset choice
# ---------------------------------------------------------------
from dynamic_image_networks.hmdb51.models.resnext50_temppool import get_model
from dynamic_image_networks.hmdb51.dataloaders.hmdb51_dataloader import get_train_loader
from dynamic_image_networks.hmdb51.utilities.calculate_training_metrics import calculate_accuracy
from dynamic_image_networks.hmdb51.utilities.logger import initialize_logger
from dynamic_image_networks.hmdb51.utilities.meters import AverageMeter


def main():
    # ============================================================================================
    # Setup
    # ============================================================================================
    # ---------------------------------------------------------------
    # Random seeds
    # ---------------------------------------------------------------
    torch.manual_seed(590238490)
    torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------------
    # GPU
    # ---------------------------------------------------------------
    device = torch.device("cuda:0")
    fp16 = False
    if fp16:
        print('!!! MIXED PRECISION TRAINING IS ENABLED -- ONLY USE FOR VOLTA AND TURING GPUs!!!')

    # ---------------------------------------------------------------
    # Training settings
    # ---------------------------------------------------------------
    batch_size = 32
    num_epochs = 60
    num_workers = 6
    max_segment_size = 10
    save_best_models = True
    image_augmentation = False

    # ----------------------------------------------------------------------------
    # Get the model
    # ----------------------------------------------------------------------------
    net = get_model(num_classes=51)
    net.to(device)

    # ----------------------------------------------------------------------------
    # Initialize optimizer and loss function
    # ----------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    if fp16:
        net, optimizer = apex.amp.initialize(net, optimizer, opt_level="O1")

    # ---------------------------------------------------------------
    # Logging set-up
    # ---------------------------------------------------------------
    # File-name
    file_name = ''.join(os.path.basename(__file__).split('.py')[:-1])
    logger = initialize_logger(file_name, log_dir='./logs/')

    # ============================================================================================
    # Train
    # ============================================================================================
    time_start = datetime.now()
    fold_i = 1

    # ---------------------------------------------------------------
    # Load dataloaders
    # ---------------------------------------------------------------
    train_loader, validation_loader = get_train_loader(fold_id=fold_i,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers,
                                                       image_augmenation=image_augmentation,
                                                       segment_size=max_segment_size)

    logger.info('Starting Training on Fold: {}\n'.format(fold_i))

    best_val_loss = float('inf')
    best_val_acc = 0
    for epoch_i in range(num_epochs):
        # ---------------------------------------------------------------
        # Training and validation loop
        # ---------------------------------------------------------------

        avg_loss, avg_acc = training_loop('train', net, device, train_loader,
                                          optimizer, criterion, fp16)

        avg_val_loss, avg_val_acc = training_loop('val', net, device, validation_loader,
                                                  None, criterion, fp16)

        if scheduler:
            scheduler.step(avg_val_loss)

        # ---------------------------------------------------------------
        # Track the best model
        # ---------------------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            if save_best_models:
                logger.info('Saving model because of best loss...')
                os.makedirs('./saved_models/', exist_ok=True)
                torch.save(net.state_dict(),
                           './saved_models/{}_fold_{}_best_loss_state.pt'.format(file_name, fold_i))

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc

            if save_best_models:
                logger.info('Saving model because of best acc...')
                os.makedirs('./saved_models/', exist_ok=True)
                torch.save(net.state_dict(),
                           './saved_models/{}_fold_{}_best_acc_state.pt'.format(file_name, fold_i))

        # ---------------------------------------------------------------
        # Log the training status
        # ---------------------------------------------------------------
        time_elapsed = datetime.now() - time_start
        output_msg = 'Fold {}, Epoch: {}/{}\n' \
                     '---------------------\n' \
                     'train loss: {:.6f}, val loss: {:.6f}\n' \
                     'train acc: {:.6f}, val acc: {:.6f}\n' \
                     'best val loss: {:.6f}, best val acc: {:.6f}\n' \
                     'time elapsed: {}\n'. \
            format(fold_i, epoch_i, num_epochs - 1,
                   avg_loss, avg_val_loss,
                   avg_acc, avg_val_acc,
                   best_val_loss, best_val_acc,
                   str(time_elapsed).split('.')[0])
        logger.info(output_msg)

    logger.info('Finished Training')


def training_loop(phase, net, device, dataloader, optimizer, criterion, fp16):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # Set the model into the appropriate mode.
    if phase == 'train':
        net.train()
    elif phase == 'val':
        net.eval()
    else:
        raise ValueError

    # Enable gradient accumulation only for the training phase.
    with torch.set_grad_enabled(phase == 'train'):
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y, = data
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Prediction.
            y_pred = net(x).float()

            # Loss and step.
            loss = criterion(y_pred, y)
            if phase == 'train':
                optimizer.zero_grad()
                if fp16 is True:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

            # Metrics
            batch_size = len(y)
            loss_meter.add(loss.item(), batch_size)
            acc_meter.add(calculate_accuracy(y_pred, y), batch_size)

    avg_loss = loss_meter.get_average()
    avg_acc = acc_meter.get_average()
    return avg_loss, avg_acc


if __name__ == '__main__':
    main()
