# Python Dynamic Images for Action Recognition

This section contains the implementation of the Dynamic Image Networks discussed in
Bilen et al. in PyTorch. This implementation achives approximately 47.5% accuracy, whereas Bilen et al. reports
57% accurracy using a similar technique.

## Installation
Install the pre-requisites modules:

~~~
dynamicimage
pytorch / torchvision
pandas
opencv
numpy
tqdm
imgaug
apex (optional)
~~~


## Preprocessing

1. Download the HMDB51 dataset (including the test splits) and extract them to your working directory.
2. Modify the three scripts in the ```dynamic_image_networoks/preprocessing``` directory to point to the location
of your HMDB51 dataset. 
3. Run the preprocessing scripts. These scripts will parse the HMDB51 dataset to generate a metadata file, 
extract the video frames, and precompute the dynamic images.

## Training
1. Modify the ```dynamic_image_networoks/dataloaders/hmdb51_dataloader.py``` script to point to your preprocessed
dynamic images.
2. Run any of training scripts located in the ```dynamic_image_networoks/training_scripts``` directory. 
3. If you want to make further changes to the models or the dataloaders, the relevant files are located
in the ```dynamic_image_networoks/models``` and ```dynamic_image_networoks/dataloaders```  directories. The training scripts
are designed to be modular with different models and dataloaders with minimal changes.