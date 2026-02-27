# Learning rate parameters
BASE_LR = 0.001
# number of epochs after which the Learning rate is decayed exponentially.
EPOCH_DECAY = 30
DECAY_WEIGHT = 0.1  # factor by which the learning rate is reduced.


# DATASET INFO
NUM_CLASSES = 9  # set the number of classes in your dataset
# to run with the sample dataset, just set to 'hymenoptera_data'
DATA_DIR = 'RODI-DATA_split'

# DATALOADER PROPERTIES
# Set as high as possible. If you keep it too high, you'll get an out of memory error.
BATCH_SIZE = 64


# GPU SETTINGS
# Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
CUDA_DEVICE = "mps"
GPU_MODE = 1  # set to 1 if want to run on gpu.


# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 0  # if you want to use tensorboard set this to 1.
TENSORBOARD_SERVER = "YOUR TENSORBOARD SERVER ADDRESS HERE"  # If you set.
# if using tensorboard, enter name of experiment you want it to be displayed as.
EXP_NAME = "fine_tuning_experiment"
