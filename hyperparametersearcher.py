# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import math
import os
# import shutil
# import sys
import random
import re
import time
# import platform
# import psutil
import winsound
from PIL import Image
# from matplotlib.ticker import NullFormatter
# noinspection PyUnresolvedReferences
from matplotlib import cm
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wx
import wx.adv
import wx.lib.newevent
import wx.lib.scrolledpanel as scrolled

from matplotlib.collections import PolyCollection
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from mpl_toolkits.mplot3d import axes3d

# Copyright by Chester Lau
#
# Program to train and recognize baby faces on pictures.
# The labels of the training and testing pictures are at the end of the filename with the
# format of 12 bytes following a "_". The first four bytes are babies. Subsequent bytes
# are father, mother, fracturnal grandparents, maturnal grandparents, baby sitter, and one other adult.
# from matplotlib.mlab import bivariate_normal
# from tensorflow.python.framework import ops

# use DEBUG, INFO, WARN, ERROR, and FATAL to denote different level messages during run and debug
# tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Hyperparameters

# Input parameters
TRAIN_DATA_DIR = r"e:\Chester\BabyPictureTrainingData"
TEST_DATA_DIR = r"e:\Chester\BabyPictureTestingData"
COMPRESSED_TRAIN_DATA_DIR = r"e:\Chester\CompressedBabyPictureTrainingData"
COMPRESSED_TEST_DATA_DIR = r"e:\Chester\CompressedBabyPictureTestingData"
RBG_TRAIN_DATA_DIR = r"e:\Chester\RBGBabyPictureTrainingData"
RBG_TEST_DATA_DIR = r"e:\Chester\RBGBabyPictureTestingData"
TAG_DIR = r"e:\Chester\BabyPictureTrainLog\Tag"
# TENSORBOARD_LOG_DIR = r"c:\Users\Chester\BabyPictureTrainLog(test)"
TENSORBOARD_LOG_DIR = r"f:\hyperparameter_training"
HYPERPARAMETER_LOG_DIR = r"f:\hyperparameter_training"
TRAIN_LABEL_DIR = r"e:\Chester\LabelBabyPictureTrainingData"
TEST_LABEL_DIR = r"e:\Chester\LabelBabyPictureTestingData"
SAVE_MODEL_FILE = r"e:\Chester\BabyPictureTrainLog\model.ckpt"
WORKING_DIR = r"e:\Chester\PythonSourceCodes\HyperparameterSearcher"

SCREEN_SIZE = (1200, 500)

CONVOLUTION_LAYER = 0
MULTI_NEURON_LAYER = 1
RECURSIVE_LAYER = 2
LONG_SHORT_TERM_MEMORY_LAYER = 3
SIMOID_LAYER = 4
CROSS_ENTROPY_LAYER = 5
CROSS_ENTROPY_SIGMOID_LAYER = 6
LEARNING_LAYER = 7
NUMBER_OF_STEPS_LAYER = 8
CUSTOM_LAYER = 99

STATIC_BOX = 0
SPIN_CTRL_DOUBLE = 1
CHOICE_CTRL = 2
DISPLAY_CTRL = 3

TITLE_HOLDER = 3
STARTING_HOLDER = STARTING_CHOICE_HOLDER = 5
ENDING_HOLDER = ENDING_CHOICE_HOLDER = 6
STEP_HOLDER = STEP_CHOICE_HOLDER = 7

NUMBER_CONVOLUTION_LAYERS = 2
NUMBER_MULTI_NEURON_LAYERS = 3
NUMBER_SIGMOID_LAYERS = 1
NUMBER_LEARNING_LAYERS = 1
NUMBER_CROSS_ENTROPY_LAYERS = 1
NUMBER_STEPS_LAYER = 1
NUMBER_CROSS_ENTROPY_SIGMOID_LAYERS = 1

NUMBER_THREAD = 1
BATCH_SIZE = 16
NUMBER_ACCURACY_IMAGES = 16
MAX_NUMBER_IN_TRAIN_QUEUE = 3
MAX_NUMBER_IN_VALIDATE_QUEUE = 3
MAX_NUMBER_IN_TEST_QUEUE = 3
NUMBER_PICTURES_PER_EPOCH_FOR_TRAIN = 1600  # it was 1600
NUMBER_PICTURES_PER_EPOCH_FOR_TEST = 1000  # Need to change before testing; it was 100
MIN_FRACTION_OF_PICTURES_IN_QUEUE = 0.4
NUMBER_STEPS_PER_EPOCH = 1000        # it was 1000
NUMBER_OF_EPOCHES = 1                     # this is temporary; it needs to be changed into a global variable

NUMBER_OF_STEPS_PER_REPORT = 20

NUMBER_POSSIBLE_FACES = 12
NUMBER_FOLDS = 15
NUMBER_VALIDATE_FOLDS = 3
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
# OUT_CHANNELS = 3     #number of filters in the convolution layer

# for now it is loss, validate image accuracy, validate category accuracy, dbl check images accuracy,
#  dbl check category accuracy, sensitivities, and specificities for each faces.
# Validate images accuaracy and validate category accuracy will be dropped off after debugging.
# NUMBER_OF STATS will become 29
NUMBER_OF_STATS = 31

LOSS = 0
IMAGE_ACCURACY = 1
POSITIVE_FACE_ACCURACY = 2
NEGATIVE_FACE_ACCURACY = 3
TOTAL_FACE_ACCURACY = 4
TENSORFLOW_IMAGE_ACCURACY = 5
TENSORFLOW_FACE_ACCURACY = 6
SEN1 = 101
SEN2 = 102
SEN3 = 103
SEN4 = 104
SEN5 = 105
SEN6 = 106
SEN7 = 107
SEN8 = 108
SEN9 = 109
SEN10 = 110
SEN11 = 111
SEN12 = 112
SP1 = 201
SP2 = 202
SP3 = 203
SP4 = 204
SP5 = 205
SP6 = 206
SP7 = 207
SP8 = 208
SP9 = 209
SP10 = 210
SP11 = 211
SP12 = 212

THREE_D_WIRE_FRAME = 0
THREE_D_SURFACE = 1
THREE_D_CONTOUR_PLOT = 2
THREE_D_WIRE_GRAPH_BY_EPOCH_NUMBER = 3
THREE_D_WIRE_GRAPH_BY_STEPS = 4
THREE_D_BAR = 5
THREE_D_POLYGON = 6
# TWO_D_BAR = 7
TWO_D_LINES = 7
# TWO_D_CUMULATIVE = 8
TWO_D_LOG = 8
TWO_D_BROKEN_Y_AXIS = 9

FLOATING_POINT_PRECISION = tf.float32
TOWER_NAME = 'tower'
SHRINK = True
VERBOSE = True

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor. Iniitally was 0.1
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate. Initially was 0.1

# Preprocessing parameters
MIN_FRACTION_IN_QUEUE = 0.4
SHUFFLE = True
FLIP = True
ALTER_BRIGHTNESS = True
ALTER_CONTRAST = True
MAXIMUM_BRIGHTNESS_DELTA = 63
UPPER_CONTRAST_CHANGE = 1.6
LOWER_CONTRAST_CHANGE = 0.4

# ASCII codes
ASCII_BACKSPACE = 8
ASCII_ENTER = 13
ASCII_0 = 48
ASCII_9 = 57
ASCII_DELETE = 127
ASCII_HOME = 312
ASCII_ARROW_RIGHT = 316
ASCII_ARROW_DOWN = 317
ASCII_INSERT = 322
ASCII_PAGEUP = 366
ASCII_PAGEDOWN = 377
ASCII_PERIOD = 46
ASCII_MINUS = 45


# Layer parameters
XAVIER = True  # The initial weights and bias for the layers will be ignore with XAVIER being true
# Conv1 hyperparameters
CNN1_WEIGHT = 4.0
CNN1_BIAS = 0.1
CNN1_STANDARD_DEVIATION = 0.050
CNN1_FILTER_HEIGHT = 3
CNN1_FILTER_WIDTH = 3
CNN1_STRIDE = [1, 1, 1, 1]
CNN1_STRIDE_SELECTION = 0
CNN1_HPACE = 1  # CNN1_STRIDE and CNN1_PACE have to match
CNN1_WPACE = 1
CNN1_CHANNELS = CHANNELS
CNN1_OUT_CHANNELS = 6  # number of filters in the convolution layer

# Max_pool1 hyperparameters
MAX_POOL1_STRIDE = [1, 2, 2, 1]
MAX_POOL1_STRIDE_SELECTION = round(1.0, 0)
MAX_POOL1_HPACE = 2  # MAX_POOL1_STRIDE and MAX_POOL1_PACE have to match
MAX_POOL1_WPACE = 2
MAX_POOL1_KSIZE = [1, 3, 3, 1]  # This is previously sat at [1, 15, 15, 1]
MAX_POOL1_KSIZE_SELECTION = round(2.0, 0)
MAX_POOL1_FILTER_HEIGHT = 3     # This is previously sat at 15
MAX_POOL1_FILTER_WIDTH = 3      # This is previously sat at 15

# Conv2 hyperparameters
CNN2_WEIGHT = 4.0
CNN2_BIAS = 0.1
CNN2_STANDARD_DEVIATION = 0.050
CNN2_FILTER_HEIGHT = 3
CNN2_FILTER_WIDTH = 3
CNN2_STRIDE = [1, 1, 1, 1]
CNN2_STRIDE_SELECTION = 0.0
CNN2_HPACE = 1  # CNN2_STRIDE and CNN2_PACE have to match
CNN2_WPACE = 1
CNN2_CHANNELS = CNN1_OUT_CHANNELS
CNN2_OUT_CHANNELS = CNN2_CHANNELS * 2  # number of filters in the convolution layer

# Max_pool2 hyperparameters
MAX_POOL2_STRIDE = [1, 2, 2, 1]
MAX_POOL2_STRIDE_SELECTION = round(1.0, 0)
MAX_POOL2_HPACE = 2  # MAX_POOL2_STRIDE and MAX_POOL2_PACE have to match
MAX_POOL2_WPACE = 2
MAX_POOL2_KSIZE = [1, 3, 3, 1]  # This is previously sat at [1, 15, 15, 1]
MAX_POOL2_KSIZE_SELECTION = round(2.0, 0)
MAX_POOL2_FILTER_HEIGHT = 3.0    # This is previously sat at 15
MAX_POOL2_FILTER_WIDTH = 3.0      # This is previously sat at 15

# MLN1 hyperparameters
MLN1_WEIGHT = 0.004
MLN1_BIAS = 0.1
MLN1_STANDARD_DEVIATION = 0.040
MLN1_KEEP_PROB = 0.7
MLN1_NUMBER_NODES = round(1024.0, 0)

# MLN2 hyperparameters
MLN2_WEIGHT = 0.004
MLN2_BIAS = 0.1
MLN2_STANDARD_DEVIATION = 0.040
MLN2_KEEP_PROB = 0.7
MLN2_NUMBER_NODES = round(256.0, 0)

# MLN3 hyperparameters
MLN3_WEIGHT = 0.004
MLN3_BIAS = 0.1
MLN3_STANDARD_DEVIATION = 0.040
MLN3_KEEP_PROB = 0.7
MLN3_NUMBER_NODES = round(64.0, 0)

# SIGMOID_ hyperparameters
SIGMOID_WEIGHT = 0.0040
SIGMOID_BIAS = 0.0
SIGMOID_STANDARD_DEVIATION = round((1 / MLN3_NUMBER_NODES), 3)

# CROSS_ENTROPY hyperparameters
CROSS_ENTROPY_STD_DEV = 0.040

# Widget entry steps
CNN_STEP_WEIGHT = round(0.1, 1)
CNN_STEP_BIAS = round(0.1, 1)
CNN_STEP_STDEV = round(0.010, 3)
CNN_STEP_CHANNELS = round(1.0, 0)
CNN_STEP_OUTCHANNELS = round(1.0, 0)
CNN_STEP_FILTERHEIGHT = round(1.0, 0)
CNN_STEP_FILTERWIDTH = round(1.0, 0)

MLN_STEP_WEIGHT = round(0.1, 1)
MLN_STEP_BIAS = round(0.1, 1)
MLN_STEP_STDEV = round(0.010, 3)
MLN_STEP_KEEPPROB = round(0.1, 1)
MLN_STEP_NODES = round(1.0, 1)

SMD_STEP_WEIGHT = round(0.01, 2)
SMD_STEP_BIAS = round(0.01, 2)
SMD_STEP_STDEV = round(0.0001, 4)

CE_STEP_STDEV = round(0.001, 3)

LR_STEP = round(0.0001, 4)
LD_STEP = round(0.0001, 4)

STEPS_PER_EPOCH_INC = round(500.0, 0)

# FLAGS
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', TENSORBOARD_LOG_DIR,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', int(NUMBER_PICTURES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE),
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

# Global variables
Beginning_Train = True
Tensor_Name = ""
Run = ""
Max_Number_Train_Pictures = 0
Max_Number_File_in_Each_Fold = 0
Max_Training_Steps_in_Fold = 0
Number_Steps_Used_in_Fold = 0
Max_Number_Files_in_Validate_Folds = 0
Change_Fold = True
Train_File_Dict = {}
Validate_File_Dict = {}
Test_File_Dict = {}
First_Test_Data_Run = True

Default_Primary_Hierarchy_Color = wx.Colour(127, 195, 87)          # Lim Green
Default_Secondary_Hierarchy_Color = wx.Colour(13, 184, 155)        # Teal Green
Default_Tertiary_Hierarchy_Color = wx.Colour(248, 239, 98)         # Yellow
Default_Quaternary_Hierarchy_Color = wx.Colour(248, 160, 106)      # Orange
Default_Higher_Hierarchy_Color = wx.Colour(237, 14, 106)           # Red
Default_Background_Color = wx.Colour(155, 163, 165)                # Grey

Logit_list = [(CONVOLUTION_LAYER, NUMBER_CONVOLUTION_LAYERS), (MULTI_NEURON_LAYER, NUMBER_MULTI_NEURON_LAYERS),
              (SIMOID_LAYER, NUMBER_SIGMOID_LAYERS), (CROSS_ENTROPY_LAYER, NUMBER_CROSS_ENTROPY_LAYERS),
              (LEARNING_LAYER, NUMBER_LEARNING_LAYERS), (NUMBER_OF_STEPS_LAYER, NUMBER_STEPS_LAYER)]

# The following dictionaries may be replaced by lists; however, the codes are easier to read with the dictionaries
Sensitivity_dict = {"SENSITIVITY" + repr(k): k + 7 for k in range(NUMBER_POSSIBLE_FACES)}
Specificity_dict = {"SPECIFICITY" + repr(k): k + 7 + NUMBER_POSSIBLE_FACES for k in range(NUMBER_POSSIBLE_FACES)}

# The default values may not be necessary; however, the codes are much easier to read with them
Default_CNN_Weight_List = [CNN1_WEIGHT, CNN2_WEIGHT]
Default_CNN_Weight_Min_List = [0.0, 0.0]
Default_CNN_Weight_Max_List = [100.0, 100.0]
Default_CNN_Weight_Inc_List = [0.1, 0.1]

Default_CNN_Biases_List = [CNN1_BIAS, CNN2_BIAS]
Default_CNN_Biases_Min_List = [0.0, 0.0]
Default_CNN_Biases_Max_List = [100.0, 100.0]
Default_CNN_Biases_Inc_List = [0.1, 0.1]

Default_CNN_StDev_List = [CNN1_STANDARD_DEVIATION, CNN2_STANDARD_DEVIATION]
Default_CNN_StDev_Min_List = [0.000, 0.000]
Default_CNN_StDev_Max_List = [10.000, 10.000]
Default_CNN_StDev_Inc_List = [0.001, 0.001]

Default_CNN_Channels_List = [CNN1_CHANNELS, CNN2_CHANNELS]
# Default_CNN_Channels_Min_List = [round(1.0, 0), round(1.0, 0)]
# Default_CNN_Channels_Max_List = [round(64.0, 0), round(64.0, 0)]
# Default_CNN_Channels_Inc_List = [round(1.0, 0), round(1.0, 0)]

Default_CNN_OutChannels_List = [CNN1_OUT_CHANNELS, CNN2_OUT_CHANNELS]
Default_CNN_OutChannels_Min_List = [round(1.0, 0), round(1.0, 0)]
Default_CNN_OutChannels_Max_List = [round(256.0, 0), round(256.0, 0)]
Default_CNN_OutChannels_Inc_List = [round(1.0, 0), round(1.0, 0)]

Default_CNN_FilterWidth_List = [CNN1_FILTER_WIDTH, CNN2_FILTER_WIDTH]
Default_CNN_FilterWidth_Min_List = [round(1.0, 0), round(1.0, 0)]
Default_CNN_FilterWidth_Max_List = [round(40.0, 0), round(40.0, 0)]
Default_CNN_FilterWidth_Inc_List = [round(1.0, 0), round(1.0, 0)]

Default_CNN_FilterHeight_List = [CNN1_FILTER_HEIGHT, CNN2_FILTER_HEIGHT]
Default_CNN_FilterHeight_Min_List = [round(1.0, 0), round(1.0, 0)]
Default_CNN_FilterHeight_Max_List = [round(40.0, 0), round(40.0, 0)]
Default_CNN_FilterHeight_Inc_List = [round(1.0, 0), round(1.0, 0)]

Default_CNN_Stride_List = [CNN1_STRIDE, CNN2_STRIDE]
Default_CNN_Stride_Selection_List = [CNN1_STRIDE_SELECTION, CNN2_STRIDE_SELECTION]
Default_CNN_Stride_Step_List = [round(1.0, 0), round(1.0, 0)]
Default_CNN_Stride_Sel_Min_List = [round(1.0, 0), round(1.0, 0)]
Default_CNN_Stride_Sel_Max_List = [round(100.0, 0), round(100.0, 0)]
Default_CNN_Stride_Sel_Inc_List = [round(1.0, 0), round(1.0, 0)]

Default_CNN_HPace_List = [CNN1_HPACE, CNN2_HPACE]
Default_CNN_WPACE_List = [CNN1_WPACE, CNN2_WPACE]

Default_MP_Stride_List = [MAX_POOL1_STRIDE, MAX_POOL2_STRIDE]
Default_MP_Stride_Selection_List = [MAX_POOL1_STRIDE_SELECTION, MAX_POOL2_STRIDE_SELECTION]
Default_MP_Stride_Step_List = [round(1.0, 0), round(1.0, 0)]
Default_MP_Stride_Sel_Min_List = [round(1.0, 0), round(1.0, 0)]
Default_MP_Stride_Sel_Max_List = [round(100.0, 0), round(100.0, 0)]
Default_MP_Stride_Sel_Inc_List = [round(1.0, 0), round(1.0, 0)]
Default_MP_HPace_List = [MAX_POOL1_HPACE, MAX_POOL2_HPACE]
Default_MP_WPace_List = [MAX_POOL1_WPACE, MAX_POOL2_WPACE]
Default_MP_Ksize_List = [MAX_POOL1_KSIZE, MAX_POOL2_KSIZE]
Default_MP_KSize_Selection_List = [MAX_POOL1_KSIZE_SELECTION, MAX_POOL2_KSIZE_SELECTION]
Default_MP_KSize_Step_List = [round(1.0, 0), round(1.0, 0)]
Default_MP_KSize_Sel_Min_List = [round(1.0, 0), round(1.0, 0)]
Default_MP_KSize_Sel_Max_List = [round(100.0, 0), round(100.0, 0)]
Default_MP_KSize_Sel_Inc_List = [round(1.0, 0), round(1.0, 0)]
Default_MP_Filter_Height_List = [MAX_POOL1_FILTER_HEIGHT, MAX_POOL2_FILTER_HEIGHT]
Default_MP_Filter_Width_List = [MAX_POOL1_FILTER_WIDTH, MAX_POOL2_FILTER_WIDTH]

Default_DL_Weight_List = [MLN1_WEIGHT, MLN2_WEIGHT, MLN3_WEIGHT]
Default_DL_Weight_Min_List = [0.0, 0.0]
Default_DL_Weight_Max_List = [100.0, 100.0]
Default_DL_Weight_Inc_List = [0.1, 0.1]

Default_DL_Biases_List = [MLN1_BIAS, MLN2_BIAS, MLN3_BIAS]
Default_DL_Biases_Min_List = [0.0, 0.0]
Default_DL_Biases_Max_List = [50.0, 50.0]
Default_DL_Biases_Inc_List = [0.1, 0.1]

Default_DL_StDev_List = [MLN1_STANDARD_DEVIATION, MLN2_STANDARD_DEVIATION, MLN3_STANDARD_DEVIATION]
Default_DL_StDev_Min_List = [0.000, 0.000]
Default_DL_StDev_Max_List = [10.000, 10.000]
Default_DL_StDev_Inc_List = [0.001, 0.001]

Default_DL_KeepProb_List = [MLN1_KEEP_PROB, MLN2_KEEP_PROB, MLN3_KEEP_PROB]
Default_DL_KeepProb_Min_List = [0.1, 0.1]
Default_DL_KeepProb_Max_List = [1.0, 1.0]
Default_DL_KeepProb_Inc_List = [0.1, 0.1]

Default_DL_Nodes_List = [MLN1_NUMBER_NODES, MLN2_NUMBER_NODES, MLN3_NUMBER_NODES]
Default_DL_Nodes_Min_List = [round(1.0, 0), round(1.0, 0)]
Default_DL_Nodes_Max_List = [round(3200.0, 0), round(3200.0, 0)]
Default_DL_Nodes_Inc_List = [round(1.0, 0), round(1.0, 0)]

Default_SMD_Weight_List = [SIGMOID_WEIGHT]
Default_SMD_Weight_Min_List = [0.0]
Default_SMD_Weight_Max_List = [100.0]
Default_SMD_Weight_Inc_List = [0.1]

Default_SMD_Bias_List = [SIGMOID_BIAS]
Default_SMD_Biases_Min_List = [0.0]
Default_SMD_Biases_Max_List = [50.0]
Default_SMD_Biases_Inc_List = [0.1]

Default_SMD_StDev_List = [SIGMOID_STANDARD_DEVIATION]
Default_SMD_StDev_Min_List = [0.000]
Default_SMD_StDev_Max_List = [10.000]
Default_SMD_StDev_Inc_List = [0.001]

Default_CE_StDev_List = [CROSS_ENTROPY_STD_DEV]
Default_CE_StDev_Min_List = [0.001]
Default_CE_StDev_Max_List = [10.000]
Default_CE_StDev_Inc_List = [0.001]

Default_Learning_Rate_List = [INITIAL_LEARNING_RATE]
Default_Learning_Rate_Min_List = [0.0001]
Default_Learning_Rate_Max_List = [1.0000]
Default_Learning_Rate_Inc_List = [0.0001]

Default_Learning_Decay_List = [LEARNING_RATE_DECAY_FACTOR]
Default_Learning_Decay_Min_List = [0.0000]
Default_Learning_Decay_Max_List = [1.0000]
Default_Learning_Decay_Inc_List = [0.0001]

Default_Number_Steps_Per_Epoch_List = [NUMBER_STEPS_PER_EPOCH]
Default_Number_Steps_Per_Epoch_Min_List = [round(500.0, 0)]
Default_Number_Steps_Per_Epoch_Max_List = [round(20000.0, 0)]
Default_Number_Steps_Per_Epoch_Inc_List = [STEPS_PER_EPOCH_INC]

# This is to set up a custom message to notify the mainframe class that the setting dialog was closed.
# This is necessary to communicate between the two classes.
# EVT_CUSTOM_DIALOG_CLOSED is, of course, the flag.
# The flag is used in the binding of the self.panel when the settings dialog is closed in the initUI() method.
# DiaglogClosedEvent is the method to give the event ID associated with the EVT_CUSTOM_DIALOG_CLOSED.
# It is called during the initiation of the SettingDialog class to generate an unique event ID
# wx.lib.newevent.NewEvent() is imported from wx.lib.newevent
DialogClosedEvent, EVT_CUSTOM_DIALOG_CLOSED = wx.lib.newevent.NewEvent()


def test_image_input():
    # Process to make train dictionary and validate dictionary
    label_list = []
    file_list = []
    label_filename = ""
    file_filename = ""

    # Set up, train label directory, and label.txt
    if not os.path.exists(TEST_LABEL_DIR):
        print("No train label directory: ", TEST_LABEL_DIR)
        exit(506)

    # Read in label list
    try:
        label_filename = os.path.join(TEST_LABEL_DIR, "label.txt")
        label_file = open(label_filename, 'r+')
        label_list = json.load(label_file)
    except IOError:
        print("cannot open and read ", label_filename)
        exit(507)

    # Read in file list
    try:
        file_filename = os.path.join(TEST_LABEL_DIR, "file.txt")
        file_file = open(file_filename, 'r+')
        file_list = json.load(file_file)
        file_file.close()
    except IOError:
        print("cannot open and read ", file_filename)
        exit(508)

    global Test_File_Dict, First_Test_Data_Run
    if First_Test_Data_Run:
        First_Test_Data_Run = False
        for x in range(0, len(file_list)):
            Test_File_Dict[(file_list[x])] = np.float32(label_list[x])

    images_list = []
    labels_list = []
    # Get BATCH_SIZE random files with labels
    for x in range(NUMBER_ACCURACY_IMAGES):
        random_file, label = random.choice(list(Test_File_Dict.items()))

        # Check for existence of RGBtrain data directory
        if not os.path.exists(RBG_TEST_DATA_DIR):
            print("No test data directory: ", RBG_TEST_DATA_DIR)
            exit(505)

        # Set up random image list
        full_rgb_file = os.path.join(RBG_TEST_DATA_DIR, random_file)

        # Get the rgb data from file
        rgb_data = [IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS]
        try:
            rgb_data = np.load(full_rgb_file)
        except IOError:
            print("cannot open ", full_rgb_file)
            exit(507)

        # Preprocess image
        raw_image = tf.convert_to_tensor(rgb_data)
        distorted_image = tf.reshape(raw_image, [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])

        # Randomly flip the image horizontally
        if FLIP:
            distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly change the brightness of the image
        if ALTER_BRIGHTNESS:
            distorted_image = tf.image.random_brightness(distorted_image, max_delta=MAXIMUM_BRIGHTNESS_DELTA)

        # Radomly change the contrast of the image
        if ALTER_CONTRAST:
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=LOWER_CONTRAST_CHANGE, upper=UPPER_CONTRAST_CHANGE)

        # Put the batched images in a list
        images_list.append(distorted_image)

        # Convert the labels into a tensor and place it in a list
        label_tensor = tf.convert_to_tensor(np.float32(label))
        labels_list.append(label_tensor)

    images_list = tf.convert_to_tensor(images_list)
    labels_list = tf.convert_to_tensor(labels_list)

    return images_list, labels_list


def get_tag():
    # Setup the BabyPictureTrainLog directory and the data subdirectories
    tag_filename = ""
    # Test for existence of TensorBoard log directory and make one if not exist
    if not os.path.exists(TENSORBOARD_LOG_DIR):
        print("No Tensorboard data directory: ", TENSORBOARD_LOG_DIR, "and a new one will be created")
        os.mkdir(TENSORBOARD_LOG_DIR)

    # Test for existence of tag directory and tag file. Make one if not exist
    if not os.path.exists(WORKING_DIR):
        print("No Tag directory: ", WORKING_DIR, "and a new one will be created")
        os.mkdir(WORKING_DIR)
    try:
        tag_filename = os.path.join(WORKING_DIR, "tag.txt")
        tag_file = open(tag_filename, "r+")
        _run_ = tag_file.read()

    except OSError:
        while True:
            _user_input1 = input("Cannot find tag_file: tag.txt, Do you want to open a new one? (y or n)")
            if _user_input1 == 'y':
                while True:
                    _user_input2 = \
                        input("If there is an old file,tag run will be resected to zero and old tag file will be "
                              "erased. "
                              " For sure to use a new tag file? (y or n);")
                    if _user_input2 == 'y':
                        tag_file = open(tag_filename, "w")
                        _run_ = "run_0/"
                        tag_file.write(_run_)
                        break  # This break is for the while loop and not for the elif
                    elif _user_input2 == 'n':
                        print("Exit for now to allow for further decision!")
                        exit(508)
                    else:
                        print("Please just enter 'y' or 'n'.  Try again...")

                break  # This break is for the while loop and not for the elif
            elif _user_input1 == 'n':
                print("Exit for now to allow for further decision!")
                exit(509)
            else:
                print("Please just enter 'y' or 'n'.  Try again...")

    # The annotation for the tag is the following:
    # CNN stands for convolution layers. The two sets of numbers stands for weight + biases ~ standard deviation.
    # FC stands for fully connected laysers. The sets of numbers following stand for weight + biases ~ st. dev.
    # SIGMOID stands for the sigmoid layer. The numbers following stand for weight + biases.
    # XENT stands for cross entropy. The number following stands for st. dev.
    if XAVIER:
        _current_tag1 = "__CNN" + "_XAVIER" + "_" + repr(CNN1_BIAS) + "__" + repr(CNN1_STANDARD_DEVIATION)
        _current_tag2 = "_XAVIER" + "_" + repr(CNN2_BIAS) + "__" + repr(CNN2_STANDARD_DEVIATION)
        _current_tag3 = "_FC" + "_XAVIER" + "_" + repr(MLN1_BIAS) + "__" + repr(MLN1_STANDARD_DEVIATION)
        _current_tag4 = "_XAVIER" + "_" + repr(MLN2_BIAS) + "__" + repr(MLN2_STANDARD_DEVIATION)
        _current_tag5 = "_XAVIER" + "_" + repr(MLN3_BIAS) + "__" + repr(MLN3_STANDARD_DEVIATION)
        _current_tag6 = "_SIGMOID" + "_XAVIER" + "_" + repr(SIGMOID_BIAS) + "_XENT" + "__" + repr(CROSS_ENTROPY_STD_DEV)
        _current_tag = _current_tag1 + _current_tag2 + _current_tag3 + _current_tag4 + _current_tag5 + _current_tag6

    else:
        _current_tag1 = "__CNN" + "_" + repr(CNN1_WEIGHT) + "_" + repr(CNN1_BIAS) + "__" + repr(CNN1_STANDARD_DEVIATION)
        _current_tag2 = "_" + repr(CNN2_WEIGHT) + "_" + repr(CNN2_BIAS) + "__" + repr(CNN2_STANDARD_DEVIATION) + "_"
        _current_tag3 = "_FC" + "_" + repr(MLN1_WEIGHT) + "_" + repr(MLN1_BIAS) + "__" + repr(MLN1_STANDARD_DEVIATION)
        _current_tag4 = "_" + repr(MLN2_WEIGHT) + "_" + repr(MLN2_BIAS) + "__" + repr(MLN2_STANDARD_DEVIATION)
        _current_tag5 = "_" + repr(MLN3_WEIGHT) + "_" + repr(MLN3_BIAS) + "__" + repr(MLN3_STANDARD_DEVIATION) + "_"
        _current_tag6 = "_SIGMOID" + "_" + repr(SIGMOID_WEIGHT) + "_" + repr(SIGMOID_BIAS)
        _current_tag7 = "_" + "_XENT" + "_" + "__" + repr(CROSS_ENTROPY_STD_DEV)
        _current_tag8 = _current_tag6 + _current_tag7
        _current_tag = _current_tag1 + _current_tag2 + _current_tag3 + _current_tag4 + _current_tag5 + _current_tag8

    # Extract the run number; add one to it; and write back to file
    _final_run_number = 0
    _length_ = len(_run_)
    for x in range(6, _length_, 1):
        _run_number_string = _run_[x]
        if not _run_number_string == "/":
            _run_number = int(_run_number_string)
            _final_run_number *= 10
            _final_run_number += _run_number
    _final_run_number += 1
    _run_ = "epoch_" + repr(_final_run_number) + "/"
    tag_file.seek(0)
    tag_file.truncate()
    tag_file.write(_run_)
    tag_file.close()

    return _run_, _current_tag
 
 
def get_label_(name, shape, dtype):
    label_ = None
    with tf.device('/cpu:0'):
        try:
            initializer = tf.truncated_normal(shape, dtype=dtype)
            label_ = tf.get_variable(name, initializer=initializer, dtype=dtype)
        except ValueError:
            print("Memory for ", name, "cannot be initialized")
            """label_.assign(labels)"""

    return label_


class ParameterPanel(scrolled.ScrolledPanel):
    def __init__(self, parent):
        super(ParameterPanel, self).__init__(parent)
        
        self.SetAutoLayout(1)
        
        self.parent = parent

        # Setting up the timer for delayed activations of changing self.ordering_list
        # in self.on_field_changed() and in on_choice_field_changed() methods
        self.field_changed_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.do_change_ordering_list)
        self.field_changed_index = -1
        # self.choice_field_timer_flag = False
        # self.field_timer_flag = False
    
        # interactive_sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
        #                                                                    log_device_placement=True))
        self.epoch_number = -1
        self.num_epoch = 1
    
        bmp = wx.Bitmap(u"e:/Chester/bmpFiles/AppIcons/bmp/16x16/Script.bmp")
        self.icon = wx.Icon(bmp)
        self.parent.SetIcon(self.icon)
    
        # report = int(NUMBER_STEPS_PER_EPOCH / NUMBER_OF_STEPS_PER_REPORT)
        self.np_stats = None
        self.stat_name_list = [LOSS, IMAGE_ACCURACY, POSITIVE_FACE_ACCURACY, NEGATIVE_FACE_ACCURACY,
                               TOTAL_FACE_ACCURACY, TENSORFLOW_IMAGE_ACCURACY, TENSORFLOW_FACE_ACCURACY,
                               SEN1, SEN2, SEN3, SEN4, SEN5, SEN6, SEN7, SEN8, SEN9, SEN10, SEN11, SEN12,
                               SP1, SP2, SP3, SP4, SP5, SP6, SP7, SP8, SP9, SP10, SP11, SP12]

        self.panel = wx.Panel(self, wx.ID_ANY)
        self.idholder = None
        self.save_flag = False
        self.saved_ordering_list = []
        self.ordering_list = []
        self.ordering_value_list = []
        self.ordering_index = -1
        self.ordering_hierarchy_unchanged_flag = True
        self.user_data_list = []
        self.empty_user_data_dict_flag = True
        self.color_data_unchanged_flag = True
        self.empty_ordering_flag = True

        self.file_list = []
        self.label_list = []
        self.max_num_file_in_each_fold = 0
        self.num_folds = 15
        self.change_fold = True

        self.estimated_epoch_time = None
        self.previous_time_elapsed = None
        self.previous_epoch_time = None
        self.average_epoch_time = None
        self.time_to_complete = None
        self.epoch_remaining = None
        self.epoch_time_established_flag = False
        
        self.train_file_dict = {}
        self.validate_file_dict = {}
       
        self.primarycolor = Default_Primary_Hierarchy_Color
        self.secondarycolor = Default_Secondary_Hierarchy_Color
        self.tertiarycolor = Default_Tertiary_Hierarchy_Color
        self.quaternarycolor = Default_Quaternary_Hierarchy_Color
        self.highercolor = Default_Higher_Hierarchy_Color
    
        # Setting up the timer for distinguishing single click versus double click
        # self.timer = wx.Timer(self, wx.ID_ANY)
    
        # Binding the timer to trigger the self.onTimeSingleClick() method once the time is up (300msec for our case)
        # Basically, the timer starts when one click is performed.
        # If the second click comes in before 300 msec, the first and second clicks would be perceived
        # to be a double click. If the second click comes in after the first click,
        # the second click would be perceived as a separate single click.
        # self.Bind(wx.EVT_TIMER, self.on_timed_single_click)
    
        self.nt_starting_wt_list = Default_CNN_Weight_List
        self.nt_starting_wt_min_list = Default_CNN_Weight_Min_List
        self.nt_starting_wt_max_list = Default_CNN_Weight_Max_List
        self.nt_starting_wt_inc_list = Default_CNN_Weight_Inc_List
        self.nt_ending_wt_list = Default_CNN_Weight_List
        self.nt_ending_wt_min_list = Default_CNN_Weight_Min_List
        self.nt_ending_wt_max_list = Default_CNN_Weight_Max_List
        self.nt_ending_wt_inc_list = Default_CNN_Weight_Inc_List
        self.nt_step_wt_list = Default_CNN_Weight_Inc_List
        self.nt_step_wt_min_list = Default_CNN_Weight_Inc_List
        self.nt_step_wt_max_list = Default_CNN_Weight_Max_List
        self.nt_step_wt_inc_list = Default_CNN_Weight_Inc_List
    
        self.nt_starting_bias_list = Default_CNN_Biases_List
        self.nt_starting_bias_min_list = Default_CNN_Biases_Min_List
        self.nt_starting_bias_max_list = Default_CNN_Biases_Max_List
        self.nt_starting_bias_inc_list = Default_CNN_Biases_Inc_List
        self.nt_ending_bias_list = Default_CNN_Biases_List
        self.nt_ending_bias_min_list = Default_CNN_Biases_Min_List
        self.nt_ending_bias_max_list = Default_CNN_Biases_Max_List
        self.nt_ending_bias_inc_list = Default_CNN_Biases_Inc_List
        self.nt_step_bias_list = Default_CNN_Biases_Inc_List
        self.nt_step_bias_min_list = Default_CNN_Biases_Inc_List
        self.nt_step_bias_max_list = Default_CNN_Biases_Max_List
        self.nt_step_bias_inc_list = Default_CNN_Biases_Inc_List
    
        self.nt_starting_stdev_list = Default_CNN_StDev_List
        self.nt_starting_stdev_min_list = Default_CNN_StDev_Min_List
        self.nt_starting_stdev_max_list = Default_CNN_StDev_Max_List
        self.nt_starting_stdev_inc_list = Default_CNN_StDev_Inc_List
        self.nt_ending_stdev_list = Default_CNN_StDev_List
        self.nt_ending_stdev_min_list = Default_CNN_StDev_Min_List
        self.nt_ending_stdev_max_list = Default_CNN_StDev_Max_List
        self.nt_ending_stdev_inc_list = Default_CNN_StDev_Inc_List
        self.nt_step_stdev_list = Default_CNN_StDev_Inc_List
        self.nt_step_stdev_min_list = Default_CNN_StDev_Inc_List
        self.nt_step_stdev_max_list = Default_CNN_StDev_Max_List
        self.nt_step_stdev_inc_list = Default_CNN_StDev_Inc_List
    
        self.nt_starting_chnnl_list = Default_CNN_Channels_List
        
        self.nt_starting_outchnnl_list = Default_CNN_OutChannels_List
        self.nt_starting_outchnnl_min_list = Default_CNN_OutChannels_Min_List
        self.nt_starting_outchnnl_max_list = Default_CNN_OutChannels_Max_List
        self.nt_starting_outchnnl_inc_list = Default_CNN_OutChannels_Inc_List
        self.nt_ending_outchnnl_list = Default_CNN_OutChannels_List
        self.nt_ending_outchnnl_min_list = Default_CNN_OutChannels_Min_List
        self.nt_ending_outchnnl_max_list = Default_CNN_OutChannels_Max_List
        self.nt_ending_outchnnl_inc_list = Default_CNN_OutChannels_Inc_List
        self.nt_step_outchnnl_list = Default_CNN_OutChannels_Inc_List
        self.nt_step_outchnnl_min_list = Default_CNN_OutChannels_Inc_List
        self.nt_step_outchnnl_max_list = Default_CNN_OutChannels_Max_List
        self.nt_step_outchnnl_inc_list = Default_CNN_OutChannels_Inc_List
    
        self.nt_starting_fltrhght_list = Default_CNN_FilterHeight_List
        self.nt_starting_fltrhght_min_list = Default_CNN_FilterHeight_Min_List
        self.nt_starting_fltrhght_max_list = Default_CNN_FilterHeight_Max_List
        self.nt_starting_fltrhght_inc_list = Default_CNN_FilterHeight_Inc_List
        self.nt_ending_fltrhght_list = Default_CNN_FilterHeight_List
        self.nt_ending_fltrhght_min_list = Default_CNN_FilterHeight_Min_List
        self.nt_ending_fltrhght_max_list = Default_CNN_FilterHeight_Max_List
        self.nt_ending_fltrhght_inc_list = Default_CNN_FilterHeight_Inc_List
        self.nt_step_fltrhght_list = Default_CNN_FilterHeight_Inc_List
        self.nt_step_fltrhght_min_list = Default_CNN_FilterHeight_Inc_List
        self.nt_step_fltrhght_max_list = Default_CNN_FilterHeight_Max_List
        self.nt_step_fltrhght_inc_list = Default_CNN_FilterHeight_Inc_List
    
        self.nt_starting_fltrwdth_list = Default_CNN_FilterWidth_List
        self.nt_starting_fltrwdth_min_list = Default_CNN_FilterWidth_Min_List
        self.nt_starting_fltrwdth_max_list = Default_CNN_FilterWidth_Max_List
        self.nt_starting_fltrwdth_inc_list = Default_CNN_FilterWidth_Inc_List
        self.nt_ending_fltrwdth_list = Default_CNN_FilterWidth_List
        self.nt_ending_fltrwdth_min_list = Default_CNN_FilterWidth_Min_List
        self.nt_ending_fltrwdth_max_list = Default_CNN_FilterWidth_Max_List
        self.nt_ending_fltrwdth_inc_list = Default_CNN_FilterWidth_Inc_List
        self.nt_step_fltrwdth_list = Default_CNN_FilterWidth_Inc_List
        self.nt_step_fltrwdth_min_list = Default_CNN_FilterWidth_Inc_List
        self.nt_step_fltrwdth_max_list = Default_CNN_FilterWidth_Max_List
        self.nt_step_fltrwdth_inc_list = Default_CNN_FilterWidth_Inc_List
    
        self.nt_strd_list = Default_CNN_Stride_List
        self.nt_starting_strd_sel_list = Default_CNN_Stride_Selection_List
        self.nt_ending_strd_sel_list = Default_CNN_Stride_Selection_List
        self.nt_strd_step_list = Default_CNN_Stride_Step_List
        self.nt_strd_sel_min_list = Default_CNN_Stride_Sel_Inc_List
        self.nt_strd_sel_max_list = Default_CNN_Stride_Sel_Max_List
        self.nt_strd_sel_inc_list = Default_CNN_Stride_Sel_Inc_List
        self.nt_starting_hpce_list = Default_CNN_HPace_List
        self.nt_starting_wpce_list = Default_CNN_WPACE_List
        self.nt_ending_hpce_list = Default_CNN_HPace_List
        self.nt_ending_wpce_list = Default_CNN_WPACE_List
    
        self.mp_ksze_list = Default_MP_Ksize_List
        self.mp_starting_ksze_sel_list = Default_MP_KSize_Selection_List
        self.mp_ending_ksze_sel_list = Default_MP_KSize_Selection_List
        self.mp_ksze_step_list = Default_MP_KSize_Step_List
        self.mp_ksze_sel_min_list = Default_MP_KSize_Sel_Inc_List
        self.mp_ksze_sel_max_list = Default_MP_KSize_Sel_Max_List
        self.mp_ksze_sel_inc_list = Default_MP_KSize_Sel_Inc_List
        self.mp_starting_fltrhght_list = Default_MP_Filter_Height_List
        self.mp_ending_fltrhght_list = Default_MP_Filter_Height_List
        self.mp_starting_fltrwdth_list = Default_MP_Filter_Width_List
        self.mp_ending_fltrwdth_list = Default_MP_Filter_Width_List
        self.mp_strd_list = Default_MP_Stride_List
        self.mp_starting_strd_sel_list = Default_MP_Stride_Selection_List
        self.mp_ending_strd_sel_list = Default_MP_Stride_Selection_List
        self.mp_strd_step_list = Default_MP_Stride_Step_List
        self.mp_strd_sel_min_list = Default_MP_Stride_Sel_Inc_List
        self.mp_strd_sel_max_list = Default_MP_Stride_Sel_Max_List
        self.mp_strd_sel_inc_list = Default_MP_Stride_Sel_Inc_List
        self.mp_starting_hpce_list = Default_MP_HPace_List
        self.mp_ending_hpce_list = Default_MP_HPace_List
        self.mp_starting_wpce_list = Default_MP_WPace_List
        self.mp_ending_wpce_list = Default_MP_WPace_List
    
        self.dl_starting_wt_list = Default_DL_Weight_List
        self.dl_starting_wt_min_list = Default_DL_Weight_Min_List
        self.dl_starting_wt_max_list = Default_DL_Weight_Max_List
        self.dl_starting_wt_inc_list = Default_DL_Weight_Inc_List
        self.dl_ending_wt_list = Default_DL_Weight_List
        self.dl_ending_wt_min_list = Default_DL_Weight_Min_List
        self.dl_ending_wt_max_list = Default_DL_Weight_Max_List
        self.dl_ending_wt_inc_list = Default_DL_Weight_Inc_List
        self.dl_step_wt_list = Default_DL_Weight_Inc_List
        self.dl_step_wt_min_list = Default_DL_Weight_Inc_List
        self.dl_step_wt_max_list = Default_DL_Weight_Max_List
        self.dl_step_wt_inc_list = Default_DL_Weight_Inc_List
    
        self.dl_starting_bias_list = Default_DL_Biases_List
        self.dl_starting_bias_min_list = Default_DL_Biases_Min_List
        self.dl_starting_bias_max_list = Default_DL_Biases_Max_List
        self.dl_starting_bias_inc_list = Default_DL_Biases_Inc_List
        self.dl_ending_bias_list = Default_DL_Biases_List
        self.dl_ending_bias_min_list = Default_DL_Biases_Min_List
        self.dl_ending_bias_max_list = Default_DL_Biases_Max_List
        self.dl_ending_bias_inc_list = Default_DL_Biases_Inc_List
        self.dl_step_bias_list = Default_DL_Biases_Inc_List
        self.dl_step_bias_min_list = Default_DL_Biases_Inc_List
        self.dl_step_bias_max_list = Default_DL_Biases_Max_List
        self.dl_step_bias_inc_list = Default_DL_Biases_Inc_List
    
        self.dl_starting_stdev_list = Default_DL_StDev_List
        self.dl_starting_stdev_min_list = Default_DL_StDev_Min_List
        self.dl_starting_stdev_max_list = Default_DL_StDev_Max_List
        self.dl_starting_stdev_inc_list = Default_DL_StDev_Inc_List
        self.dl_ending_stdev_list = Default_DL_StDev_List
        self.dl_ending_stdev_min_list = Default_DL_StDev_Min_List
        self.dl_ending_stdev_max_list = Default_DL_StDev_Max_List
        self.dl_ending_stdev_inc_list = Default_DL_StDev_Inc_List
        self.dl_step_stdev_list = Default_DL_StDev_Inc_List
        self.dl_step_stdev_min_list = Default_DL_StDev_Inc_List
        self.dl_step_stdev_max_list = Default_DL_StDev_Max_List
        self.dl_step_stdev_inc_list = Default_DL_StDev_Inc_List
    
        self.dl_starting_kpprb_list = Default_DL_KeepProb_List
        self.dl_starting_kpprb_min_list = Default_DL_KeepProb_Min_List
        self.dl_starting_kpprb_max_list = Default_DL_KeepProb_Max_List
        self.dl_starting_kpprb_inc_list = Default_DL_KeepProb_Inc_List
        self.dl_ending_kpprb_list = Default_DL_KeepProb_List
        self.dl_ending_kpprb_min_list = Default_DL_KeepProb_Min_List
        self.dl_ending_kpprb_max_list = Default_DL_KeepProb_Max_List
        self.dl_ending_kpprb_inc_list = Default_DL_KeepProb_Inc_List
        self.dl_step_kpprb_list = Default_DL_KeepProb_Inc_List
        self.dl_step_kpprb_min_list = Default_DL_KeepProb_Inc_List
        self.dl_step_kpprb_max_list = Default_DL_KeepProb_Max_List
        self.dl_step_kpprb_inc_list = Default_DL_KeepProb_Inc_List
    
        self.dl_starting_nodes_list = Default_DL_Nodes_List
        self.dl_starting_nodes_min_list = Default_DL_Nodes_Min_List
        self.dl_starting_nodes_max_list = Default_DL_Nodes_Max_List
        self.dl_starting_nodes_inc_list = Default_DL_Nodes_Inc_List
        self.dl_ending_nodes_list = Default_DL_Nodes_List
        self.dl_ending_nodes_min_list = Default_DL_Nodes_Min_List
        self.dl_ending_nodes_max_list = Default_DL_Nodes_Max_List
        self.dl_ending_nodes_inc_list = Default_DL_Nodes_Inc_List
        self.dl_step_nodes_list = Default_DL_Nodes_Inc_List
        self.dl_step_nodes_min_list = Default_DL_Nodes_Inc_List
        self.dl_step_nodes_max_list = Default_DL_Nodes_Max_List
        self.dl_step_nodes_inc_list = Default_DL_Nodes_Inc_List
    
        self.smd_starting_wt_list = Default_SMD_Weight_List
        self.smd_starting_wt_min_list = Default_SMD_Weight_Min_List
        self.smd_starting_wt_max_list = Default_SMD_Weight_Max_List
        self.smd_starting_wt_inc_list = Default_SMD_Weight_Inc_List
        self.smd_ending_wt_list = Default_SMD_Weight_List
        self.smd_ending_wt_min_list = Default_SMD_Weight_Min_List
        self.smd_ending_wt_max_list = Default_SMD_Weight_Max_List
        self.smd_ending_wt_inc_list = Default_SMD_Weight_Inc_List
        self.smd_step_wt_list = Default_SMD_Weight_Inc_List
        self.smd_step_wt_min_list = Default_SMD_Weight_Inc_List
        self.smd_step_wt_max_list = Default_SMD_Weight_Max_List
        self.smd_step_wt_inc_list = Default_SMD_Weight_Inc_List
    
        self.smd_starting_bias_list = Default_SMD_Bias_List
        self.smd_starting_bias_min_list = Default_SMD_Biases_Min_List
        self.smd_starting_bias_max_list = Default_SMD_Biases_Max_List
        self.smd_starting_bias_inc_list = Default_SMD_Biases_Inc_List
        self.smd_ending_bias_list = Default_SMD_Bias_List
        self.smd_ending_bias_min_list = Default_SMD_Biases_Min_List
        self.smd_ending_bias_max_list = Default_SMD_Biases_Max_List
        self.smd_ending_bias_inc_list = Default_SMD_Biases_Inc_List
        self.smd_step_bias_list = Default_SMD_Biases_Inc_List
        self.smd_step_bias_min_list = Default_SMD_Biases_Inc_List
        self.smd_step_bias_max_list = Default_SMD_Biases_Max_List
        self.smd_step_bias_inc_list = Default_SMD_Biases_Inc_List
    
        self.smd_starting_stdev_list = Default_SMD_StDev_List
        self.smd_starting_stdev_min_list = Default_SMD_StDev_Min_List
        self.smd_starting_stdev_max_list = Default_SMD_StDev_Max_List
        self.smd_starting_stdev_inc_list = Default_SMD_StDev_Inc_List
        self.smd_ending_stdev_list = Default_SMD_StDev_List
        self.smd_ending_stdev_min_list = Default_SMD_StDev_Min_List
        self.smd_ending_stdev_max_list = Default_SMD_StDev_Max_List
        self.smd_ending_stdev_inc_list = Default_SMD_StDev_Inc_List
        self.smd_step_stdev_list = Default_SMD_StDev_Inc_List
        self.smd_step_stdev_min_list = Default_SMD_StDev_Inc_List
        self.smd_step_stdev_max_list = Default_SMD_StDev_Max_List
        self.smd_step_stdev_inc_list = Default_SMD_StDev_Inc_List
    
        self.starting_lrnrate_list = Default_Learning_Rate_List
        self.starting_lrnrate_min_list = Default_Learning_Rate_Min_List
        self.starting_lrnrate_max_list = Default_Learning_Rate_Max_List
        self.starting_lrnrate_inc_list = Default_Learning_Rate_Inc_List
        self.ending_lrnrate_list = Default_Learning_Rate_List
        self.ending_lrnrate_min_list = Default_Learning_Rate_Min_List
        self.ending_lrnrate_max_list = Default_Learning_Rate_Max_List
        self.ending_lrnrate_inc_list = Default_Learning_Rate_Inc_List
        self.step_lrnrate_list = Default_Learning_Rate_Inc_List
        self.step_lrnrate_min_list = Default_Learning_Rate_Inc_List
        self.step_lrnrate_max_list = Default_Learning_Rate_Max_List
        self.step_lrnrate_inc_list = Default_Learning_Rate_Inc_List
    
        self.starting_lrndecay_list = Default_Learning_Decay_List
        self.starting_lrndecay_min_list = Default_Learning_Decay_Min_List
        self.starting_lrndecay_max_list = Default_Learning_Decay_Max_List
        self.starting_lrndecay_inc_list = Default_Learning_Decay_Inc_List
        self.ending_lrndecay_list = Default_Learning_Decay_List
        self.ending_lrndecay_min_list = Default_Learning_Decay_Min_List
        self.ending_lrndecay_max_list = Default_Learning_Decay_Max_List
        self.ending_lrndecay_inc_list = Default_Learning_Decay_Inc_List
        self.step_lrndecay_list = Default_Learning_Decay_Inc_List
        self.step_lrndecay_min_list = Default_Learning_Decay_Inc_List
        self.step_lrndecay_max_list = Default_Learning_Decay_Max_List
        self.step_lrndecay_inc_list = Default_Learning_Decay_Inc_List
    
        self.ce_starting_stdev_list = Default_CE_StDev_List
        self.ce_starting_stdev_min_list = Default_CE_StDev_Min_List
        self.ce_starting_stdev_max_list = Default_CE_StDev_Max_List
        self.ce_starting_stdev_inc_list = Default_CE_StDev_Inc_List
        self.ce_ending_stdev_list = Default_CE_StDev_List
        self.ce_ending_stdev_min_list = Default_CE_StDev_Min_List
        self.ce_ending_stdev_max_list = Default_CE_StDev_Max_List
        self.ce_ending_stdev_inc_list = Default_CE_StDev_Inc_List
        self.ce_step_stdev_list = Default_CE_StDev_Inc_List
        self.ce_step_stdev_min_list = Default_CE_StDev_Inc_List
        self.ce_step_stdev_max_list = Default_CE_StDev_Max_List
        self.ce_step_stdev_inc_list = Default_CE_StDev_Inc_List
    
        self.starting_num_steps_per_epoch_list = Default_Number_Steps_Per_Epoch_List
        self.starting_num_steps_per_epoch_min_list = Default_Number_Steps_Per_Epoch_Min_List
        self.starting_num_steps_per_epoch_max_list = Default_Number_Steps_Per_Epoch_Max_List
        self.starting_num_steps_per_epoch_inc_list = Default_Number_Steps_Per_Epoch_Inc_List
        self.ending_num_steps_per_epoch_list = Default_Number_Steps_Per_Epoch_List
        self.ending_num_steps_per_epoch_min_list = Default_Number_Steps_Per_Epoch_Min_List
        self.ending_num_steps_per_epoch_max_list = Default_Number_Steps_Per_Epoch_Max_List
        self.ending_num_steps_per_epoch_inc_list = Default_Number_Steps_Per_Epoch_Inc_List
        self.step_num_steps_per_epoch_list = Default_Number_Steps_Per_Epoch_List
        self.step_num_steps_per_epoch_min_list = Default_Number_Steps_Per_Epoch_Inc_List
        self.step_num_steps_per_epoch_max_list = Default_Number_Steps_Per_Epoch_Max_List
        self.step_num_steps_per_epoch_inc_list = Default_Number_Steps_Per_Epoch_Inc_List
    
        self.num_steps_per_report = NUMBER_OF_STEPS_PER_REPORT
        self.num_steps_per_epoch = NUMBER_STEPS_PER_EPOCH

        self.data_list = []
    
        # working_dir = u"c:\\Users\\Chester\\PythonSourceCodes\\HyperparameterSearcher"
    
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
    
        _user_data_filename = os.path.join(WORKING_DIR, "user_reset_data.txt")
        if os.path.exists(_user_data_filename):
        
            try:
                _user_data_file = open(_user_data_filename, 'r')
                try:
                    self.user_data_list = json.load(_user_data_file)
                    self.empty_user_data_dict_flag = False
                except:
                    # json file is empty or corrupted. Replace it with the original template
                    self.empty_user_data_dict_flag = True
                    _user_data_file.close()
                _user_data_file.close()
            except IOError:
                print("cannot open and read ", _user_data_filename)
    
        _color_data_filename = os.path.join(WORKING_DIR, "color_data.txt")
        if os.path.exists(_color_data_filename):
    
            try:
                _color_data_file = open(_color_data_filename, 'r')
                try:
                    (primary_rgb, secondary_rgb,
                     tertiary_rgb, quaternary_rgb, higher_rgb) = json.load(_color_data_file)
                    self.primarycolor = wx.Colour(primary_rgb)
                    self.secondarycolor = wx.Colour(secondary_rgb)
                    # noinspection PyUnresolvedReferences
                    self.tertiarycolor = wx.Colour(tertiary_rgb)
                    self.quaternarycolor = wx.Colour(quaternary_rgb)
                    self.highercolor = wx.Colour(higher_rgb)
                    self.color_data_unchanged_flag = False
                except:
                    # json file is empty or corrupted. Replace it with the original template
                    self.color_data_unchanged_flag = True
                    _color_data_file.close()
                _color_data_file.close()
            except IOError:
                print("cannot open and read ", _color_data_filename)
    
        _hierarchy_data_filename = os.path.join(WORKING_DIR, "hierarchy_data.txt")
        if os.path.exists(_hierarchy_data_filename):
            try:
                _hierarchy_data_file = open(_hierarchy_data_filename, 'r')
                try:
                    ordering_dict = json.load(_hierarchy_data_file)
                    self.ordering_list.clear()
                    self.ordering_list = [ordering_dict[repr(x)] for x in range(len(ordering_dict))]
                    self.ordering_hierarchy_unchanged_flag = False
                except:
                    # json file is empty or corrupted. Replace it with the original template
                    self.ordering_hierarchy_unchanged_flag = True
                    _hierarchy_data_file.close()
                _hierarchy_data_file.close()
            except IOError:
                print("cannot open and read ", _hierarchy_data_filename)
    
        self.nt_current_wt_list = []
        self.nt_current_bias_list = []
        self.nt_current_stdev_list = []
        self.nt_current_channels_list = []
        self.nt_current_outchannels_list = []
        self.nt_current_filterheight_list = []
        self.nt_current_filterwidth_list = []
        self.nt_current_stride_list = []
        self.dl_current_wt_list = []
        self.dl_current_bias_list = []
        self.dl_current_stdev_list = []
        self.dl_current_keepprob_list = []
        self.dl_current_nodes_list = []
        self.smd_current_wt = None
        self.smd_current_bias = None
        self.smd_current_stdev = None
        self.current_learningrate = None
        self.current_learningdecay = None
        self.ce_current_stdev = None
        
        self.nt_current_hpce_list = []
        self.nt_current_wpce_list = []
        self.mp_current_ksize_list = []
        self.mp_current_stride_list = []
        self.mp_current_hpce_list = []
        self.mp_current_wpce_list = []
        self.mp_current_fltrhght_list = []
        self.mp_current_fltrwdth_list = []
    
        self.widget_dict = {}
       
        self.nt_title = u"Convolution"
        self.nt_display_list = [("CHANNELS", self.nt_starting_chnnl_list, Default_CNN_Channels_List)]
        self.nt_parameter_list = [("WEIGHT", self.nt_starting_wt_list, self.nt_ending_wt_list, self.nt_step_wt_list,
                                   self.nt_starting_wt_min_list, self.nt_starting_wt_max_list,
                                   self.nt_starting_wt_inc_list,
                                   self.nt_ending_wt_min_list, self.nt_ending_wt_max_list, self.nt_ending_wt_inc_list,
                                   self.nt_step_wt_min_list, self.nt_step_wt_max_list, self.nt_step_wt_inc_list,
                                   Default_CNN_Weight_List, Default_CNN_Weight_Min_List, Default_CNN_Weight_Max_List,
                                   Default_CNN_Weight_Inc_List),
                                  ("BIASES", self.nt_starting_bias_list, self.nt_ending_bias_list,
                                   self.nt_step_bias_list,
                                   self.nt_starting_bias_min_list, self.nt_starting_bias_max_list,
                                   self.nt_starting_bias_inc_list,
                                   self.nt_ending_bias_min_list, self.nt_ending_bias_max_list,
                                   self.nt_ending_bias_inc_list,
                                   self.nt_step_bias_min_list, self.nt_step_bias_max_list, self.nt_step_bias_inc_list,
                                   Default_CNN_Biases_List, Default_CNN_Biases_Min_List, Default_CNN_Biases_Max_List,
                                   Default_CNN_Biases_Inc_List),
                                  ("ST. DEV.", self.nt_starting_stdev_list, self.nt_ending_stdev_list,
                                   self.nt_step_stdev_list,
                                   self.nt_starting_stdev_min_list, self.nt_starting_stdev_max_list,
                                   self.nt_starting_stdev_inc_list,
                                   self.nt_ending_stdev_min_list, self.nt_ending_stdev_max_list,
                                   self.nt_ending_stdev_inc_list,
                                   self.nt_step_stdev_min_list, self.nt_step_stdev_max_list,
                                   self.nt_step_stdev_inc_list,
                                   Default_CNN_StDev_List, Default_CNN_StDev_Min_List, Default_CNN_StDev_Max_List,
                                   Default_CNN_StDev_Inc_List),
                                  ("OUT CHANNELS", self.nt_starting_outchnnl_list, self.nt_ending_outchnnl_list,
                                   self.nt_step_outchnnl_list,
                                   self.nt_starting_outchnnl_min_list, self.nt_starting_outchnnl_max_list,
                                   self.nt_starting_outchnnl_inc_list,
                                   self.nt_ending_outchnnl_min_list, self.nt_ending_outchnnl_max_list,
                                   self.nt_ending_outchnnl_inc_list,
                                   self.nt_step_outchnnl_min_list, self.nt_step_outchnnl_max_list,
                                   self.nt_step_outchnnl_inc_list,
                                   Default_CNN_OutChannels_List, Default_CNN_OutChannels_Min_List,
                                   Default_CNN_OutChannels_Max_List, Default_CNN_OutChannels_Inc_List),
                                  ("FILTER HEIGHT", self.nt_starting_fltrhght_list, self.nt_ending_fltrhght_list,
                                   self.nt_step_fltrhght_list,
                                   self.nt_starting_fltrhght_min_list, self.nt_starting_fltrhght_max_list,
                                   self.nt_starting_fltrwdth_inc_list,
                                   self.nt_ending_fltrhght_min_list, self.nt_ending_fltrhght_max_list,
                                   self.nt_ending_fltrhght_inc_list,
                                   self.nt_step_fltrhght_min_list, self.nt_step_fltrhght_max_list,
                                   self.nt_step_fltrhght_inc_list,
                                   Default_CNN_FilterHeight_List, Default_CNN_FilterHeight_Min_List,
                                   Default_CNN_FilterHeight_Max_List, Default_CNN_FilterHeight_Inc_List),
                                  ("FILTER WIDTH", self.nt_starting_fltrwdth_list, self.nt_ending_fltrwdth_list,
                                   self.nt_step_fltrwdth_list,
                                   self.nt_starting_fltrwdth_min_list, self.nt_starting_fltrhght_max_list,
                                   self.nt_starting_fltrwdth_inc_list,
                                   self.nt_ending_fltrwdth_min_list, self.nt_ending_fltrwdth_max_list,
                                   self.nt_ending_fltrwdth_inc_list,
                                   self.nt_step_fltrhght_min_list, self.nt_step_fltrwdth_max_list,
                                   self.nt_step_fltrwdth_inc_list,
                                   Default_CNN_FilterWidth_List, Default_CNN_FilterWidth_Min_List,
                                   Default_CNN_FilterWidth_Max_List, Default_CNN_FilterWidth_Inc_List)]
    
        self.nt_pace_list = [("NEURONETWORK STRIDE", self.nt_starting_strd_sel_list, self.nt_ending_strd_sel_list,
                              self.nt_strd_step_list,
                              self.nt_starting_hpce_list, self.nt_starting_wpce_list,
                              self.nt_ending_hpce_list, self.nt_ending_wpce_list,
                              self.nt_strd_sel_min_list, self.nt_strd_sel_max_list, self.nt_strd_sel_inc_list,
                              Default_CNN_Stride_Selection_List, Default_CNN_HPace_List, Default_CNN_WPACE_List,
                              Default_CNN_Stride_Sel_Min_List, Default_CNN_Stride_Sel_Max_List,
                              Default_CNN_Stride_Sel_Inc_List),
                             ("MAX POOL STRIDE", self.mp_starting_strd_sel_list, self.mp_ending_strd_sel_list,
                              self.mp_strd_step_list,
                              self.mp_starting_hpce_list, self.mp_starting_wpce_list,
                              self.mp_ending_hpce_list, self.mp_ending_wpce_list,
                              self.mp_strd_sel_min_list, self.mp_strd_sel_max_list, self.mp_strd_sel_inc_list,
                              Default_MP_Stride_Selection_List, Default_MP_HPace_List, Default_MP_WPace_List,
                              Default_MP_Stride_Sel_Min_List, Default_MP_Stride_Sel_Max_List,
                              Default_MP_Stride_Sel_Inc_List),
                             ("MAX POOL KSIZE", self.mp_starting_ksze_sel_list, self.mp_ending_ksze_sel_list,
                              self.mp_ksze_step_list,
                              self.mp_starting_fltrhght_list, self.mp_starting_fltrwdth_list,
                              self.mp_ending_fltrhght_list, self.mp_ending_fltrwdth_list,
                              self.mp_ksze_sel_min_list, self.mp_ksze_sel_max_list, self.mp_ksze_sel_inc_list,
                              Default_MP_KSize_Selection_List, Default_MP_Filter_Height_List,
                              Default_MP_Filter_Width_List,
                              Default_MP_KSize_Sel_Min_List, Default_MP_KSize_Sel_Max_List,
                              Default_MP_KSize_Sel_Inc_List)]
            
        self.dl_title = u"Multi-neuron"
        self.dl_display_list = []
        self.dl_parameter_list = [("WEIGHT", self.dl_starting_wt_list, self.dl_ending_wt_list, self.dl_step_wt_list,
                                   self.dl_starting_wt_min_list, self.dl_starting_wt_max_list,
                                   self.dl_starting_wt_inc_list,
                                   self.dl_ending_wt_min_list, self.dl_ending_wt_max_list, self.dl_ending_wt_inc_list,
                                   self.dl_step_wt_min_list, self.dl_step_wt_max_list, self.dl_step_wt_inc_list,
                                   Default_DL_Weight_List, Default_DL_Weight_Min_List, Default_DL_Weight_Max_List,
                                   Default_DL_Weight_Inc_List),
                                  ("BIASES", self.dl_starting_bias_list, self.dl_ending_bias_list,
                                   self.dl_step_bias_list,
                                   self.dl_starting_bias_min_list, self.dl_starting_bias_max_list,
                                   self.dl_starting_bias_inc_list,
                                   self.dl_ending_bias_min_list, self.dl_ending_bias_max_list,
                                   self.dl_ending_bias_inc_list,
                                   self.dl_step_bias_min_list, self.dl_step_bias_max_list, self.dl_step_bias_inc_list,
                                   Default_DL_Biases_List, Default_DL_Biases_Min_List, Default_DL_Biases_Max_List,
                                   Default_DL_Biases_Inc_List),
                                  ("ST. DEV.", self.dl_starting_stdev_list, self.dl_ending_stdev_list,
                                   self.dl_step_stdev_list,
                                   self.dl_starting_stdev_min_list, self.dl_starting_stdev_max_list,
                                   self.dl_starting_stdev_inc_list,
                                   self.dl_ending_stdev_min_list, self.dl_ending_stdev_max_list,
                                   self.dl_ending_stdev_inc_list,
                                   self.dl_step_stdev_min_list, self.dl_step_stdev_max_list,
                                   self.dl_step_stdev_inc_list,
                                   Default_DL_StDev_List, Default_DL_StDev_Min_List, Default_DL_StDev_Max_List,
                                   Default_DL_StDev_Inc_List),
                                  ("KEEP PROB.", self.dl_starting_kpprb_list, self.dl_ending_kpprb_list,
                                   self.dl_step_kpprb_list,
                                   self.dl_starting_kpprb_min_list, self.dl_starting_kpprb_max_list,
                                   self.dl_starting_kpprb_inc_list,
                                   self.dl_ending_kpprb_min_list, self.dl_ending_kpprb_max_list,
                                   self.dl_ending_kpprb_inc_list,
                                   self.dl_step_kpprb_min_list, self.dl_step_kpprb_max_list,
                                   self.dl_step_kpprb_inc_list,
                                   Default_DL_KeepProb_List, Default_DL_KeepProb_Min_List, Default_DL_KeepProb_Max_List,
                                   Default_DL_KeepProb_Inc_List),
                                  ("NODES", self.dl_starting_nodes_list, self.dl_ending_nodes_list,
                                   self.dl_step_nodes_list,
                                   self.dl_starting_nodes_min_list, self.dl_starting_nodes_max_list,
                                   self.dl_starting_nodes_inc_list,
                                   self.dl_ending_nodes_min_list, self.dl_ending_nodes_max_list,
                                   self.dl_ending_nodes_inc_list,
                                   self.dl_step_nodes_min_list, self.dl_step_nodes_max_list,
                                   self.dl_step_nodes_inc_list,
                                   Default_DL_Nodes_List, Default_DL_Nodes_Min_List, Default_DL_Nodes_Max_List,
                                   Default_DL_Nodes_Inc_List)]
        self.dl_pace_list = []
    
        self.cm_title = u"Custom"
        self.cm_display_list = []
        self.cm_parameter_list = []
        self.cm_pace_list = []
    
        self.smd_title = u"Sigmoid"
        self.smd_display_list = []
        self.smd_parameter_list = [("WEIGHT", self.smd_starting_wt_list, self.smd_ending_wt_list, self.smd_step_wt_list,
                                    self.smd_starting_wt_min_list, self.smd_starting_wt_max_list,
                                    self.smd_starting_wt_inc_list,
                                    self.smd_ending_wt_min_list, self.smd_ending_wt_max_list,
                                    self.smd_ending_wt_inc_list,
                                    self.smd_step_wt_min_list, self.smd_step_wt_max_list, self.smd_step_wt_inc_list,
                                    Default_SMD_Weight_List, Default_SMD_Weight_Min_List, Default_SMD_Weight_Max_List,
                                    Default_SMD_Weight_Inc_List),
                                   ("BIASES", self.smd_starting_bias_list, self.smd_ending_bias_list,
                                    self.smd_step_bias_list,
                                    self.smd_starting_wt_min_list, self.smd_starting_wt_max_list,
                                    self.smd_starting_wt_inc_list,
                                    self.smd_ending_bias_min_list, self.smd_ending_bias_max_list,
                                    self.smd_ending_bias_inc_list,
                                    self.smd_step_bias_min_list, self.smd_step_bias_max_list,
                                    self.smd_step_bias_inc_list,
                                    Default_SMD_Bias_List, Default_SMD_Biases_Min_List, Default_SMD_Biases_Max_List,
                                    Default_SMD_Biases_Inc_List),
                                   ("ST. DEV.", self.smd_starting_stdev_list, self.smd_ending_stdev_list,
                                    self.smd_step_stdev_list,
                                    self.smd_starting_stdev_min_list, self.smd_starting_stdev_max_list,
                                    self.smd_starting_stdev_inc_list,
                                    self.smd_ending_stdev_min_list, self.smd_ending_stdev_max_list,
                                    self.smd_ending_stdev_inc_list,
                                    self.smd_step_stdev_min_list, self.smd_step_stdev_max_list,
                                    self.smd_step_stdev_inc_list,
                                    Default_SMD_StDev_List, Default_SMD_StDev_Min_List, Default_SMD_StDev_Max_List,
                                    Default_SMD_StDev_Inc_List)]
        self.smd_pace_list = []
    
        self.learning_title = u"Learning"
        self.learning_display_list = []
        self.learning_parameter_list = [("LEARNING RATE", self.starting_lrnrate_list, self.ending_lrnrate_list,
                                         self.step_lrnrate_list,
                                         self.starting_lrnrate_min_list, self.starting_lrnrate_max_list,
                                         self.starting_lrnrate_inc_list,
                                         self.ending_lrnrate_min_list, self.ending_lrnrate_max_list,
                                         self.ending_lrnrate_inc_list,
                                         self.step_lrnrate_min_list, self.step_lrnrate_max_list,
                                         self.step_lrnrate_inc_list,
                                         Default_Learning_Rate_List, Default_Learning_Rate_Min_List,
                                         Default_Learning_Rate_Max_List, Default_Learning_Rate_Inc_List),
                                        ("LEARNING DECAY", self.starting_lrndecay_list, self.ending_lrndecay_list,
                                         self.step_lrndecay_list,
                                         self.starting_lrndecay_min_list, self.starting_lrndecay_max_list,
                                         self.starting_lrndecay_inc_list,
                                         self.ending_lrndecay_min_list, self.ending_lrndecay_max_list,
                                         self.ending_lrndecay_inc_list,
                                         self.step_lrndecay_min_list, self.step_lrndecay_max_list,
                                         self.step_lrndecay_inc_list,
                                         Default_Learning_Decay_List, Default_Learning_Decay_Min_List,
                                         Default_Learning_Decay_Max_List, Default_Learning_Decay_Inc_List)]
    
        self.learning_pace_list = []
    
        self.ce_title = u"Cross-entropy"
        self.ce_display_list = []
        self.ce_parameter_list = [("ST. DEV.", self.ce_starting_stdev_list, self.ce_ending_stdev_list,
                                   self.ce_step_stdev_list,
                                   self.ce_starting_stdev_min_list, self.ce_starting_stdev_max_list,
                                   self.ce_starting_stdev_inc_list,
                                   self.ce_ending_stdev_min_list, self.ce_ending_stdev_max_list,
                                   self.ce_ending_stdev_inc_list,
                                   self.ce_step_stdev_min_list, self.ce_step_stdev_max_list,
                                   self.ce_step_stdev_inc_list,
                                   Default_CE_StDev_List, Default_CE_StDev_Min_List, Default_CE_StDev_Max_List,
                                   Default_CE_StDev_Inc_List)]
        self.ce_pace_list = []
    
        self.num_steps_title = u"Number of Steps"
        self.num_steps_display_list = [("TOTAL NUM EPOCHS", [" "], [" "]),
                                       ("EST. TIME PER EPOCH", [" "], [" "]),
                                       ("TIME ELAPSED", [" "], [" "]),
                                       ("EST. TIME TO FINISH", [" "], [" "])]
        self.num_steps_parameter_list = [("STEPS PER EPOCH", self.step_num_steps_per_epoch_list, None, None,
                                          self.step_num_steps_per_epoch_min_list,
                                          self.step_num_steps_per_epoch_max_list,
                                          self.step_num_steps_per_epoch_inc_list,
                                          None, None, None, None, None, None,
                                          Default_Number_Steps_Per_Epoch_List, Default_Number_Steps_Per_Epoch_Min_List,
                                          Default_Number_Steps_Per_Epoch_Max_List,
                                          Default_Number_Steps_Per_Epoch_Inc_List)]
        self.num_steps_pace_list = []
    
        # main_boxsizer = wx.BoxSizer(wx.VERTICAL)
        # noinspection PyUnresolvedReferences
        self.display_x, _ = wx.GetDisplaySize()
    
        self.layer_type = None
        self.layer_num = None
    
        self.title = None
        self.display_list = []
        self.display_num = None
        self.parameter_num = None
        self.parameter_list = []
        self.pace_list = []
        self.pace_num = None
        
        self.saved_display_reset = None
        self.saved_display_reset_dict = {}
        
        self.starting_holder_list = []
        self.ending_holder_list = []
        self.step_holder_list = []
        self.name_holder_list = []
        self.ongoing_holder_list = []
        self.static_text_to_list = []
        self.static_text_step_list = []
    
        widget_group_index = -1
        static_text_index = -1
        user_data_list_index = -1

        _choices = ['[1, 1, 1, 1]', '[1, 2, 2, 1]', '[1, 3, 3, 1]', '[1, 4, 4, 1]',
                    '[1, 5, 5, 1]', '[1, 6, 6, 1]', '[1, 7, 7, 1]', '[1, 8, 8, 1]',
                    '[1, 9, 9, 1]', '[1, 10, 10, 1]', '[1, 11, 11, 1]', '[1, 12, 12, 1]',
                    '[1, 13, 13, 1]', '[1, 14, 14, 1]', '[1, 15, 15, 1]', '[1, 16, 16, 1]',
                    '[1, 17, 17, 1]', '[1, 18, 18, 1]', '[1, 19, 19, 1]', '[1, 20, 20, 1]',
                    '[1, 21, 21, 1]', '[1, 22, 22, 1]', '[1, 23, 23, 1]', '[1, 24, 24, 1]',
                    '[1, 25, 25, 1]', '[1, 26, 26, 1]', '[1, 27, 27, 1]', '[1, 28, 28, 1]',
                    '[1, 29, 29, 1]', '[1, 30, 30, 1]', '[1, 31, 31, 1]', '[1, 32, 32, 1]',
                    '[1, 33, 33, 1]', '[1, 34, 34, 1]', '[1, 35, 35, 1]', '[1, 36, 36, 1]',
                    '[1, 37, 37, 1]', '[1, 38, 38, 1]', '[1, 39, 39, 1]', '[1, 40, 40, 1]',
                    '[1, 41, 41, 1]', '[1, 42, 42, 1]', '[1, 43, 43, 1]', '[1, 44, 44, 1]',
                    '[1, 45, 45, 1]', '[1, 46, 46, 1]', '[1, 47, 47, 1]', '[1, 48, 48, 1]',
                    '[1, 49, 49, 1]', '[1, 50, 50, 1]', '[1, 51, 51, 1]', '[1, 52, 52, 1]',
                    '[1, 53, 53, 1]', '[1, 54, 54, 1]', '[1, 55, 55, 1]', '[1, 56, 56, 1]',
                    '[1, 57, 57, 1]', '[1, 58, 58, 1]', '[1, 59, 59, 1]', '[1, 60, 60, 1]',
                    '[1, 61, 61, 1]', '[1, 62, 62, 1]', '[1, 63, 63, 1]', '[1, 64, 64, 1]',
                    '[1, 65, 65, 1]', '[1, 66, 66, 1]', '[1, 67, 67, 1]', '[1, 68, 68, 1]',
                    '[1, 69, 69, 1]', '[1, 70, 70, 1]', '[1, 71, 71, 1]', '[1, 72, 72, 1]',
                    '[1, 73, 73, 1]', '[1, 74, 74, 1]', '[1, 75, 75, 1]', '[1, 76, 76, 1]',
                    '[1, 77, 77, 1]', '[1, 78, 78, 1]', '[1, 79, 79, 1]', '[1, 80, 80, 1]',
                    '[1, 81, 81, 1]', '[1, 82, 82, 1]', '[1, 83, 83, 1]', '[1, 84, 84, 1]',
                    '[1, 85, 85, 1]', '[1, 86, 86, 1]', '[1, 87, 87, 1]', '[1, 88, 88, 1]',
                    '[1, 89, 89, 1]', '[1, 90, 90, 1]', '[1, 91, 91, 1]', '[1, 92, 92, 1]',
                    '[1, 93, 93, 1]', '[1, 94, 94, 1]', '[1, 95, 95, 1]', '[1, 96, 96, 1]',
                    '[1, 97, 97, 1]', '[1, 98, 98, 1]', '[1, 99, 99, 1]', '[1, 100, 100, 1]',

                    '[1, 101, 101, 1]', '[1, 102, 102, 1]', '[1, 103, 103, 1]', '[1, 104, 104, 1]',
                    '[1, 105, 105, 1]', '[1, 106, 106, 1]', '[1, 107, 107, 1]', '[1, 108, 108, 1]',
                    '[1, 109, 109, 1]', '[1, 110, 110, 1]', '[1, 111, 111, 1]', '[1, 112, 112, 1]',
                    '[1, 113, 113, 1]', '[1, 114, 114, 1]', '[1, 115, 115, 1]', '[1, 116, 116, 1]',
                    '[1, 117, 117, 1]', '[1, 118, 118, 1]', '[1, 119, 119, 1]', '[1, 120, 120, 1]',
                    '[1, 121, 121, 1]', '[1, 122, 122, 1]', '[1, 123, 123, 1]', '[1, 124, 124, 1]',
                    '[1, 125, 125, 1]', '[1, 126, 126, 1]', '[1, 127, 127, 1]', '[1, 128, 128, 1]',
                    '[1, 129, 129, 1]', '[1, 130, 130, 1]', '[1, 131, 131, 1]', '[1, 132, 132, 1]',
                    '[1, 133, 133, 1]', '[1, 134, 134, 1]', '[1, 135, 135, 1]', '[1, 136, 136, 1]',
                    '[1, 137, 137, 1]', '[1, 138, 138, 1]', '[1, 139, 139, 1]', '[1, 140, 140, 1]',
                    '[1, 141, 141, 1]', '[1, 142, 142, 1]', '[1, 143, 143, 1]', '[1, 144, 144, 1]',
                    '[1, 145, 145, 1]', '[1, 146, 146, 1]', '[1, 147, 147, 1]', '[1, 148, 148, 1]',
                    '[1, 149, 149, 1]', '[1, 150, 150, 1]', '[1, 151, 151, 1]', '[1, 152, 152, 1]',
                    '[1, 153, 153, 1]', '[1, 154, 154, 1]', '[1, 155, 515, 1]', '[1, 156, 156, 1]',
                    '[1, 157, 157, 1]', '[1, 158, 158, 1]', '[1, 159, 159, 1]', '[1, 160, 160, 1]',
                    '[1, 161, 161, 1]', '[1, 162, 162, 1]', '[1, 163, 163, 1]', '[1, 164, 164, 1]',
                    '[1, 165, 165, 1]', '[1, 166, 166, 1]', '[1, 167, 167, 1]', '[1, 168, 168, 1]',
                    '[1, 169, 169, 1]', '[1, 170, 170, 1]', '[1, 171, 171, 1]', '[1, 172, 172, 1]',
                    '[1, 173, 173, 1]', '[1, 174, 174, 1]', '[1, 175, 175, 1]', '[1, 176, 176, 1]',
                    '[1, 177, 177, 1]', '[1, 178, 178, 1]', '[1, 179, 179, 1]', '[1, 180, 180, 1]',
                    '[1, 181, 181, 1]', '[1, 182, 182, 1]', '[1, 183, 183, 1]', '[1, 184, 184, 1]',
                    '[1, 185, 185, 1]', '[1, 186, 186, 1]', '[1, 187, 187, 1]', '[1, 188, 188, 1]',
                    '[1, 189, 189, 1]', '[1, 190, 190, 1]', '[1, 191, 191, 1]', '[1, 192, 192, 1]',
                    '[1, 193, 193, 1]', '[1, 194, 194, 1]', '[1, 195, 195, 1]', '[1, 196, 196, 1]',
                    '[1, 197, 197, 1]', '[1, 198, 198, 1]', '[1, 199, 199, 1]', '[1, 200, 200, 1]']
        
        for layer in range(len(Logit_list)):
            self.layer_type, self.layer_num = Logit_list[layer]
            if self.layer_type == CONVOLUTION_LAYER:
                self.title = self.nt_title
                self.display_list = self.nt_display_list
                self.display_num = len(self.nt_display_list)
                self.parameter_list = self.nt_parameter_list
                self.parameter_num = len(self.nt_parameter_list)
                self.pace_list = self.nt_pace_list
                self.pace_num = len(self.nt_pace_list)
        
            elif self.layer_type == MULTI_NEURON_LAYER:
                self.title = self.dl_title
                self.display_list = self.dl_display_list
                self.display_num = len(self.dl_display_list)
                self.parameter_list = self.dl_parameter_list
                self.parameter_num = len(self.dl_parameter_list)
                self.pace_list = self.dl_pace_list
                self.pace_num = len(self.dl_pace_list)
        
            elif self.layer_type == CUSTOM_LAYER:
                self.title = self.cm_title
                self.display_list = self.cm_display_list
                self.display_num = len(self.cm_display_list)
                self.parameter_list = self.cm_parameter_list
                self.parameter_num = len(self.cm_parameter_list)
                self.pace_list = self.cm_pace_list
                self.pace_num = len(self.cm_pace_list)
        
            elif self.layer_type == SIMOID_LAYER:
                self.title = self.smd_title
                self.display_list = self.smd_display_list
                self.display_num = len(self.smd_display_list)
                self.parameter_list = self.smd_parameter_list
                self.parameter_num = len(self.smd_parameter_list)
                self.pace_list = self.smd_pace_list
                self.pace_num = len(self.smd_pace_list)
        
            elif self.layer_type == CROSS_ENTROPY_LAYER:
                self.title = self.ce_title
                self.display_list = self.ce_display_list
                self.display_num = len(self.ce_display_list)
                self.parameter_list = self.ce_parameter_list
                self.parameter_num = len(self.ce_parameter_list)
                self.pace_list = self.ce_pace_list
                self.pace_num = len(self.ce_pace_list)
        
            elif self.layer_type == LEARNING_LAYER:
                self.title = self.learning_title
                self.display_list = self.learning_display_list
                self.display_num = len(self.learning_display_list)
                self.parameter_list = self.learning_parameter_list
                self.parameter_num = len(self.learning_parameter_list)
                self.pace_list = self.learning_pace_list
                self.pace_num = len(self.learning_pace_list)
        
            elif self.layer_type == NUMBER_OF_STEPS_LAYER:
                self.title = self.num_steps_title
                self.display_list = self.num_steps_display_list
                self.display_num = len(self.num_steps_display_list)
                self.parameter_list = self.num_steps_parameter_list
                self.parameter_num = len(self.num_steps_parameter_list)
                self.pace_list = self.num_steps_pace_list
                self.pace_num = len(self.num_steps_pace_list)
        
            elif self.layer_type == CROSS_ENTROPY_SIGMOID_LAYER:
                pass
        
            # last_row_display_sizer = wx.BoxSizer(wx.HORIZONTAL)

            # endoflayer_flag = False
            for x in range(self.layer_num):
                endoflayer_flag = False
                line_count = 0
                total_line_per_block = self.display_num + self.parameter_num + self.pace_num
                if self.layer_num == 1:
                    _label = self.title
                else:
                    _label = self.title + repr(x + 1)
                
                for w in range(self.display_num):
                    endofblock_flag = False
                    widget_group_index += 1
                    line_count += 1
                    # starting_initial = default_initial = None
                    (name, starting_var_list, default_var_list) = self.display_list[w]
                   
                    if len(starting_var_list) > 1:
                        if len(starting_var_list) - 1 < x:
                            starting_initial = starting_var_list[len(starting_var_list) - 1]
                            default_initial = default_var_list[len(default_var_list) - 1]
                        else:
                            starting_initial = starting_var_list[x]
                            default_initial = default_var_list[x]
                    else:
                        starting_initial = starting_var_list[0]
                        default_initial = default_var_list[0]
                    
                    saved_display_flag = False
                    
                    # This following set of line codes is to communicate with
                    # the outchannel spindle double control of the previous controls blocks in the layer
                    # via the self.save_display_reset, which stays as None until the outchannel spindle control is set
                    # up at self.init_ui() method. The self.save_display_reset simply points
                    # to the outchannel spindledouble control of the preious block of the same layer
                    if name == "CHANNELS":
                        
                        if x > 0:
                            # The channel display only correlated to the channel spindledouble control
                            # of the preious block of the same layer.
                            starting_initial = int(self.saved_display_reset.GetValue())
                            saved_display_flag = True
                            
                        # Setting up the channel display control for every blcok in the layer.
                        # Putting the starting_initials as the default,
                        # especially for the first channel display control. The subsequent channel display channel
                        # controls will be overwrittten once the previous outchannel spindledouble controls infos
                        # become available
                        # noinspection PyUnresolvedReferences
                        self.starting_holder_list.append(wx.StaticText(self.panel, wx.ID_ANY,
                                                                       label=repr(starting_initial), size=(40, 30),
                                                                       style=wx.ALIGN_CENTRE))
                    else:
                        # Setting up the display control for any other displays other than the channel display controls
                        self.starting_holder_list.append(wx.StaticText(self.panel, wx.ID_ANY,
                                                                       label=" ", size=(40, 30),
                                                                       style=wx.ALIGN_CENTRE))
                   
                    if saved_display_flag:
                        # Save the channel display in a dictionary
                        get_id = wx.Window.GetId(self.saved_display_reset)
                        self.saved_display_reset_dict[get_id] = self.starting_holder_list[widget_group_index]
                        # Clear the self.save_display_reset for the next outchannel spindledouble control
                        # to set the flag again
                        self.saved_display_reset = None
                    
                    # This is a display control. There is no ending_holder or step_holder controls
                    self.ending_holder_list.append(None)
                    self.step_holder_list.append(None)
                    
                    # Setting up the name of the display control
                    self.name_holder_list.append(wx.StaticText(self.panel, wx.ID_ANY, name, size=(100, 30)))
                    # Setting up the display ongoing holdder controls for future usage
                    style = wx.TE_CENTRE | wx.TE_READONLY | wx.NO_BORDER | wx.TE_RICH
                    self.ongoing_holder_list.append(wx.TextCtrl(self.panel, wx.ID_ANY, size=(40, 30),
                                                                style=style))
                    self.ongoing_holder_list[widget_group_index].SetBackgroundColour(Default_Background_Color)
                    self.ongoing_holder_list[widget_group_index].SetForegroundColour(wx.Colour(55, 63, 65))
                    # self.ongoing_holder_list[widget_group_index].SetDefaultStyle(wx.TextAttr(wx.Colour(55, 63, 65)))
                    self.ongoing_holder_list[widget_group_index].SetFont(wx.Font(12, 70, 90, 92, False, "  "))
                    # self.ongoing_holder_list[widget_group_index].SetForegroundColour(wx.Colour(55, 63, 65))
                    
                    # if y == self.display_num - 1 and y == self.pace_num == 0 and self.parameter_num == 0:
                    if line_count >= total_line_per_block:
                        endofblock_flag = True
                        line_count = 0
                        if x == self.layer_num - 1:
                            endoflayer_flag = True
                            if self.title == u"Sigmoid":
                                endoflayer_flag = False
                            elif self.title == u"Learning":
                                endoflayer_flag = False
                            elif self.title == u"Cross-entropy":
                                endoflayer_flag = False

                    self.data_list.append((DISPLAY_CTRL, _label, name, None,
                                           starting_initial, None, None,
                                           None, None, None,
                                           None, None, None,
                                           None, None, None,
                                           default_initial, None, None, None,
                                           endofblock_flag, endoflayer_flag))

                    starting_id = wx.Window.GetId(self.starting_holder_list[widget_group_index])
                    self.widget_dict[starting_id] = (widget_group_index, STARTING_HOLDER)
                
                for y in range(self.parameter_num):
                    endofblock_flag = False
                    line_count += 1
                    widget_group_index += 1
                    if self.empty_user_data_dict_flag:
                        (name, starting_var_list, ending_var_list, step_initial_list,
                         starting_min_list, starting_max_list, starting_inc_list,
                         ending_min_list, ending_max_list, ending_inc_list,
                         step_min_list, step_max_list, step_inc_list,
                         default_var_list, default_min_list, default_max_list,
                         default_inc_list) = self.parameter_list[y]
        
                        if len(starting_min_list) - 1 < x:
                            starting_min = starting_min_list[len(starting_min_list) - 1]
                        else:
                            starting_min = starting_min_list[x]
        
                        ending_min = None
                        if ending_min_list is not None:
                            # noinspection PyTypeChecker
                            if len(ending_min_list) - 1 < x:
                                # noinspection PyTypeChecker
                                ending_min = ending_min_list[len(ending_min_list) - 1]
                            else:
                                ending_min = ending_min_list[x]
        
                        step_min = None
                        if step_min_list is not None:
                            # noinspection PyTypeChecker
                            if len(step_min_list) - 1 < x:
                                # noinspection PyTypeChecker
                                step_min = step_min_list[len(step_min_list) - 1]
                            else:
                                step_min = step_min_list[x]
        
                        if len(starting_max_list) - 1 < x:
                            starting_max = starting_max_list[len(starting_max_list) - 1]
                        else:
                            starting_max = starting_max_list[x]
        
                        ending_max = None
                        if ending_max_list is not None:
                            # noinspection PyTypeChecker
                            if len(ending_max_list) - 1 < x:
                                # noinspection PyTypeChecker
                                ending_max = ending_max_list[len(ending_max_list) - 1]
                            else:
                                ending_max = ending_max_list[x]
        
                        step_max = None
                        if step_max_list is not None:
                            # noinspection PyTypeChecker
                            if len(step_max_list) - 1 < x:
                                # noinspection PyTypeChecker
                                step_max = step_max_list[len(step_max_list) - 1]
                            else:
                                step_max = step_max_list[x]
        
                        if len(starting_inc_list) - 1 < x:
                            starting_inc = starting_inc_list[len(starting_inc_list) - 1]
                        else:
                            starting_inc = starting_inc_list[x]
        
                        ending_inc = None
                        if ending_inc_list is not None:
                            # noinspection PyTypeChecker
                            if len(ending_inc_list) - 1 < x:
                                # noinspection PyTypeChecker
                                ending_inc = ending_inc_list[len(ending_inc_list) - 1]
                            else:
                                ending_inc = ending_inc_list[x]
        
                        step_inc = None
                        if step_inc_list is not None:
                            # noinspection PyTypeChecker
                            if len(step_inc_list) - 1 < x:
                                # noinspection PyTypeChecker
                                step_inc = step_inc_list[len(step_inc_list) - 1]
                            else:
                                step_inc = step_inc_list[x]
        
                        if len(starting_var_list) - 1 < x:
                            starting_initial = starting_var_list[len(starting_var_list) - 1]
                        else:
                            starting_initial = starting_var_list[x]
        
                        self.starting_holder_list.append(wx.SpinCtrlDouble(self.panel, wx.ID_ANY, size=(60, 20),
                                                                           style=wx.SP_ARROW_KEYS | wx.SP_WRAP,
                                                                           min=starting_min, max=starting_max,
                                                                           initial=starting_initial, inc=starting_inc))
        
                        if ending_var_list is None:
                            self.ending_holder_list.append(None)
                            ending_initial = None
                        else:
                            # noinspection PyTypeChecker
                            if len(ending_var_list) - 1 < x:
                                # noinspection PyTypeChecker
                                ending_initial = ending_var_list[len(ending_var_list) - 1]
                            else:
                                ending_initial = ending_var_list[x]
                            # noinspection PyUnresolvedReferences
                            self.ending_holder_list.append(wx.SpinCtrlDouble(self.panel, wx.ID_ANY, size=(60, 20),
                                                                             style=wx.SP_ARROW_KEYS | wx.SP_WRAP,
                                                                             min=ending_min, max=ending_max,
                                                                             initial=ending_initial, inc=ending_inc))
        
                        if step_initial_list is None:
                            self.step_holder_list.append(None)
                            step_initial = None
                        else:
                            # noinspection PyTypeChecker
                            if len(step_initial_list) - 1 < x:
                                # noinspection PyTypeChecker
                                step_initial = step_initial_list[len(step_initial_list) - 1]
                            else:
                                step_initial = step_initial_list[x]
                            # noinspection PyUnresolvedReferences
                            self.step_holder_list.append(wx.SpinCtrlDouble(self.panel, wx.ID_ANY, size=(60, 20),
                                                                           style=wx.SP_ARROW_KEYS | wx.SP_WRAP,
                                                                           min=step_min, max=step_max / 2,
                                                                           initial=step_initial, inc=step_inc))
                            self.step_holder_list[widget_group_index].Enable(False)
        
                        if len(default_var_list) - 1 < x:
                            default_initial = default_var_list[len(default_var_list) - 1]
                        else:
                            default_initial = default_var_list[x]
        
                        if len(default_min_list) - 1 < x:
                            default_min = default_min_list[len(default_min_list) - 1]
                        else:
                            default_min = default_min_list[x]
        
                        if len(default_max_list) - 1 < x:
                            default_max = default_max_list[len(default_max_list) - 1]
                        else:
                            default_max = default_max_list[x]
        
                        if len(default_inc_list) - 1 < x:
                            default_inc = default_inc_list[len(default_inc_list) - 1]
                        else:
                            default_inc = default_inc_list[x]
    
                    else:
                        user_data_list_index += 1
                        (name, starting_initial, ending_initial, step_initial,
                         starting_min, starting_max, starting_inc,
                         ending_min, ending_max, ending_inc,
                         step_min, step_max, step_inc,
                         default_initial, default_min, default_max,
                         default_inc) = self.user_data_list[user_data_list_index]

                        # noinspection PyUnresolvedReferences
                        self.starting_holder_list.append(wx.SpinCtrlDouble(self.panel, wx.ID_ANY, size=(60, 20),
                                                                           style=wx.SP_ARROW_KEYS | wx.SP_WRAP,
                                                                           min=starting_min, max=starting_max,
                                                                           initial=starting_initial, inc=starting_inc))
        
                        if ending_initial is None:
                            self.ending_holder_list.append(None)
                            ending_initial = None
                        else:
                            self.ending_holder_list.append(wx.SpinCtrlDouble(self.panel, wx.ID_ANY, size=(60, 20),
                                                                             style=wx.SP_ARROW_KEYS | wx.SP_WRAP,
                                                                             min=ending_min, max=ending_max,
                                                                             initial=ending_initial, inc=ending_inc))
        
                        if step_initial is None:
                            self.step_holder_list.append(None)
                            step_initial = None
                        else:
                            # noinspection PyUnresolvedReferences
                            self.step_holder_list.append(wx.SpinCtrlDouble(self.panel, wx.ID_ANY, size=(60, 20),
                                                                           style=wx.SP_ARROW_KEYS | wx.SP_WRAP,
                                                                           min=step_min, max=step_max / 2,
                                                                           initial=step_initial, inc=step_inc))
                            
                            self.step_holder_list[widget_group_index].Enable(False)
                            
                    if name == u"OUT CHANNELS":
                        self.saved_display_reset = self.starting_holder_list[widget_group_index]
                    
                    # name_holder = wx.StaticText(self.panel, wx.ID_ANY, name, size=(80, 30))
                    
                    if not name == u"Number of Steps":
                        # noinspection PyUnresolvedReferences
                        self.name_holder_list.append(wx.StaticText(self.panel, wx.ID_ANY, name, size=(100, 30)))
                        style = wx.TE_CENTRE | wx.TE_READONLY | wx.NO_BORDER | wx.TE_RICH
                        self.ongoing_holder_list.append(wx.TextCtrl(self.panel, wx.ID_ANY, size=(40, 30),
                                                                    style=style))
                        self.ongoing_holder_list[widget_group_index].SetBackgroundColour(Default_Background_Color)
                        self.ongoing_holder_list[widget_group_index].SetForegroundColour(wx.Colour(55, 63, 65))
                        # self.ongoing_holder_list[widget_group_index].SetDefaultStyle(wx.TextAttr(wx.Colour(55, 63, 65)))
                        self.ongoing_holder_list[widget_group_index].SetFont(wx.Font(12, 70, 90, 92, False, "  "))
                        # self.ongoing_holder_list[widget_group_index].SetForegroundColour(wx.Colour(55, 63, 65))
                        
                        static_text_index += 1
                        self.static_text_to_list.append(wx.StaticText(self.panel, wx.ID_ANY, "TO", size=(20, 20)))
                        # noinspection PyUnresolvedReferences
                        self.static_text_to_list[static_text_index].SetForegroundColour(wx.Colour(120, 120, 123))
                        self.static_text_step_list.append(wx.StaticText(self.panel, wx.ID_ANY, "STEP", size=(30, 20)))
                        # noinspection PyUnresolvedReferences
                        self.static_text_step_list[static_text_index].SetForegroundColour(wx.Colour(120, 120, 123))

                    # if y == self.display_num - 1 and y == self.parameter_num - 1 and self.pace_num == 0:
                    if line_count >= total_line_per_block:
                        endofblock_flag = True
                        line_count = 0
                        if x == self.layer_num - 1:
                            endoflayer_flag = True
                            if self.title == u"Sigmoid":
                                endoflayer_flag = False
                            elif self.title == u"Learning":
                                endoflayer_flag = False
                            elif self.title == u"Cross-entropy":
                                endoflayer_flag = False
    
                    self.data_list.append((SPIN_CTRL_DOUBLE, _label, name, None,
                                           starting_initial, ending_initial, step_initial,
                                           starting_min, starting_max, starting_inc,
                                           ending_min, ending_max, ending_inc,
                                           step_min, step_max, step_inc,
                                           default_initial, default_min, default_max, default_inc,
                                           endofblock_flag, endoflayer_flag))
    
                    starting_id = wx.Window.GetId(self.starting_holder_list[widget_group_index])
                    self.widget_dict[starting_id] = (widget_group_index, STARTING_HOLDER)
                    if self.ending_holder_list[widget_group_index] is not None:
                        ending_id = wx.Window.GetId(self.ending_holder_list[widget_group_index])
                        self.widget_dict[ending_id] = (widget_group_index, ENDING_HOLDER)
                    if self.step_holder_list[widget_group_index] is not None:
                        # noinspection PyUnresolvedReferences
                        step_id = wx.Window.GetId(self.step_holder_list[widget_group_index])
                        self.widget_dict[step_id] = (widget_group_index, STEP_HOLDER)
            
                for z in range(self.pace_num):
                    endofblock_flag = False
                    line_count += 1
                    if not self.pace_num == 0:
                        step_initial = 1.0
                        if self.empty_user_data_dict_flag:
                            (name, starting_selection_list, ending_selection_list, step_selection_list,
                             starting_hpace_list, starting_wpace_list,
                             ending_hpace_list, ending_wpace_list,
                             step_selection_min_list, step_selection_max_list, step_selection_inc_list,
                             default_selection_list, default_hpace_list, default_wpace_list,
                             default_selection_min_list, default_selection_max_list,
                             default_selection_inc_list) = self.pace_list[z]
            
                            if len(starting_selection_list) - 1 < x:
                                starting_selection = starting_selection_list[len(starting_selection_list) - 1]
                            else:
                                starting_selection = starting_selection_list[x]
            
                            if len(ending_selection_list) - 1 < x:
                                ending_selection = ending_selection_list[len(ending_selection_list) - 1]
                            else:
                                ending_selection = ending_selection_list[x]
            
                            # if len(step_selection_list) - 1 < x:
                            #     step_selection = step_selection_list[len(step_selection_list) - 1]
                            # else:
                            #     step_selection = step_selection_list[x]
            
                            if len(step_selection_min_list) - 1 < x:
                                step_selection_min = step_selection_min_list[len(step_selection_min_list) - 1]
                            else:
                                step_selection_min = step_selection_min_list[x]
            
                            if len(step_selection_max_list) - 1 < x:
                                step_selection_max = step_selection_max_list[len(step_selection_max_list) - 1]
                            else:
                                step_selection_max = step_selection_max_list[x]
            
                            if len(step_selection_inc_list) - 1 < x:
                                step_selection_inc = step_selection_inc_list[len(step_selection_inc_list) - 1]
                            else:
                                step_selection_inc = step_selection_inc_list[x]
            
                            if len(starting_hpace_list) - 1 < x:
                                starting_hpace = starting_hpace_list[len(starting_hpace_list) - 1]
                            else:
                                starting_hpace = starting_hpace_list[x]
            
                            if len(ending_hpace_list) - 1 < x:
                                ending_hpace = ending_hpace_list[len(ending_hpace_list) - 1]
                            else:
                                ending_hpace = ending_hpace_list[x]
            
                            if len(starting_wpace_list) - 1 < x:
                                starting_wpace = starting_wpace_list[len(starting_wpace_list) - 1]
                            else:
                                starting_wpace = starting_wpace_list[x]
            
                            if len(ending_wpace_list) - 1 < x:
                                ending_wpace = ending_wpace_list[len(ending_wpace_list) - 1]
                            else:
                                ending_wpace = ending_wpace_list[x]
            
                            if len(default_selection_list) - 1 < x:
                                default_selection = default_selection_list[len(default_selection_list) - 1]
                            else:
                                default_selection = default_selection_list[x]
            
                            # if len(default_hpace_list) - 1 < x:
                            #     default_hpace = default_hpace_list[len(default_hpace_list) - 1]
                            # else:
                            #     default_hpace = default_hpace_list[x]
                            #
                            # if len(default_wpace_list) - 1 < x:
                            #     default_wpace = default_wpace_list[len(default_wpace_list) - 1]
                            # else:
                            #     default_wpace = default_wpace_list[x]
            
                            if len(default_selection_min_list) - 1 < x:
                                default_selection_min = default_selection_min_list[len(default_selection_min_list) - 1]
                            else:
                                default_selection_min = default_selection_min_list[x]
            
                            if len(default_selection_max_list) - 1 < x:
                                default_selection_max = default_selection_max_list[len(default_selection_max_list) - 1]
                            else:
                                default_selection_max = default_selection_max_list[x]
            
                            if len(default_selection_inc_list) - 1 < x:
                                default_selection_inc = default_selection_inc_list[len(default_selection_inc_list) - 1]
                            else:
                                default_selection_inc = default_selection_inc_list[x]
        
                        else:
                            user_data_list_index += 1
                            (name, starting_selection, ending_selection, step_initial,
                             starting_selection, starting_hpace, starting_wpace,
                             ending_selection, ending_hpace, ending_wpace,
                             step_selection_min, step_selection_max, step_selection_inc,
                             default_selection, default_selection_min, default_selection_max,
                             default_selection_inc) = self.user_data_list[user_data_list_index]
        
                        widget_group_index += 1
                        self.starting_holder_list.append(wx.Choice(self.panel, wx.ID_ANY, size=(80, 20),
                                                                   choices=_choices,
                                                                   style=wx.SP_ARROW_KEYS | wx.SP_WRAP))
                        # noinspection PyUnresolvedReferences
                        self.ending_holder_list.append(wx.Choice(self.panel, wx.ID_ANY, size=(80, 20), choices=_choices,
                                                                 style=wx.SP_ARROW_KEYS | wx.SP_WRAP))
        
                        self.step_holder_list.append(wx.SpinCtrlDouble(self.panel, wx.ID_ANY, size=(60, 20),
                                                                       style=wx.SP_ARROW_KEYS | wx.SP_WRAP,
                                                                       min=step_selection_min, max=step_selection_max,
                                                                       initial=step_initial, inc=step_selection_inc))
                        self.step_holder_list[widget_group_index].Enable(False)

                        self.name_holder_list.append(wx.StaticText(self.panel, wx.ID_ANY, name, size=(100, 30)))
                        # noinspection PyUnresolvedReferences
                        style = wx.TE_CENTRE | wx.TE_READONLY | wx.NO_BORDER | wx.TE_RICH
                        self.ongoing_holder_list.append(wx.TextCtrl(self.panel, wx.ID_ANY, size=(120, 30),
                                                                    style=style))
                        # self.ongoing_holder_list[widget_group_index].SetDefaultStyle(wx.TextAttr(wx.Colour(55, 63, 65)))
                        self.ongoing_holder_list[widget_group_index].SetBackgroundColour(Default_Background_Color)
                        self.ongoing_holder_list[widget_group_index].SetForegroundColour(wx.Colour(55, 63, 65))
                        self.ongoing_holder_list[widget_group_index].SetFont(wx.Font(12, 70, 90, 92, False, "  "))
                        # self.ongoing_holder_list[widget_group_index].SetForegroundColour(wx.Colour(55, 63, 65))
        
                        self.starting_holder_list[widget_group_index].SetSelection(starting_selection)
                        self.ending_holder_list[widget_group_index].SetSelection(ending_selection)

                        static_text_index += 1
                        self.static_text_to_list.append(wx.StaticText(self.panel, wx.ID_ANY, "TO", size=(20, 20)))
                        self.static_text_to_list[static_text_index].SetForegroundColour(wx.Colour(120, 120, 123))
                        self.static_text_step_list.append(wx.StaticText(self.panel, wx.ID_ANY, "STEP", size=(30, 20)))
                        self.static_text_step_list[static_text_index].SetForegroundColour(wx.Colour(120, 120, 123))
 
                        # if z == self.pace_num - 1:
                        if line_count >= total_line_per_block:
                            endofblock_flag = True
                            line_count = 0
                            if x == self.layer_num - 1:
                                endoflayer_flag = True
                                if self.title == u"Sigmoid":
                                    endoflayer_flag = False
                                elif self.title == u"Learning":
                                    endoflayer_flag = False
                                elif self.title == u"Cross-entropy":
                                    endoflayer_flag = False
        
                        self.data_list.append((CHOICE_CTRL, _label, name, None,
                                               starting_selection, ending_selection, step_initial,
                                               starting_selection, starting_hpace, starting_wpace,
                                               ending_selection, ending_hpace, ending_wpace,
                                               step_selection_min, step_selection_max, step_selection_inc,
                                               default_selection, default_selection_min, default_selection_max,
                                               default_selection_inc,
                                               endofblock_flag, endoflayer_flag))
        
                        starting_id = self.starting_holder_list[widget_group_index].GetId()
                        self.widget_dict[starting_id] = (widget_group_index, STARTING_CHOICE_HOLDER)
                        ending_id = self.ending_holder_list[widget_group_index].GetId()
                        self.widget_dict[ending_id] = (widget_group_index, ENDING_CHOICE_HOLDER)
                        step_id = self.step_holder_list[widget_group_index].GetId()
                        self.widget_dict[step_id] = (widget_group_index, STEP_CHOICE_HOLDER)

        # noinspection PyUnresolvedReferences
        self.runButton = wx.Button(self.panel, wx.ID_ANY, u"RUN", wx.DefaultPosition,
                                   wx.Size(90, 30), 0)
        # noinspection PyUnresolvedReferences
        self.runButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        # noinspection PyUnresolvedReferences
        self.editconfButton = wx.Button(self.panel, wx.ID_ANY, u"LIMITS EDIT", wx.DefaultPosition,
                                        wx.Size(90, 30), 0)
        # noinspection PyUnresolvedReferences
        self.editconfButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        self.editorderButton = wx.Button(self.panel, wx.ID_ANY, u"ORDER EDIT", wx.DefaultPosition,
                                         wx.Size(90, 30), 0)
        self.editorderButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        # noinspection PyUnresolvedReferences
        self.editcolorButton = wx.Button(self.panel, wx.ID_ANY, u"COLOR EDIT", wx.DefaultPosition,
                                         wx.Size(90, 30), 0)
        self.editcolorButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        # noinspection PyUnresolvedReferences
        self.restoreButton = wx.Button(self.panel, wx.ID_ANY, u"RESTORE", wx.DefaultPosition,
                                       wx.Size(90, 30), 0)
        self.restoreButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        # noinspection PyUnresolvedReferences
        self.resetButton = wx.Button(self.panel, wx.ID_ANY, u"RESET", wx.DefaultPosition,
                                     wx.Size(90, 30), 0)
        self.resetButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        self.exitButton = wx.Button(self.panel, wx.ID_ANY, u"EXIT", wx.DefaultPosition,
                                    wx.Size(90, 30), 0)
        self.exitButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        # noinspection PyUnresolvedReferences
        self.saveButton = wx.Button(self.panel, wx.ID_ANY, u"SAVE", wx.DefaultPosition,
                                    wx.Size(90, 30), 0)
        self.saveButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        self.plotButton = wx.Button(self.panel, wx.ID_ANY, u"PLOT", wx.DefaultPosition,
                                    wx.Size(90, 30), 0)
        self.plotButton.SetBackgroundColour(wx.Colour(185, 193, 195))

        # noinspection PyUnresolvedReferences
        self.message = wx.TextCtrl(self.panel, wx.ID_ANY, "", size=(600, 40),
                                   style=wx.NO_BORDER | wx.CENTRE)
        self.message.SetBackgroundColour(Default_Background_Color)
        # self.message.SetForegroundColour(wx.Colour(55, 63, 65))
        # self.message.SetDefaultStyle(wx.TextAttr(wx.Colour(55, 63, 65)))
        self.message.SetFont(wx.Font(16, 70, 90, 92, False, "  "))
        # self.message.SetForegroundColour(wx.Colour(55, 63, 65))
                    
        self.init_ui()
    
        self.binding()
        
        self.determine_num_epoch_n_display()

    def init_ui(self):

        line_sbsizer_list = []
        line_sbsizer_index = -1
        block_sbsizer_list = []
        block_sbsizer_index = -1
        layer_sbsizer_list = []
        layer_sbsizer_index = -1
        add_to_layer_list = []
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        sizeofblock = 0
        # previous_line_size_x = 0
        
        widget_index = -1
        static_text_index = -1
        previous_endofblock_flag = True
        
        for x in range(len(self.data_list)):
            (cntrl_type, title, name, title_holder,
             starting_itital, ending_initial, step_initial,
             starting_min, starting_max, starting_inc,
             ending_min, ending_max, ending_inc,
             step_min, step_max, step_inc,
             default_var, default_min, default_max, default_inc,
             endofblock_flag, endoflayer_flag) = self.data_list[x]
            
            if previous_endofblock_flag:
    
                # noinspection PyUnresolvedReferences
                block_sbsizer_list.append(wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, title), wx.VERTICAL))
                previous_endofblock_flag = False
                block_sbsizer_index += 1
                
            if cntrl_type is DISPLAY_CTRL:
                # noinspection PyUnresolvedReferences
                line_sbsizer_list.append(wx.BoxSizer(wx.HORIZONTAL))
                line_sbsizer_index += 1
                widget_index += 1

                line_sbsizer_list[line_sbsizer_index].Add(self.name_holder_list[widget_index],
                                                          0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                line_sbsizer_list[line_sbsizer_index].Add(self.starting_holder_list[widget_index],
                                                          0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                line_sbsizer_list[line_sbsizer_index].Add(self.ongoing_holder_list[widget_index],
                                                          0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)

                block_sbsizer_list[block_sbsizer_index].Add(line_sbsizer_list[line_sbsizer_index],
                                                            0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                
            if cntrl_type is SPIN_CTRL_DOUBLE or cntrl_type is CHOICE_CTRL:
                line_sbsizer_list.append(wx.BoxSizer(wx.HORIZONTAL))
                line_sbsizer_index += 1
                widget_index += 1
                static_text_index += 1
                
                line_sbsizer_list[line_sbsizer_index].Add(self.name_holder_list[widget_index],
                                                          0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                line_sbsizer_list[line_sbsizer_index].Add(self.starting_holder_list[widget_index],
                                                          0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                if self.ending_holder_list[widget_index] is not None:
                    line_sbsizer_list[line_sbsizer_index].Add(self.static_text_to_list[static_text_index],
                                                              0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                    # noinspection PyUnresolvedReferences
                    line_sbsizer_list[line_sbsizer_index].Add(self.ending_holder_list[widget_index],
                                                              0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                if self.step_holder_list[widget_index] is not None:
                    line_sbsizer_list[line_sbsizer_index].Add(self.static_text_step_list[static_text_index],
                                                              0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                    line_sbsizer_list[line_sbsizer_index].Add(self.step_holder_list[widget_index],
                                                              0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                if self.ongoing_holder_list[widget_index] is not None:
                    line_sbsizer_list[line_sbsizer_index].Add(self.ongoing_holder_list[widget_index],
                                                              0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)

                # noinspection PyUnresolvedReferences
                block_sbsizer_list[block_sbsizer_index].Add(line_sbsizer_list[line_sbsizer_index],
                                                            0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
                
                # linebox_size = line_sbsizer_list[line_sbsizer_index].GetSize()
                # line_size_x = linebox_size.GetWidth()
                # sizeofblock = max(previous_line_size_x, line_size_x)
                # previous_line_size_x = line_size_x
                
                # if name == "CHANNELS":
                #     self.ongoing_holder_list[widget_index].Enable(False)
                #     self.starting_holder_list[widget_index].Enable(False)
                #     self.ending_holder_list[widget_index].Enable(False)
                #     self.step_holder_list[widget_index].Enable(False)

            if endofblock_flag:
                previous_endofblock_flag = True
                test_for_digit = title[len(title) - 1:]
                striped_title = title
                while test_for_digit.isdigit():
                    striped_title = striped_title[: len(striped_title) - 1]
                    test_for_digit = striped_title[- 1:]

                if striped_title == "Convolution":
                    sizeofblock += 500
                elif striped_title == "Multi-neuron":
                    sizeofblock += 450
                elif striped_title == "Sigmoid":
                    sizeofblock += 450
                elif striped_title == "Learning":
                    sizeofblock += 450
                elif striped_title == "Cross-entropy":
                    sizeofblock += 450
                elif striped_title == "Number of Steps":
                    sizeofblock += 240
    
                add_to_layer_list.append(block_sbsizer_list[block_sbsizer_index])

            if endoflayer_flag:
                add_space = 0
                if sizeofblock < self.display_x:
                    add_space = int((self.display_x - sizeofblock) / 2)
                layer_sbsizer_list.append(wx.BoxSizer(wx.HORIZONTAL))
                layer_sbsizer_index += 1
                layer_sbsizer_list[layer_sbsizer_index].AddSpacer(add_space)
                sizeofblock = 0
                for add_block_item in add_to_layer_list:
                    layer_sbsizer_list[layer_sbsizer_index].Add(add_block_item)
                add_to_layer_list.clear()
                main_sizer.Add(layer_sbsizer_list[layer_sbsizer_index])

        self.update_ordering_list()
        self.repaint_ordering()

        self.ongoing_holder_list[len(self.ongoing_holder_list) - 5].SetValue(repr(1))

        # noinspection PyUnresolvedReferences
        bn_sbsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        bn_sbsizer2 = wx.BoxSizer(wx.VERTICAL)
        bn_sbsizer = wx.BoxSizer(wx.HORIZONTAL)
        bn_sbsizer1.Add(self.runButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer1.AddSpacer(1)
        bn_sbsizer1.Add(self.editconfButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer1.AddSpacer(1)
        # noinspection PyUnresolvedReferences
        bn_sbsizer1.Add(self.editorderButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer1.AddSpacer(1)
        # noinspection PyUnresolvedReferences
        bn_sbsizer1.Add(self.editcolorButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer1.AddSpacer(1)
        bn_sbsizer1.Add(self.restoreButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer1.AddSpacer(1)
        bn_sbsizer1.Add(self.plotButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer1.AddSpacer(1)
        # noinspection PyUnresolvedReferences
        bn_sbsizer1.Add(self.resetButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer1.AddSpacer(1)
        bn_sbsizer1.Add(self.saveButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer1.AddSpacer(1)
        bn_sbsizer1.Add(self.exitButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        bn_sbsizer2.AddSpacer(1)
        bn_sbsizer2.Add(bn_sbsizer1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)

        bn_size_x, _ = bn_sbsizer2.GetMinSize()
        bn_spacer = 0
        if bn_size_x < self.display_x:
            bn_spacer = int((self.display_x - bn_size_x) / 2)
        bn_sbsizer.AddSpacer(bn_spacer)
        bn_sbsizer.Add(bn_sbsizer2)

        # noinspection PyUnresolvedReferences
        message_sbsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        # noinspection PyUnresolvedReferences
        message_sbsizer = wx.BoxSizer(wx.HORIZONTAL)
        message_sbsizer1.Add(self.message, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)

        message_size_x, _ = message_sbsizer1.GetMinSize()
        message_spacer = 0
        if message_size_x < self.display_x:
            message_spacer = int((self.display_x - message_size_x) / 2)
        message_sbsizer.AddSpacer(message_spacer)
        message_sbsizer.Add(message_sbsizer1)

        main_sizer.Add(bn_sbsizer, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)
        # noinspection PyUnresolvedReferences
        main_sizer.Add(message_sbsizer, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 1)

        # size = main_sizer.GetMinSize()
        # self.SetMinSize(size)

        self.panel.SetSizerAndFit(main_sizer)
        
        self.SetSizer(main_sizer)
        self.SetupScrolling()
        
        self.Fit()
        # self.Center()

        self.Show()
        
    def binding(self):
        # Binding all the buttons
        # title = " "
        # data_index = -1
        for data_index in range(len(self.data_list)):
            (cntrl_type, _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _, _,
             _, _) = self.data_list[data_index]

            if cntrl_type == SPIN_CTRL_DOUBLE:
                # data_index += 1
                self.starting_holder_list[data_index].Bind(wx.EVT_CHAR, self.test_for_key_being_pressed)
                self.starting_holder_list[data_index].Bind(wx.EVT_SPINCTRLDOUBLE, self.on_field_changed)
                
                if self.ending_holder_list[data_index] is not None:
                    self.ending_holder_list[data_index].Bind(wx.EVT_CHAR, self.test_for_key_being_pressed)
                    self.ending_holder_list[data_index].Bind(wx.EVT_SPINCTRLDOUBLE, self.on_field_changed)
                    
                if self.step_holder_list[data_index] is not None:
                    # noinspection PyUnresolvedReferences
                    self.step_holder_list[data_index].Bind(wx.EVT_CHAR, self.test_for_key_being_pressed)
                    self.step_holder_list[data_index].Bind(wx.EVT_SPINCTRLDOUBLE, self.on_field_changed)
                    
            if cntrl_type == CHOICE_CTRL:
                # data_index += 1
                self.starting_holder_list[data_index].Bind(wx.EVT_CHOICE, self.on_choice_field_changed)
                self.ending_holder_list[data_index].Bind(wx.EVT_CHOICE, self.on_choice_field_changed)
                self.step_holder_list[data_index].Bind(wx.EVT_SPINCTRLDOUBLE, self.on_choice_field_changed)
                self.step_holder_list[data_index].Bind(wx.EVT_CHAR, self.test_for_key_being_pressed)
                
        self.parent.Bind(wx.adv.EVT_TASKBAR_LEFT_UP, self.on_taskbar_left_click)
        self.editconfButton.Bind(wx.EVT_LEFT_DOWN, self.on_edit_limits)
        self.editorderButton.Bind(wx.EVT_LEFT_DOWN, self.on_edit_orders)
        self.editcolorButton.Bind(wx.EVT_LEFT_DOWN, self.on_edit_colors)
        # noinspection PyUnresolvedReferences
        self.runButton.Bind(wx.EVT_LEFT_DOWN, self.on_run)
        # noinspection PyUnresolvedReferences
        self.resetButton.Bind(wx.EVT_LEFT_DOWN, self.on_hard_reset)
        self.restoreButton.Bind(wx.EVT_LEFT_DOWN, self.on_soft_reset)
        self.saveButton.Bind(wx.EVT_LEFT_DOWN, self.on_save)
        self.plotButton.Bind(wx.EVT_LEFT_DOWN, self.on_plot_data)

        self.exitButton.Bind(wx.EVT_LEFT_DOWN, self.on_exit)
   
    # noinspection PyUnusedLocal
    def on_run(self, event):
        self.epoch_time_established_flag = False
        self.epoch_remaining = self.num_epoch - 1
        self.average_epoch_time = None
        self.runButton.Hide()
        self.editconfButton.Hide()
        self.editorderButton.Hide()
        self.editcolorButton.Hide()
        self.restoreButton.Hide()
        self.resetButton.Hide()
        self.saveButton.Hide()
        self.exitButton.Hide()
        self.plotButton.Hide()
        self.ordering_value_list = [None for x in range(len(self.ordering_list))]
        self.set_all_parameters()
        self.message.SetValue("Testing runs are conducting with the above parameters...")
        # self.message.Refresh()
        # wx.EventBlocker(self)
        # time.sleep(1)
        # self.parent.Iconize()
    
        # if SHRINK:
        #     t_shrink_train_images = t_shrink_test_images = None
        #     try:
        #         t_shrink_train_images = Thread(target=self.shrink_images)
        #         t_shrink_train_images.start()
        #     except:
        #         wx.MessageDialog(self, "Unable to start new thread to start shrink train images", "THREADING ERROR!",
        #                          wx.ICON_EXCLAMATION | wx.OK).ShowModal()
        #         exit(1)
        #
        #     try:
        #         t_shrink_test_images = Thread(target=self.shrink_images, kwargs={"mode": "test"})
        #         t_shrink_test_images.start()
        #     except:
        #         wx.MessageDialog(self, "Unable to start new thread to start shrink test images", "THREADING ERROR!",
        #                          wx.ICON_EXCLAMATION | wx.OK).ShowModal()
        #         exit(1)
        #
        #     t_shrink_train_images.join()
        #     t_shrink_test_images.join()
        
        self.determine_num_epoch_n_display()

        if len(self.ordering_list) == 0:
            t_train = None
            self.epoch_number += 1
    
            try:
                t_train = Thread(target=self.train)
                t_train.start()
            except:
                # Something went incorrectly with the thread throwing. Send user a message and exit
                self.epoch_number -= 1
                wx.MessageDialog(self, "Unable to start new thread to start neuronetwork", "THREADING ERROR!",
                                 wx.ICON_EXCLAMATION | wx.OK).ShowModal()
                exit(1)
            t_train.join()
        else:
            self.set_ordered_hyperparameters_n_train()
    
        time_now_str = time.strftime('%m_%d_%Y(%H %M %S)')
        accuracy_filename = "accuracy_data" + time_now_str
        np_file = os.path.join(WORKING_DIR, accuracy_filename)
        try:
            np.save(np_file, np.array(self.np_stats))
    
        except:
            wx.MessageDialog(self, "Unable to write %s" % accuracy_filename, "FILE WRITING ERROR!",
                             wx.ICON_EXCLAMATION | wx.OK).ShowModal()
            exit(1)
        # Try to open data.txt.
        stat_name_filename = "stat_name_list" + time_now_str + ".txt"
        full_stat_name_file = os.path.join(WORKING_DIR, stat_name_filename)
        try:
            stat_name_file = open(full_stat_name_file, 'w')
            stat_name_file.write(str(self.stat_name_list))
            # Close the data.txt
            stat_name_file.close()
        except:
            # noinspection PyUnresolvedReferences
            wx.MessageDialog(self, "Unable to write %s" % stat_name_filename, "FILE WRITING ERROR!",
                             wx.ICON_EXCLAMATION | wx.OK).ShowModal()
            exit(1)

        steps_per_report_filename = "steps_per_report" + time_now_str + ".txt"
        full_steps_per_report_file = os.path.join(WORKING_DIR, steps_per_report_filename)
        # noinspection PyBroadException
        try:
            steps_per_report_file = open(full_steps_per_report_file, 'w')
            steps_per_report_file.write(str(self.num_steps_per_report))
            # Close the data.txt
            steps_per_report_file.close()
        except:
            wx.MessageDialog(self, "Unable to write %s" % steps_per_report_filename, "FILE WRITING ERROR!",
                             wx.ICON_EXCLAMATION | wx.OK).ShowModal()
            exit(1)
    
        # plot_data = PlotResult(self.panel)

    # noinspection PyUnusedLocal,PyUnusedLocal
    def on_plot_data(self, event):
        PlotResult(self)

    # noinspection PyUnusedLocal
    def on_save(self, event):
        self.save()
        
    def save(self):
        # working_dir = u"c:\\Users\\Chester\\PythonSourceCodes\\HyperparameterSearcher"
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
            
        if not self.empty_user_data_dict_flag:
            self.user_data_list.clear()
            # index = -1
            for index in range(len(self.data_list)):
                (cntrl_type, _, name, _,
                 starting_var, ending_var, step_initial,
                 starting_min, starting_max, starting_inc,
                 ending_min, ending_max, ending_inc,
                 step_min, step_max, step_inc,
                 default_var, default_min, default_max, default_inc,
                 _, _) = self.data_list[index]
    
                if cntrl_type == SPIN_CTRL_DOUBLE:
                    # index += 1
                    starting_var = float(self.starting_holder_list[index].GetValue())
                    if ending_var is not None:
                        ending_var = float(self.ending_holder_list[index].GetValue())
                    if step_initial is not None:
                        step_initial = float(self.step_holder_list[index].GetValue())
                    
                    self.user_data_list.append((name, starting_var, ending_var, step_initial,
                                               starting_min, starting_max, starting_inc,
                                               ending_min, ending_max, ending_inc,
                                               step_min, step_max, step_inc,
                                               default_var, default_min, default_max, default_inc))
                elif cntrl_type == CHOICE_CTRL:
                    # index += 1
                    starting_selection = float(self.starting_holder_list[index].GetSelection())
                    ending_selection = float(self.ending_holder_list[index].GetSelection())
                    step_initial = float(self.step_holder_list[index].GetValue())
                    self.user_data_list.append((name, starting_selection, ending_selection, step_initial,
                                               starting_min, starting_max, starting_inc,
                                               ending_min, ending_max, ending_inc,
                                               step_min, step_max, step_inc,
                                               default_var, default_min, default_max, default_inc))
           
            _filename = os.path.join(WORKING_DIR, "user_reset_data.txt")
            # Try to open data.txt. Create a new one if not existed
            _user_data_file = open(_filename, 'w+')
            json.dump(self.user_data_list, _user_data_file)
            
            # Close the data.txt
            _user_data_file.close()
            
        if not self.color_data_unchanged_flag:
            color_tuple = (self.primarycolor.GetRGB(), self.secondarycolor.GetRGB(), self.tertiarycolor.GetRGB(),
                           self.quaternarycolor.GetRGB(), self.highercolor.GetRGB())
           
            _color_data_filename = os.path.join(WORKING_DIR, "color_data.txt")
            # Try to open data.txt. Create a new one if not existed
            _color_file = open(_color_data_filename, 'w+')
            json.dump(color_tuple, _color_file)

            # Close the data.txt
            _color_file.close()
            
        if not self.ordering_hierarchy_unchanged_flag:
            hierarchy_dict = {}
            _hierarchy_data_filename = os.path.join(WORKING_DIR, "hierarchy_data.txt")
            for x in range(len(self.ordering_list)):
                hierarchy_dict[x] = self.ordering_list[x]
            # Try to open data.txt. Create a new one if not existed
            _hierarchy_file = open(_hierarchy_data_filename, 'w+')
            json.dump(hierarchy_dict, _hierarchy_file)
    
            # Close the data.txt
            _hierarchy_file.close()
        
        wx.MessageDialog(self.panel, "User settings had been saved into\n\n %s \n\n" % WORKING_DIR,
                         "USER SETTINGS SAVED!", wx.ICON_INFORMATION | wx.OK).ShowModal()

        self.save_flag = False

    # noinspection PyUnusedLocal
    def on_exit(self, event):
        message = "User setting(s) had been changed\n\n Do you want to save setting(s)? \n\n"
        if self.save_flag:
            # noinspection PyUnresolvedReferences
            if wx.MessageDialog(self.panel, message, "SAVE SETTINGS",
                                wx.ICON_QUESTION | wx.YES_NO | wx.YES_DEFAULT).ShowModal() == wx.ID_YES:
                self.save()
            
        self.panel.Destroy()
        
        self.Destroy()
            
        exit(0)

    # noinspection PyUnusedLocal
    def on_taskbar_left_click(self, event):
        self.parent.Maximize()

    # noinspection PyUnusedLocal
    def on_hard_reset(self, event):
        for data_list_index in range(len(self.data_list)):
            (cntrl_type, _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             default_var, default_min, default_max, default_inc,
             _, _) = self.data_list[data_list_index]

            if cntrl_type == SPIN_CTRL_DOUBLE:
                self.starting_holder_list[data_list_index].SetValue(default_var)
                self.starting_holder_list[data_list_index].SetRange(default_min, default_max)
                self.starting_holder_list[data_list_index].SetIncrement(default_inc)
                if self.ending_holder_list[data_list_index] is not None:
                    self.ending_holder_list[data_list_index].SetValue(default_var)
                    self.ending_holder_list[data_list_index].SetRange(default_min, default_max)
                    self.ending_holder_list[data_list_index].SetIncrement(default_inc)
                if self.step_holder_list[data_list_index] is not None:
                    self.step_holder_list[data_list_index].SetValue(default_inc)
                    self.step_holder_list[data_list_index].SetRange(default_min, default_max)
                    self.step_holder_list[data_list_index].SetIncrement(default_inc)
                    self.step_holder_list[data_list_index].Enable(False)
            elif cntrl_type == CHOICE_CTRL:
                self.starting_holder_list[data_list_index].SetSelection(default_var)
                self.ending_holder_list[data_list_index].SetSelection(default_var)
                if self.step_holder_list[data_list_index] is not None:
                    self.step_holder_list[data_list_index].SetValue(default_inc)
                    self.step_holder_list[data_list_index].SetRange(default_min, default_max)
                    self.step_holder_list[data_list_index].SetIncrement(default_inc)
                    self.step_holder_list[data_list_index].Enable(False)

        self.saved_ordering_list = self.ordering_list[:]
        self.ordering_list.clear()
        self.repaint_ordering()
        
        self.determine_num_epoch_n_display()

        self.save_flag = True
        self.empty_user_data_dict_flag = False

    # noinspection PyUnusedLocal
    def on_soft_reset(self, event):
    
        for data_list_index in range(len(self.data_list)):
            
            (cntrl_type, _, _, _,
             starting_var, ending_var, step_initial,
             starting_min, starting_max, starting_inc,
             ending_min, ending_max, ending_inc,
             step_min, step_max, step_inc,
             _, _, _, _,
             _, _) = self.data_list[data_list_index]
            
            if cntrl_type == SPIN_CTRL_DOUBLE:
                self.starting_holder_list[data_list_index].SetValue(starting_var)
                self.starting_holder_list[data_list_index].SetRange(starting_min, starting_max)
                self.starting_holder_list[data_list_index].SetIncrement(starting_inc)
                if self.ending_holder_list[data_list_index] is not None:
                    self.ending_holder_list[data_list_index].SetValue(ending_var)
                    self.ending_holder_list[data_list_index].SetRange(ending_min, ending_max)
                    self.ending_holder_list[data_list_index].SetIncrement(ending_inc)
                if self.step_holder_list[data_list_index] is not None:
                    self.step_holder_list[data_list_index].SetValue(step_initial)
                    self.step_holder_list[data_list_index].SetRange(step_min, step_max)
                    self.step_holder_list[data_list_index].SetIncrement(step_inc)
            elif cntrl_type == CHOICE_CTRL:
                self.starting_holder_list[data_list_index].SetSelection(starting_var)
                self.ending_holder_list[data_list_index].SetSelection(ending_var)
                if self.step_holder_list[data_list_index] is not None:
                    self.step_holder_list[data_list_index].SetValue(step_initial)
                    self.step_holder_list[data_list_index].SetRange(step_min, step_max)
                    self.step_holder_list[data_list_index].SetIncrement(step_inc)
        
        self.ordering_list = self.saved_ordering_list[:]
        self.update_ordering_list()
        self.repaint_ordering()
        self.saved_ordering_list.clear()
        
        self.determine_num_epoch_n_display()
        
        self.save_flag = True
        self.empty_user_data_dict_flag = False
        
    def update_ordering_list(self):
        for data_list_index in range(len(self.data_list)):
            (cntrl_type, _, _, _,
             starting_var, ending_var, step_initial,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _, _,
             _, _) = self.data_list[data_list_index]
            
            if ending_var is not None:
                if not starting_var == ending_var:
                    if self.ordering_list is not None and not len(self.ordering_list) == 0:
                        already_in_ordering_list_flag = False
                        for x in range(len(self.ordering_list)):
                            if self.ordering_list[x] == data_list_index:
                                already_in_ordering_list_flag = True
                                break
                        if not already_in_ordering_list_flag:
                            self.ordering_list.append(data_list_index)
                            self.empty_ordering_flag = False
                    else:
                        self.ordering_list.append(data_list_index)
                        self.empty_ordering_flag = False
        
    # noinspection PyUnusedLocal
    def on_edit_orders(self, event):
        rearrange_items = []
        rearrange_order = []
        # order = []
        for x in range(len(self.ordering_list)):
            data_index = self.ordering_list[x]
            (_, title, name, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _, _,
             _, _) = self.data_list[data_index]
            title_n_name = title + " " + name
            rearrange_items.append(title_n_name)
            rearrange_order.append(x)
        # noinspection PyUnresolvedReferences
        dlg = wx.RearrangeDialog(self.panel, "Select to move \n\nuncheck to delete",
                                 "EDIT ORDERING", rearrange_order, rearrange_items)
        if dlg.ShowModal() == wx.ID_OK:
            self.save_flag = True
            self.ordering_hierarchy_unchanged_flag = False
            order_list = dlg.GetOrder()
            temp_ordering_list = self.ordering_list[:]
            self.ordering_list.clear()
            hierarchy = 0
            for x in range(len(order_list)):
                data_index = temp_ordering_list[x]
                (cntrl_type, title, name, title_holder,
                 starting_var, ending_var, step_initial,
                 starting_min, starting_max, starting_inc,
                 ending_min, ending_max, ending_inc,
                 step_min, step_max, step_inc,
                 default_init, default_min, default_max, default_inc,
                 endofblock_flag, endoflayer_flag) = self.data_list[data_index]
                if order_list[x] > -1:
                    self.ordering_list.append(temp_ordering_list[x])
                    hierarchy += 1
                    if hierarchy == 1:
                        self.name_holder_list[data_index].SetBackgroundColour(self.primarycolor)
                        self.name_holder_list[data_index].Refresh()
                    elif hierarchy == 2:
                        self.name_holder_list[data_index].SetBackgroundColour(self.secondarycolor)
                        self.name_holder_list[data_index].Refresh()
                    elif hierarchy == 3:
                        self.name_holder_list[data_index].SetBackgroundColour(self.tertiarycolor)
                        self.name_holder_list[data_index].Refresh()
                    elif hierarchy == 4:
                        self.name_holder_list[data_index].SetBackgroundColour(self.quaternarycolor)
                        self.name_holder_list[data_index].Refresh()
                    elif hierarchy >= 5:
                        self.name_holder_list[data_index].SetBackgroundColour(self.highercolor)
                        self.name_holder_list[data_index].Refresh()
                    self.step_holder_list[data_index].Enable()
                else:
                    self.name_holder_list[data_index].SetBackgroundColour(Default_Background_Color)
                    self.name_holder_list[data_index].Refresh()
    
                    if cntrl_type == SPIN_CTRL_DOUBLE:
                        ending_var = self.starting_holder_list[data_index].GetValue()
                        self.ending_holder_list[data_index].SetValue(ending_var)
                    elif cntrl_type == CHOICE_CTRL:
                        ending_var = selection = self.starting_holder_list[data_index].GetSelection()
                        self.ending_holder_list[data_index].SetSelection(selection)
                        
                    self.data_list.pop(data_index)

                    self.data_list.insert(data_index, (cntrl_type, title, name, title_holder,
                                                       starting_var, ending_var, step_initial,
                                                       starting_min, starting_max, starting_inc,
                                                       ending_min, ending_max, ending_inc,
                                                       step_min, step_max, step_inc,
                                                       default_init, default_min, default_max, default_inc,
                                                       endofblock_flag, endoflayer_flag))
                        
                    self.step_holder_list[data_index].SetValue(step_initial)
                    self.step_holder_list[data_index].Enable(False)
                    
        self.determine_num_epoch_n_display()

    # noinspection PyUnusedLocal
    def on_edit_colors(self, event):
        color = [self.primarycolor, self.secondarycolor, self.tertiarycolor, self.quaternarycolor, self.highercolor]
        returned_color = []

        dlg = ChangeColorDialog(self.panel, color)

        if dlg.ShowModal() == wx.ID_OK:
            returned_color = dlg.get_color()
            self.primarycolor = returned_color[0]
            self.secondarycolor = returned_color[1]
            self.tertiarycolor = returned_color[2]
            self.quaternarycolor = returned_color[3]
            self.highercolor = returned_color[4]
            
            self.save_flag = True
            self.color_data_unchanged_flag = False
            
            self.repaint_ordering()

    # noinspection PyUnusedLocal
    def on_edit_limits(self, event):

        # Apparently the user had double clicked. Stop the timer for process possible single click
        # self.timer.Stop()
        
        widget = wx.Window.FindFocus()

        # Get the widget info responsible for calling this on_edit().
        # The "- 1" is needed since the widgetid is -1 over the window id holding the focus
        data_index, widget_index = self.widget_dict[wx.Window.GetId(widget) - 1]
        (cntrl_type, title, name, title_holder,
         starting_initial, ending_initial, step_initial,
         starting_min, starting_max, starting_inc,
         ending_min, ending_max, ending_inc,
         step_min, step_max, step_inc,
         default_init, default_min, default_max, default_inc,
         endofblock_flag, endoflayer_flag) = self.data_list[data_index]

        # Get the original minimum, maximum, and increment for this spincontroldouble
        if cntrl_type == SPIN_CTRL_DOUBLE:
            widget_initial = widget_min = widget_max = widget_inc = 0.0
            if widget_index == STARTING_HOLDER:
                widget_initial = self.starting_holder_list[data_index].GetValue()
                widget_min = self.starting_holder_list[data_index].GetMin()
                widget_max = self.starting_holder_list[data_index].GetMax()
                widget_inc = self.starting_holder_list[data_index].GetIncrement()
            elif widget_index == ENDING_HOLDER:
                widget_initial = self.ending_holder_list[data_index].GetValue()
                widget_min = self.ending_holder_list[data_index].GetMin()
                widget_max = self.ending_holder_list[data_index].GetMax()
                widget_inc = self.ending_holder_list[data_index].GetIncrement()
            elif widget_index == STEP_HOLDER:
                widget_initial = self.step_holder_list[data_index].GetValue()
                widget_min = self.step_holder_list[data_index].GetMin()
                widget_max = self.step_holder_list[data_index].GetMax()
                widget_inc = self.step_holder_list[data_index].GetIncrement()
                
            widget_title_name = title + " " + name
    
            # Show a dialog to allow the user to give the new label name
            dlg = ChangeUserSettingDialog(self.panel, widget_initial, widget_min,
                                          widget_max, widget_inc, widget_title_name)
    
            if dlg.ShowModal() == wx.ID_OK:
                self.save_flag = True
                self.empty_user_data_dict_flag = False
                # User indicated OK. Get the new values and place them into the widget itself
                if widget_index == STARTING_HOLDER:
                    starting_initial = dlg.get_init()
                    self.starting_holder_list[data_index].SetValue(starting_initial)
                    starting_min = dlg.get_min()
                    starting_max = dlg.get_max()
                    starting_inc = dlg.get_inc()
                    self.starting_holder_list[data_index].SetRange(starting_min, starting_max)
                    self.starting_holder_list[data_index].SetIncrement(starting_inc)
                elif widget_index == ENDING_HOLDER:
                    ending_initial = dlg.get_init()
                    self.ending_holder_list[data_index].SetValue(ending_initial)
                    ending_min = dlg.get_min()
                    ending_max = dlg.get_max()
                    ending_inc = dlg.get_inc()
                    self.ending_holder_list[data_index].SetRange(ending_min, ending_max)
                    self.ending_holder_list[data_index].SetIncrement(ending_inc)
                elif widget_index == STEP_HOLDER:
                    step_initial = dlg.get_init()
                    self.step_holder_list[data_index].SetValue(step_initial)
                    step_min = dlg.get_min()
                    step_max = dlg.get_max()
                    step_inc = dlg.get_inc()
                    self.step_holder_list[data_index].SetRange(step_min, step_max)
                    self.step_holder_list[data_index].SetIncrement(step_inc)
    
                self.data_list.pop(data_index)
    
                self.data_list.insert(data_index, (cntrl_type, title, name, title_holder,
                                                   starting_initial, ending_initial, step_initial,
                                                   starting_min, starting_max, starting_inc,
                                                   ending_min, ending_max, ending_inc,
                                                   step_min, step_max, step_inc,
                                                   default_init, default_min, default_max, default_inc,
                                                   endofblock_flag, endoflayer_flag))
                
        elif cntrl_type == CHOICE_CTRL:
            widget_initial = widget_min = widget_max = widget_inc = 0.0
            if widget_index == STARTING_HOLDER:
                return
            elif widget_index == ENDING_HOLDER:
                return
            elif widget_index == STEP_HOLDER:
                widget_initial = self.step_holder_list[data_index].GetValue()
                widget_min = self.step_holder_list[data_index].GetMin()
                widget_max = self.step_holder_list[data_index].GetMax()
                widget_inc = self.step_holder_list[data_index].GetIncrement()
    
            widget_title_name = title + " " + name
    
            # Show a dialog to allow the user to give the new label name
            dlg = ChangeUserSettingDialog(self.panel, widget_initial, widget_min, widget_max, widget_inc,
                                          widget_title_name)
    
            if dlg.ShowModal() == wx.ID_OK:
                self.save_flag = True
                self.empty_user_data_dict_flag = False
                # User indicated OK. Get the new values and place them into the widget itself
                step_initial = dlg.get_init()
                self.step_holder_list[data_index].SetValue(step_initial)
                step_min = dlg.get_min()
                step_max = dlg.get_max()
                step_inc = dlg.get_inc()
                self.step_holder_list[data_index].SetRange(step_min, step_max)
                self.step_holder_list[data_index].SetIncrement(step_inc)
        
                self.data_list.pop(data_index)
        
                self.data_list.insert(data_index, (cntrl_type, title, name, title_holder,
                                                   starting_initial, ending_initial, step_initial,
                                                   starting_min, starting_max, starting_inc,
                                                   ending_min, ending_max, ending_inc,
                                                   step_min, step_max, step_inc,
                                                   default_init, default_min, default_max, default_inc,
                                                   endofblock_flag, endoflayer_flag))
            
    def repaint_ordering(self):
        for x in range(len(self.name_holder_list)):
            self.name_holder_list[x].SetBackgroundColour(Default_Background_Color)
            self.name_holder_list[x].Refresh()
        if self.ordering_list is not None:
            self.empty_ordering_flag = False
            for hierarchy in range(len(self.ordering_list)):
                data_index = self.ordering_list[hierarchy]
                
                if hierarchy == 0:
                    self.name_holder_list[data_index].SetBackgroundColour(self.primarycolor)
                    self.name_holder_list[data_index].Refresh()
                elif hierarchy == 1:
                    self.name_holder_list[data_index].SetBackgroundColour(self.secondarycolor)
                    self.name_holder_list[data_index].Refresh()
                elif hierarchy == 2:
                    self.name_holder_list[data_index].SetBackgroundColour(self.tertiarycolor)
                    self.name_holder_list[data_index].Refresh()
                elif hierarchy == 3:
                    self.name_holder_list[data_index].SetBackgroundColour(self.quaternarycolor)
                    self.name_holder_list[data_index].Refresh()
                elif hierarchy >= 4:
                    self.name_holder_list[data_index].SetBackgroundColour(self.highercolor)
                    self.name_holder_list[data_index].Refresh()

                self.step_holder_list[data_index].Enable()

    @staticmethod
    def test_for_key_being_pressed(event):
        # Get the keycode pressed
        keycode = event.GetKeyCode()
        
        # Get rid of the key pressed if it is not 0 to 9
        if ASCII_0 <= keycode <= ASCII_9:
            event.Skip()
        # 8 is the backspace key
        elif keycode == ASCII_BACKSPACE:
            event.Skip()
        elif keycode == ASCII_ENTER:
            event.Skip()
        elif keycode == ASCII_DELETE:
            event.Skip()
        elif keycode == ASCII_INSERT:
            event.Skip()
        # test for the home, end, and arrow keys
        elif ASCII_HOME <= keycode <= ASCII_ARROW_DOWN:
            event.Skip()
        # test for pageup and pagedown keys
        elif ASCII_PAGEUP <= keycode <= ASCII_PAGEDOWN:
            event.Skip()
        elif keycode == ASCII_PERIOD:
            event.Skip()
        elif keycode == ASCII_MINUS:
            event.Skip()

        else:
            # Play a sound to indicate wrong button was pressed and eat up the key pressed by not giving event.Skip()
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
    
    @staticmethod
    def get_decimal_point(num_float):
        str_float = str(num_float)
        index = str_float.find('.')
        if index == -1:
            return 0
        else:
            return len(str_float) - index - 1
        
    @staticmethod
    def get_pace_value(choice_str):

        separator1 = choice_str.find(", ") + 2
        separator2 = choice_str.find(", ", separator1) + 2
        separator3 = choice_str.find(", ", separator2) + 2
        
        hpace = round(float(choice_str[separator1: separator2 - 2]), 0)
        wpace = round(float(choice_str[separator2: separator3 - 2]), 0)
        
        return hpace, wpace
    
    def determine_num_epoch_n_display(self):
        self.num_epoch = 1
        for item in self.ordering_list:
            data_index = item
            (cntrl_type, _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _, _,
             _, _) = self.data_list[data_index]
    
            starting_value = 0.0
            ending_value = 0.0
            step_value = 1.0
            
            if cntrl_type == SPIN_CTRL_DOUBLE:
                starting_value = float(self.starting_holder_list[data_index].GetValue())
                ending_value = float(self.ending_holder_list[data_index].GetValue())
                step_value = float(self.step_holder_list[data_index].GetValue())
            elif cntrl_type == CHOICE_CTRL:
                starting_value = float(self.starting_holder_list[data_index].GetSelection())
                ending_value = float(self.ending_holder_list[data_index].GetSelection())
                step_value = float(self.step_holder_list[data_index].GetValue())
            self.num_epoch *= math.ceil(abs(starting_value - ending_value) / step_value) + 1
            if self.num_epoch > 1:
                self.empty_ordering_flag = False
        
        self.ongoing_holder_list[len(self.ongoing_holder_list) - 5].SetValue(repr(self.num_epoch))
        self.ongoing_holder_list[len(self.ongoing_holder_list) - 5].Update()
        
        self.np_stats = np.zeros((self.num_epoch, NUMBER_OF_STATS,
                                  int(self.num_steps_per_epoch / self.num_steps_per_report)))

    # noinspection PyUnusedLocal
    def do_change_ordering_list(self, event):
        self.field_changed_timer.Stop()
        pop_index = -1
        for x in range(len(self.ordering_list)):
            if self.field_changed_index == self.ordering_list[x]:
                pop_index = x
                self.field_changed_index = -1
        if pop_index > -1:
            self.ordering_list.pop(pop_index)
        self.repaint_ordering()
        if len(self.ordering_list) == 0:
            self.empty_ordering_flag = True
        
    def on_choice_field_changed(self, event):
        self.save_flag = True
        self.empty_user_data_dict_flag = False
        self.color_data_unchanged_flag = False
        self.ordering_hierarchy_unchanged_flag = False
        widget_field_changed = event.GetEventObject()
        # widget_field_changed.SetFocus()

        # noinspection PyUnresolvedReferences
        data_index, widget_index = self.widget_dict[wx.Window.GetId(widget_field_changed)]

        (cntrl_type, title, name, title_holder,
         starting_selection, ending_selection, step_initial,
         starting_selection, starting_hpace, starting_wpace,
         ending_selection, ending_hpace, ending_wpace,
         step_selection_min, step_selection_max, step_selection_inc,
         default_selection, default_selection_min, default_selection_max, default_selection_inc,
         endofblock_flag, endoflayer_flag) = self.data_list[data_index]
 
        starting_string = self.starting_holder_list[data_index].GetString(self.starting_holder_list[data_index].GetSelection())
        ending_string = self.ending_holder_list[data_index].GetString(self.ending_holder_list[data_index].GetSelection())
        starting_selection = round(float(self.starting_holder_list[data_index].GetSelection()), 0)
        ending_selection = round(float(self.ending_holder_list[data_index].GetSelection()), 0)
        step_initial = float(self.step_holder_list[data_index].GetValue())
        
        if not starting_string == ending_string:
            self.field_changed_timer.Stop()
            append_flag = True
            # ordering_hierarchy = 0
            self.step_holder_list[data_index].Enable(True)
            if not len(self.ordering_list) == 0:
                for x in range(len(self.ordering_list)):
                    ordered_index = self.ordering_list[x]
                    if ordered_index == data_index:
                        self.ordering_list.pop(x)
                        self.ordering_list.insert(x, data_index)
                        # ordering_hierarchy = x
                        append_flag = False
                        break
            if append_flag:
                self.ordering_list.append(data_index)
                self.repaint_ordering()
        else:
            self.step_holder_list[data_index].Enable(False)
            self.field_changed_timer.Start(1000)
            self.field_changed_index = data_index
            # self.choice_field_timer_flag = True
            # pop_index = -1
            # for x in range(len(self.ordering_list)):
            #     if data_index == self.ordering_list[x]:
            #         pop_index = x
            # if pop_index > -1:
            #     self.ordering_list.pop(pop_index)
            # if len(self.ordering_list) == 0:
            #     self.empty_ordering_flag = True
        # self.repaint_ordering()

        self.data_list.pop(data_index)
        self.data_list.insert(data_index, (cntrl_type, title, name, title_holder,
                                           starting_selection, ending_selection, step_initial,
                                           starting_selection, starting_hpace, starting_wpace,
                                           ending_selection, ending_hpace, ending_wpace,
                                           step_selection_min, step_selection_max, step_selection_inc,
                                           default_selection, default_selection_min, default_selection_max,
                                           default_selection_inc,
                                           endofblock_flag, endoflayer_flag))
        self.determine_num_epoch_n_display()

    # noinspection PyUnusedLocal,PyUnusedLocal
    def on_field_changed(self, event):
        self.save_flag = True
        self.empty_user_data_dict_flag = False
        self.color_data_unchanged_flag = False
        self.ordering_hierarchy_unchanged_flag = False
        widget_field_changed = event.GetEventObject()
        widget_field_changed.SetFocus()

        (data_index, _) = self.widget_dict[wx.Window.GetId(widget_field_changed)]
        
        (cntrl_type, title, name, title_holder,
         _, _, _,
         starting_min, starting_max, starting_inc,
         ending_min, ending_max, ending_inc,
         step_min, step_max, step_inc,
         default_init, default_min, default_max, default_inc,
         endofblock_flag, endoflayer_flag) = self.data_list[data_index]
        
        starting_value = float(self.starting_holder_list[data_index].GetValue())
        if self.ending_holder_list[data_index] is not None:
            ending_value = float(self.ending_holder_list[data_index].GetValue())
        else:
            return
        if self.step_holder_list[data_index] is not None:
            step_value = float(self.step_holder_list[data_index].GetValue())
        else:
            return
        
        if not starting_value == ending_value:
            self.field_changed_timer.Stop()
            append_flag = True
            ordering_hierarchy = 0
            self.step_holder_list[data_index].Enable(True)
            if not len(self.ordering_list) == 0:
                for x in range(len(self.ordering_list)):
                    ordered_index = self.ordering_list[x]
                    if ordered_index == data_index:
                        self.ordering_list.pop(x)
                        self.ordering_list.insert(x, data_index)
                        # ordering_hierarchy = x
                        append_flag = False
                        break
            if append_flag:
                self.ordering_list.append(data_index)
                self.repaint_ordering()
        else:
            self.step_holder_list[data_index].Enable(False)
            self.field_changed_timer.Start(1000)
            self.field_changed_index = data_index
            # self.field_timer_flag = True
            # pop_index = -1
            # for x in range(len(self.ordering_list)):
            #     if data_index == self.ordering_list[x]:
            #         pop_index = x
            # if pop_index > -1:
            #     self.ordering_list.pop(pop_index)
            # if len(self.ordering_list) == 0:
            #     self.empty_ordering_flag = True
            # print(self.ordering_list)
        # self.repaint_ordering()

        if name == u"OUT CHANNELS":
            widget_id = wx.Window.GetId(widget_field_changed)
            
            if widget_id in self.saved_display_reset_dict:
                self.saved_display_reset_dict[widget_id].SetValue(repr(widget_field_changed.GetValue()))
                
        elif name == u"Number of Steps":
            self.num_steps_per_epoch = starting_value
                    
        self.data_list.pop(data_index)
        self.data_list.insert(data_index, (cntrl_type, title, name, title_holder,
                                           starting_value, ending_value, step_value,
                                           starting_min, starting_max, starting_inc,
                                           ending_min, ending_max, ending_inc,
                                           step_min, step_max, step_inc,
                                           default_init, default_min, default_max, default_inc,
                                           endofblock_flag, endoflayer_flag))
        
        self.determine_num_epoch_n_display()

    def select_parameter(self, cntrl_type, title, name, value, hpace=None, wpace=None, index=-1):
        title_str = title
        while True:
            title_index = title_str[-1]
            if title_index.isnumeric():
                title_str = title_str[: -1]
            else:
                break
        if title_str == self.nt_title:
            if cntrl_type == DISPLAY_CTRL:
                if index == -1:
                    if name == "CHANNELS":
                        self.nt_current_channels_list.append(value)
                else:
                    if name == "CHANNELS":
                        self.nt_current_channels_list[index] = value
                        
            if cntrl_type == SPIN_CTRL_DOUBLE:
                if index == -1:
                    if name == "WEIGHT":
                        self.nt_current_wt_list.append(value)
                    elif name == "BIASES":
                        self.nt_current_bias_list.append(value)
                    elif name == "ST. DEV.":
                        self.nt_current_stdev_list.append(value)
                    elif name == "OUT CHANNELS":
                        self.nt_current_outchannels_list.append(value)
                    elif name == "FILTER HEIGHT":
                        self.nt_current_filterheight_list.append(value)
                    elif name == "FILTER WIDTH":
                        self.nt_current_filterwidth_list.append(value)
                else:
                    if name == "WEIGHT":
                        self.nt_current_wt_list[index] = value
                    elif name == "BIASES":
                        self.nt_current_bias_list[index] = value
                    elif name == "ST. DEV.":
                        self.nt_current_stdev_list[index] = value
                    elif name == "OUT CHANNELS":
                        self.nt_current_outchannels_list[index] = value
                    elif name == "FILTER HEIGHT":
                        self.nt_current_filterheight_list[index] = value
                    elif name == "FILTER WIDTH":
                        self.nt_current_filterwidth_list[index] = value
        
            if cntrl_type == CHOICE_CTRL:
                value_list = value
                if index == -1:
                    if name == "NEURONETWORK STRIDE":
                        self.nt_current_stride_list.append(value_list)
                        self.nt_current_hpce_list.append(hpace)
                        self.nt_current_wpce_list.append(wpace)
                    elif name == "MAX POOL STRIDE":
                        self.mp_current_stride_list.append(value_list)
                        self.mp_current_hpce_list.append(hpace)
                        self.mp_current_wpce_list.append(wpace)
                    elif name == "MAX POOL KSIZE":
                        self.mp_current_ksize_list.append(value_list)
                        self.mp_current_fltrhght_list.append(hpace)
                        self.mp_current_fltrwdth_list.append(wpace)
                else:
                    if name == "NEURONETWORK STRIDE":
                        self.nt_current_stride_list[index] = value_list
                        self.nt_current_hpce_list[index] = hpace
                        self.nt_current_wpce_list[index] = wpace
                    elif name == "MAX POOL STRIDE":
                        self.mp_current_stride_list[index] = value_list
                        self.mp_current_hpce_list[index] = hpace
                        self.mp_current_wpce_list[index] = wpace
                    elif name == "MAX POOL KSIZE":
                        self.mp_current_ksize_list[index] = value_list
                        self.mp_current_fltrhght_list[index] = hpace
                        self.mp_current_fltrwdth_list[index] = wpace
    
        elif title_str == self.dl_title:
            if index == -1:
                if name == "WEIGHT":
                    self.dl_current_wt_list.append(value)
                elif name == "BIASES":
                    self.dl_current_bias_list.append(value)
                elif name == "ST. DEV.":
                    self.dl_current_stdev_list.append(value)
                elif name == "KEEP PROB.":
                    self.dl_current_keepprob_list.append(value)
                elif name == "NODES":
                    self.dl_current_nodes_list.append(value)
            else:
                if name == "WEIGHT":
                    self.dl_current_wt_list[index] = value
                elif name == "BIASES":
                    self.dl_current_bias_list[index] = value
                elif name == "ST. DEV.":
                    self.dl_current_stdev_list[index] = value
                elif name == "KEEP PROB.":
                    self.dl_current_keepprob_list[index] = value
                elif name == "NODES":
                    self.dl_current_nodes_list[index] = value
        
        elif title_str == self.cm_title:
            pass
        elif title_str == self.smd_title:
            if name == "WEIGHT":
                self.smd_current_wt = value
            elif name == "BIASES":
                self.smd_current_bias = value
            elif name == "ST. DEV.":
                self.smd_current_stdev = value
        
        elif title_str == self.ce_title:
            self.ce_current_stdev = value
            
        elif title_str == self.learning_title:
            if name == "LEARNING RATE":
                self.current_learningrate = value
            elif name == "LEARNING DECAY":
                self.current_learningdecay = value
        
    def set_all_parameters(self):
        for x in range(len(self.data_list)):
            (cntrl_type, title, name, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _, _,
             _, _) = self.data_list[x]
            
            if cntrl_type == SPIN_CTRL_DOUBLE:
    
                value = self.starting_holder_list[x].GetValue()
                if name == "NODES" or title == "Number of Steps" or name == "OUT CHANNELS":
                    value = int(value)
                self.select_parameter(cntrl_type, title, name, value)
                self.ongoing_holder_list[x].SetValue(repr(value))
                self.ongoing_holder_list[x].Update()
                self.starting_holder_list[x].Enable(False)
                self.starting_holder_list[x].Update()
                
                if not title == self.num_steps_title:
                    self.ending_holder_list[x].Enable(False)
                    self.ending_holder_list[x].Update()
                    self.step_holder_list[x].Enable(False)
                    self.step_holder_list[x].Update()
                
            elif cntrl_type == CHOICE_CTRL:
                choice_str = self.starting_holder_list[x].GetString(self.starting_holder_list[x].GetSelection())
                (hpace, wpace) = self.get_pace_value(choice_str)
                value = [1, int(hpace), int(wpace), 1]
                value_str = "[1, " + repr(int(hpace)) + ", " + repr(int(wpace)) + ", 1]"
                self.select_parameter(cntrl_type, title, name, value, hpace=hpace, wpace=wpace)
                self.ongoing_holder_list[x].SetValue(value_str)
                self.ongoing_holder_list[x].Update()
                self.starting_holder_list[x].Enable(False)
                self.starting_holder_list[x].Update()
                self.ending_holder_list[x].Enable(False)
                self.ending_holder_list[x].Update()
                self.step_holder_list[x].Enable(False)
                self.step_holder_list[x].Update()
                # wx.Yield()
                
            elif cntrl_type == DISPLAY_CTRL and not title == "Number of Steps":
                value_str = self.starting_holder_list[x].GetLabel()
                if not value_str == "":
                    self.select_parameter(cntrl_type, title, name, float(value_str))
                    self.starting_holder_list[x].Enable(False)
                    self.starting_holder_list[x].Update()
                    self.ongoing_holder_list[x].SetValue(value_str)
                    self.ongoing_holder_list[x].Update()
                    # wx.Yield()
             
    def set_ordered_hyperparameters_n_train(self):
        starting_value_list = []
        initial_starting_value_list = []
        ending_value_list = []
        step_value_list = []
        data_index_list = []
        index_list = []
        cntrl_type_list = []
        title_list = []
        name_list = []
        title_holder_list = []
        
        for x in range(len(self.ordering_list)):
            data_index = self.ordering_list[x]
            data_index_list.append(data_index)
    
            (cntrl_type, title, name, title_holder,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _,
             _, _, _, _,
             _, _) = self.data_list[data_index]
    
            cntrl_type_list.append(cntrl_type)
            title_list.append(title)
            name_list.append(name)
            title_holder_list.append(title_holder)
            
            str_holder = ""
            title_str = title_list[x]
            
            while True:
                if title_str[-1:].isnumeric():
                    str_holder = title_str[-1:] + str_holder
                    title_str = title_str[: -1]
                else:
                    if str_holder == "":
                        index_list.append(0)
                    else:
                        index_list.append(int(str_holder) - 1)
                    break
           
            if cntrl_type is SPIN_CTRL_DOUBLE:
                initial_starting_value_list.append(self.starting_holder_list[data_index].GetValue())
                starting_value_list.append(initial_starting_value_list[x])
                ending_value_list.append(self.ending_holder_list[data_index].GetValue())
                step_value_list.append(abs(self.step_holder_list[data_index].GetValue()))
            elif cntrl_type is CHOICE_CTRL:
                initial_starting_value_list.append(self.starting_holder_list[data_index].GetSelection())
                starting_value_list.append(initial_starting_value_list[x])
                ending_value_list.append(self.ending_holder_list[data_index].GetSelection())
                step_value_list.append(abs(self.step_holder_list[data_index].GetValue()))
            else:
                wx.MessageDialog(self, "Unable to find control type",
                                 "PROGRAMMING ERROR!",
                                 wx.ICON_EXCLAMATION | wx.OK).ShowModal()
                exit(1)
    
            if starting_value_list[x] > ending_value_list[x]:
                step_value_list[x] *= -1
            if starting_value_list[x] < ending_value_list[x]:
                self.empty_ordering_flag = False
    
        advance_token = False
        while True:
            order_finished_flag_list = [False] * len(self.ordering_list)
            for w in range(len(self.ordering_list) - 1, -1, -1):
                if advance_token:
                    starting_value_list[w] += step_value_list[w]
                    advance_token = False
                    if step_value_list[w] > -1:
                        if starting_value_list[w] > ending_value_list[w]:
                            advance_token = True
                            if w > 0:
                                starting_value_list[w] = initial_starting_value_list[w]
                            else:
                                starting_value_list[0] = ending_value_list[0]
                    else:
                        if starting_value_list[w] < ending_value_list[w]:
                            advance_token = True
                            if w > 0:
                                starting_value_list[w] = initial_starting_value_list[w]
                            else:
                                starting_value_list[0] = ending_value_list[0]

                value_str = ""
                if starting_value_list[w] == ending_value_list[w]:
                    order_finished_flag_list[w] = True
                else:
                    order_finished_flag_list[w] = False
                    if cntrl_type_list[w] is SPIN_CTRL_DOUBLE:
                        self.select_parameter(cntrl_type_list[w], title_list[w], name_list[w],
                                              value=starting_value_list[w], index=index_list[w])
                        value_str = repr(starting_value_list[w])
                    elif cntrl_type_list[w] is CHOICE_CTRL:
                        # print(" ### % s " % self.starting_holder_list[w].GetLabelText())
                        choice_index = self.ordering_list[w]
                        n = self.starting_holder_list[choice_index].GetSelection()
                        choice_str = self.starting_holder_list[choice_index].GetString(n)
                        (hpace, wpace) = self.get_pace_value(choice_str)
                        value = [1, int(hpace), int(wpace), 1]
                        self.select_parameter(cntrl_type_list[w], title_list[w], name_list[w],
                                              value=value, hpace=hpace, wpace=wpace, index=index_list[w])
                        value_str = "[1, " + repr(int(hpace)) + ", " + repr(int(wpace)) + ", 1]"
        
                    if name_list[w] == "OUT CHANNELS" and index_list[w] <= len(self.nt_current_channels_list) - 2:
                        self.nt_current_channels_list[index_list[w] + 1] = self.nt_current_outchannels_list[index_list[w]]
                        value_str = repr(self.nt_current_channels_list[index_list[w] + 1])
                        self.ongoing_holder_list[data_index_list[w] + 6].SetValue(value_str)
               
                self.ongoing_holder_list[data_index_list[w]].SetValue(value_str)
                
                if w == 1:
                    
                    self.ongoing_holder_list[data_index_list[w]].SetBackgroundColour(self.secondarycolor)
                    self.ongoing_holder_list[data_index_list[w]].Update()
                    self.name_holder_list[data_index_list[w]].SetBackgroundColour(self.secondarycolor)
                    self.name_holder_list[data_index_list[w]].Refresh()
                    
                elif w == 2:
                    self.ongoing_holder_list[data_index_list[w]].SetBackgroundColour(self.tertiarycolor)
                    self.ongoing_holder_list[data_index_list[w]].Update()
                    self.name_holder_list[data_index_list[w]].SetBackgroundColour(self.tertiarycolor)
                    self.name_holder_list[data_index_list[w]].Refresh()
                elif w == 3:
                    
                    self.ongoing_holder_list[data_index_list[w]].SetBackgroundColour(self.quaternarycolor)
                    self.ongoing_holder_list[data_index_list[w]].Update()
                    self.name_holder_list[data_index_list[w]].SetBackgroundColour(self.quaternarycolor)
                    self.name_holder_list[data_index_list[w]].Refresh()
                elif w >= 4:
                    
                    self.ongoing_holder_list[data_index_list[w]].SetBackgroundColour(self.highercolor)
                    self.ongoing_holder_list[data_index_list[w]].Update()
                    self.name_holder_list[data_index_list[w]].SetBackgroundColour(self.highercolor)
                    self.name_holder_list[data_index_list[w]].Refresh()
                elif w == 0:
                    self.ongoing_holder_list[data_index_list[w]].SetBackgroundColour(self.primarycolor)
                    self.ongoing_holder_list[data_index_list[w]].Update()
                    self.name_holder_list[data_index_list[w]].SetBackgroundColour(self.primarycolor)
                    self.name_holder_list[data_index_list[w]].Refresh()
                    advance_token = True

                # This print is needed to get the look correct on the screen. I cannot figure out why
                print("")
                
                # self.panel.Enable(False)
                
                # self.panel.Refresh()
    
            # Throw a thread to start an epoch of training
            t_train = None
            self.epoch_number += 1
            try:
                t_train = Thread(target=self.train)
                t_train.start()
            except:
                # Something went incorrectly with the thread throwing. Send user a message and exit
                self.epoch_number -= 1
                wx.MessageDialog(self, "Unable to start new thread to start neuronetwork",
                                 "THREADING ERROR!",
                                 wx.ICON_EXCLAMATION | wx.OK).ShowModal()
                exit(1)
            t_train.join()
            
            if all(order_finished_flag_list):
                break
        
    def convolution_mapit(self, nt_layer_count, layer_count, image_data, trained_variable_list):

        conv = tf.nn.conv2d(image_data, trained_variable_list[layer_count * 2],
                            self.nt_current_stride_list[nt_layer_count], padding='VALID')

        pre_activation = tf.nn.bias_add(conv, trained_variable_list[layer_count * 2 + 1])
        con = (tf.nn.relu(pre_activation))

        # Max_pool
        max_pool = (tf.nn.max_pool(con, ksize=self.mp_current_ksize_list[nt_layer_count],
                                   strides=self.mp_current_stride_list[nt_layer_count], padding='VALID'))

        return max_pool

    @staticmethod
    def multi_neuron_mapit(layer_count, image_data, trained_variable_list):

        multilayer = tf.nn.relu(tf.matmul(image_data, trained_variable_list[2 * layer_count])
                                + trained_variable_list[2 * layer_count + 1])

        return multilayer

    @staticmethod
    def sigmoid_mapit(layer_count, image_data, trained_variable_list):

        s_ = tf.add(tf.matmul(image_data, trained_variable_list[2 * layer_count]),
                    trained_variable_list[2 * layer_count + 1])

        return s_

    def mapit_ext(self, input_tensor, trained_variable_list):
        image_data = input_tensor
        layer_count = -1
        for layer in Logit_list:
            self.layer_type, number_layers = layer
            if self.layer_type == CONVOLUTION_LAYER:
                for x in range(number_layers):
                    layer_count += 1
                    image_data = self.convolution_mapit(x, layer_count, image_data, trained_variable_list)

            elif self.layer_type == MULTI_NEURON_LAYER:
                if not tf.rank(image_data) == 1:
                    # Flatten the image matrix, if this is the last set
                    image_data = tf.reshape(image_data, [BATCH_SIZE, -1])
                for x in range(number_layers):
                    layer_count += 1
                    image_data = self.multi_neuron_mapit(layer_count, image_data, trained_variable_list)

            elif self.layer_type == SIMOID_LAYER:
                layer_count += 1
                image_data = self.sigmoid_mapit(layer_count, image_data, trained_variable_list)

            elif self.layer_type == CROSS_ENTROPY_LAYER:
                pass
            elif self.layer_type == CROSS_ENTROPY_SIGMOID_LAYER:
                pass
            else:
                pass

        return image_data

    def mapit_(self, input_tensor, trained_variables_list):
        # Convulution layers
        nt_wt_list = []
        nt_ba_list = []

        dl_wt_list = []
        dl_ba_list = []

        con_list = []
        max_pool_list = []
        for x in range(NUMBER_CONVOLUTION_LAYERS):
            nt_wt_list.append(trained_variables_list[2 * x])
            nt_ba_list.append(trained_variables_list[2 * x + 1])

            if x == 0:
                conv = tf.nn.conv2d(input_tensor, nt_wt_list[x], self.nt_current_stride_list[x], padding='VALID')
                # saved_variable_list.clear()
            else:
                conv = tf.nn.conv2d(max_pool_list[x - 1], nt_wt_list[x], self.nt_current_stride_list[x], padding='VALID')

            pre_activation = tf.nn.bias_add(conv, nt_ba_list[x])
            con_list.append(tf.nn.relu(pre_activation))

            # Max_pool
            max_pool_list.append(tf.nn.max_pool(con_list[x], ksize=self.mp_current_stride_list[x],
                                                strides=self.mp_current_stride_list[x], padding='VALID'))

        # Flatten the image matrix
        flattened = tf.reshape(max_pool_list[NUMBER_CONVOLUTION_LAYERS - 1], [BATCH_SIZE, -1])

        # Multilayer Neurons layers
        multilayer_list = []
        for x in range(NUMBER_MULTI_NEURON_LAYERS):
            base_number = 2 * NUMBER_CONVOLUTION_LAYERS
            dl_wt_list.append(trained_variables_list[2 * x + base_number])
            dl_ba_list.append(trained_variables_list[2 * x + base_number + 1])

            if x == 0:
                multilayer_list.append(
                    tf.nn.relu(tf.matmul(flattened, dl_wt_list[x]) + dl_ba_list[x]))
            else:
                multilayer_list.append(
                    tf.nn.relu(tf.matmul(multilayer_list[x - 1], dl_wt_list[x]) + dl_ba_list[x]))

        # linear layer(WX + b),
        # We don't apply sigmoid here because
        # tf.nn.sigmoid_cross_entropy_with_logits accepts the unscaled logits
        # and performs the sigmoid internally for efficiency.

        # Sigmoid layer
        smd_wt = trained_variables_list[2 * (NUMBER_CONVOLUTION_LAYERS + NUMBER_MULTI_NEURON_LAYERS)]
        smd_ba = trained_variables_list[2 * (NUMBER_CONVOLUTION_LAYERS + NUMBER_MULTI_NEURON_LAYERS) + 1]

        s_ = tf.add(tf.matmul(multilayer_list[NUMBER_MULTI_NEURON_LAYERS - 1], smd_wt), smd_ba)

        return s_

    # noinspection PyUnusedLocal,PyUnusedLocal
    def calculate_accuracies_ext(self, interval_step, np_validate_y, np_validate_y_):
        # The following block calculate the accuracy in a different way to double check
        image_correct_flag = True
        total_positive_faces_number = 0.0
        total_negative_faces_number = 0.0
        total_images_number = 0.0
        correct_positive_faces_number = 0.0
        correct_negative_faces_number = 0.0
        correct_images_number = 0.0
        images_accuracy = 0.0
        total_face_accuracy = 0.0

        # Set up lists with 12 (NUMBER_POSSIBLE_FACES) zeros
        true_positive_list = [0.0] * NUMBER_POSSIBLE_FACES
        true_negative_list = [0.0] * NUMBER_POSSIBLE_FACES
        false_positive_list = [0.0] * NUMBER_POSSIBLE_FACES
        false_negative_list = [0.0] * NUMBER_POSSIBLE_FACES
        sensitivity_list = [0.0] * NUMBER_POSSIBLE_FACES
        specificity_list = [0.0] * NUMBER_POSSIBLE_FACES

        z = -1
        for x in range(len(np_validate_y_)):
            z += 1
            if np_validate_y_[x] == 1.0:
                total_positive_faces_number += 1.0
                # validate_y_ indicates that this face exist in the image
                # The total face number is increment by one
                if np_validate_y[x] == 1.0:
                    # validate_y recognizes the same face
                    # The number of correct faces is incremented by one
                    correct_positive_faces_number += 1.0
                    true_positive_list[z] += 1.0
                else:
                    # validate_y doesn't recognize the same face. Not only the number of correct faces
                    # is not incremented. The flag to alert an incorrect image recognition is set
                    # and false negative is incremented by one
                    image_correct_flag = False
                    false_negative_list[z] += 1.0
            else:
                total_negative_faces_number += 1.0
                # Validate_y_ doesn't think this face is in the image, but computer misrecognized that the same face
                # is in the image. Not only the number of correct faces is not incremented.
                # The flag to alert an incorrect image recognition is set
                # and the false positive is incremented by one
                if np_validate_y[x] == 1.0:
                    image_correct_flag = False
                    false_positive_list[z] += 1.0
                else:
                    # Both the validate_y_ and validate_y don't think that the face is there.
                    # True negative is incremented by one
                    correct_negative_faces_number += 1.0
                    true_negative_list[z] += 1.0
            if z == (NUMBER_POSSIBLE_FACES - 1):
                # If total number of faces equals NUMBER_POSSIBLE_FACES - 1,
                # the total image number is incremented by one
                total_images_number += 1.0
                # Reset the bits_count
                z = -1

                if image_correct_flag:
                    # If the image_correct_flag is still True, number of correct images increments by one
                    correct_images_number += 1.0

                # Reset the image_correct_flag
                image_correct_flag = True

        images_accuracy = correct_images_number / total_images_number
        
        # category_accuracy = correct_faces_number / total_faces_number
        if total_positive_faces_number == 0.0:
            positive_face_accuracy = 0.0
        else:
            positive_face_accuracy = correct_positive_faces_number / total_positive_faces_number

        if total_negative_faces_number == 0.0:
            negative_face_accuracy = 0.0
        else:
            negative_face_accuracy = correct_negative_faces_number / total_negative_faces_number

        if total_negative_faces_number + total_positive_faces_number == 0.0:
            total_face_accuracy = 0.0
        else:
            total_face_accuracy = (correct_positive_faces_number
                                   + correct_negative_faces_number) / (total_positive_faces_number
                                                                       + total_negative_faces_number)

        for x in range(NUMBER_POSSIBLE_FACES):
            if true_positive_list[x] == 0.0:
                sensitivity_list[x] = 0.0
            else:
                sensitivity_list[x] = true_positive_list[x] / (true_positive_list[x] + false_negative_list[x])

            if true_negative_list[x] == 0.0:
                specificity_list[x] = 0.0
            else:
                specificity_list[x] = true_negative_list[x] / (true_negative_list[x] + false_positive_list[x])

        self.np_stats[self.epoch_number, IMAGE_ACCURACY, interval_step] = images_accuracy
        self.np_stats[self.epoch_number, TOTAL_FACE_ACCURACY, interval_step] = total_face_accuracy
        self.np_stats[self.epoch_number, POSITIVE_FACE_ACCURACY, interval_step] = positive_face_accuracy
        self.np_stats[self.epoch_number, NEGATIVE_FACE_ACCURACY, interval_step] = negative_face_accuracy
        
        print(true_positive_list)
        print(true_negative_list)
        print(false_positive_list)
        print(false_negative_list)
        # print(sensitivity_list)
        # print(specificity_list)

        for x in range(NUMBER_POSSIBLE_FACES):
            self.np_stats[self.epoch_number,
                          Sensitivity_dict["SENSITIVITY" + repr(x)], interval_step] = sensitivity_list[x]
            self.np_stats[self.epoch_number,
                          Specificity_dict["SPECIFICITY" + repr(x)], interval_step] = specificity_list[x]

    @staticmethod
    def loss_(logits, labels):
        """Add L2Loss to all the trainable variables.
          Add summary for "Loss" and "Loss/avg".
          Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
          Returns:
            Loss tensor of type float.
          """
        # Calculate the average cross entropy loss across the batch.

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        # return tf.add_n(tf.get_collection('losses'), name='total_loss')

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.cast(tf.add_n(tf.get_collection('losses'), name='total_loss'), FLOATING_POINT_PRECISION)

    @staticmethod
    def add_loss_summaries(total_loss):
        """Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
        total_loss: Total loss from loss().
        Returns:
        loss_averages_op: op for generating moving averages of losses.
        """

        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for x in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
            global Tensor_Name
            tf.summary.scalar(Run + tensor_name + '___raw' + Tensor_Name, x)
            tf.summary.scalar(Run + tensor_name + '___average_loss' + Tensor_Name, loss_averages.average(x))

        return loss_averages_op

    def train_op_(self, total_loss, global_step):
        """Train_op
          Create an optimizer and apply to all trainable variables. Add moving
          average for all trainable variables.
          Args:
            total_loss: Total loss from loss().
            global_step: Integer Variable counting the number of training steps
              processed.
          Returns:
            train_op: op for training.
          """
        num_batches_per_epoch = NUMBER_PICTURES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        global Tensor_Name, Run
        tf.summary.scalar(Run + 'learning_rate' + Tensor_Name, lr)

        # Generate moving averages of all losses and associated summaries
        loss_averages_op = self.add_loss_summaries(total_loss)

        # Compute gradients
        with tf.control_dependencies([loss_averages_op]):
            # opt = tf.train.GradientDescentOptimizer(lr)
            # wx.Yield()
            opt = tf.train.AdamOptimizer()
            # wx.Yield()
            grads = opt.compute_gradients(total_loss)

        # Apply gradients
        # noinspection PyUnresolvedReferences
        # wx.Yield()
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables
        # wx.Yield()
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        # wx.Yield()
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

            return train_op, total_loss

    @staticmethod
    def activation_summary(x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args:
            x: Tensor
            Returns:
                nothing"""
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        global Tensor_Name, Run
        tf.summary.histogram(Run + tensor_name + '___/activations' + Tensor_Name, x)
        tf.summary.scalar(Run + tensor_name + '___/sparsity' + Tensor_Name, tf.nn.zero_fraction(x))

    @staticmethod
    def variable_on_cpu(name, shape, initializer):

        """Helper to create a Variable stored on CPU memory.
        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
        Returns:
            Variable Tensor"""

        with tf.device('/gpu:0'):
            try:
                dtype = FLOATING_POINT_PRECISION
                var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
                return var
            except ValueError:
                print("Memory for ", name, "cannot be initialized")
                exit(508)

    def variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
            Variable Tensor"""
        dtype = FLOATING_POINT_PRECISION
        if XAVIER:
            var = self.variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
        else:
            var = self.variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weights')
            tf.add_to_collection('losses', weight_decay)
        return var

    def nodes_needed(self):
        """Find the number of nodes needed to accommodate flattened image"""
        h1 = []
        h2 = []
        w1 = []
        w2 = []
        for x in range(NUMBER_CONVOLUTION_LAYERS):
            if x == 0:
                h1.append(math.floor((IMAGE_HEIGHT - self.nt_current_filterheight_list[x]) / self.nt_current_hpce_list[x]) + 1)
                w1.append(math.floor((IMAGE_WIDTH - self.nt_current_filterwidth_list[x]) / self.nt_current_wpce_list[x]) + 1)
            else:
                h1.append(math.floor((h2[x - 1] - self.nt_current_filterheight_list[x]) / self.nt_current_hpce_list[x]) + 1)
                w1.append(math.floor((w2[x - 1] - self.nt_current_filterwidth_list[x]) / self.nt_current_wpce_list[x]) + 1)

            h2.append(math.floor((h1[x] - self.mp_current_fltrhght_list[x]) / self.mp_current_hpce_list[x]) + 1)
            w2.append(math.floor((w1[x] - self.mp_current_fltrwdth_list[x]) / self.mp_current_wpce_list[x]) + 1)

        # CNN2_OUT_CHANNELS is from CNN1_CHANNELS * CNN1_OUT_CHANNELS/CNN1_CHANNELS * CNN2_OUT_CHANNELS/CNN2_CHANNELS,
        # where CNN2_CHANNELS = CNN1_OUT_CHANNELS.
        return int(h2[NUMBER_CONVOLUTION_LAYERS - 1] * w2[NUMBER_CONVOLUTION_LAYERS - 1]
                   * self.nt_current_outchannels_list[NUMBER_CONVOLUTION_LAYERS - 1])

    def convolution_layer(self, nt_layer_count, image_data):
        # noinspection PyUnresolvedReferences
        # wx.Yield()
        with tf.variable_scope("cnn" + repr(nt_layer_count + 1)) as scope:

            kernel = self.variable_with_weight_decay('weights',
                                                     shape=[self.nt_current_filterheight_list[nt_layer_count],
                                                            self.nt_current_filterwidth_list[nt_layer_count],
                                                            self.nt_current_channels_list[nt_layer_count],
                                                            self.nt_current_outchannels_list[nt_layer_count]],
                                                     stddev=self.nt_current_stdev_list[nt_layer_count],
                                                     wd=self.nt_current_wt_list[nt_layer_count])

            conv = tf.nn.conv2d(image_data, kernel, self.nt_current_stride_list[nt_layer_count], padding='VALID')

            biases = self.variable_on_cpu('biases', [self.nt_current_outchannels_list[nt_layer_count]],
                                          tf.constant_initializer(self.nt_current_bias_list[nt_layer_count]))

            pre_activation = tf.nn.bias_add(conv, biases)
            con = (tf.nn.relu(pre_activation, name=scope.name))
            self.activation_summary(con)

        # Max_pool
        # Performs the max pooling on output of conv1
        max_pool = tf.nn.max_pool(con, ksize=self.mp_current_ksize_list[nt_layer_count],
                                  strides=self.mp_current_stride_list[nt_layer_count], padding='VALID',
                                  name='max_pool' + repr(nt_layer_count + 1))

        # Batch Normalization1
        bn = tf.contrib.layers.batch_norm(max_pool, fused=True, data_format="NHWC", scope=scope)

        return bn, kernel, biases

    def multi_neuron_layer(self, dl_layer_count, image_data, last_layer_nodes):
        # wx.Yield()
        with tf.variable_scope('MLN' + repr(dl_layer_count + 1)) as scope:
            kernel = self.variable_with_weight_decay('weights', shape=[last_layer_nodes,
                                                                       self.dl_current_nodes_list[dl_layer_count]],
                                                     stddev=self.dl_current_stdev_list[dl_layer_count],
                                                     wd=self.dl_current_wt_list[dl_layer_count])

            biases = self.variable_on_cpu('biases', [self.dl_current_nodes_list[dl_layer_count]],
                                          tf.constant_initializer(self.dl_current_bias_list[dl_layer_count]))

            # Calculate the dropout
            keep_prob = tf.constant(self.dl_current_keepprob_list[dl_layer_count])

            flattened_drp = (tf.nn.dropout(image_data, keep_prob=keep_prob))

            # Computes rectified linear: max(flattened_drp, 0)
            multilayer = (tf.nn.relu(tf.matmul(flattened_drp, kernel) + biases, name=scope.name))

            # Document MLN1 summary
            self.activation_summary(multilayer)

        return multilayer, kernel, biases

    def sigmoid_layer(self, image_data):
        # linear layer(WX + b),
        # We don't apply sigmoid here because
        # tf.nn.sigmoid_cross_entropy_with_logits accepts the unscaled logits
        # and performs the sigmoid internally for efficiency.

        # SIGMOID_STANDARD_DEVIATION = 1 / ((IMAGE_HEIGHT - CNN1_FILTER_HEIGHT - CNN2_FILTER_HEIGHT + 1)
        #                                     * (IMAGE_WIDTH - CNN1_FILTER_WIDTH - CNN2_FILTER_WIDTH + 1))
        # wx.Yield()
        with tf.variable_scope('SMD') as scope:
            kernel = self.variable_with_weight_decay('weights',
                                                     shape=[self.dl_current_nodes_list[NUMBER_MULTI_NEURON_LAYERS - 1],
                                                            NUMBER_POSSIBLE_FACES],
                                                     stddev=self.smd_current_stdev,
                                                     wd=self.smd_current_wt)
            biases = self.variable_on_cpu('biases', [NUMBER_POSSIBLE_FACES],
                                          tf.constant_initializer(self.smd_current_bias))

            # Computes sigmoid of multilayer3
            s_ = tf.add(tf.matmul(image_data, kernel), biases, name=scope.name)

            # Document MLN3 summary
            self.activation_summary(s_)

            return s_, kernel, biases

    def logit_ext(self, batched_image):
        saved_variable_list = []
        image_data = batched_image
        first_flag = True
        for layer in Logit_list:
            self.layer_type, number_layers = layer
            if self.layer_type == CONVOLUTION_LAYER:
                for x in range(number_layers):
                    image_data, kernel, biases = self.convolution_layer(x, image_data)
                    if first_flag:
                        saved_variable_list.clear()
                        first_flag = False
                    saved_variable_list.append(kernel)
                    saved_variable_list.append(biases)

            elif self.layer_type == MULTI_NEURON_LAYER:
                if not tf.rank(image_data) == 1:
                    # Flatten the image matrix, if this is the last set
                    image_data = tf.reshape(image_data, [BATCH_SIZE, -1])
                last_layer_nodes = self.nodes_needed()
                for x in range(number_layers):
                    image_data, kernel, biases = self.multi_neuron_layer(x, image_data, last_layer_nodes)
                    last_layer_nodes = self.dl_current_nodes_list[x]
                    saved_variable_list.append(kernel)
                    saved_variable_list.append(biases)

            elif self.layer_type == SIMOID_LAYER:
                image_data, kernel, biases = self.sigmoid_layer(image_data)
                saved_variable_list.append(kernel)
                saved_variable_list.append(biases)

            elif self.layer_type == CROSS_ENTROPY_LAYER:
                pass
            elif self.layer_type == CROSS_ENTROPY_SIGMOID_LAYER:
                pass
            else:
                pass

        return image_data, saved_variable_list

    def logit_(self, batched_image):
        # Build the layers of the model
        global Run, Tensor_Name
        saved_variable_list = []
        bn_list = []
        con_list = []
        max_pool_list = []

        for x in range(NUMBER_CONVOLUTION_LAYERS):
            with tf.variable_scope("cnn" + repr(x + 1)) as scope:
                kernel = self.variable_with_weight_decay('weights',
                                                         shape=[self.nt_current_filterheight_list[x],
                                                                self.nt_current_filterwidth_list[x],
                                                                self.nt_current_channels_list[x],
                                                                self.nt_current_outchannels_list[x]],
                                                         stddev=self.nt_current_stdev_list[x],
                                                         wd=self.nt_current_wt_list[x])
                if x == 0:
                    conv = tf.nn.conv2d(batched_image, kernel, self.nt_current_stride_list[x], padding='VALID')
                    saved_variable_list.clear()
                    Run, Tensor_Name = get_tag()
                else:
                    conv = tf.nn.conv2d(bn_list[x - 1], kernel, self.nt_current_stride_list[x], padding='VALID')

                biases = self.variable_on_cpu('biases', [self.nt_current_outchannels_list[x]],
                                              tf.constant_initializer(self.nt_current_bias_list[x]))

                pre_activation = tf.nn.bias_add(conv, biases)
                con_list.append(tf.nn.relu(pre_activation, name=scope.name))
                self.activation_summary(con_list[x])
                saved_variable_list.append(kernel)
                saved_variable_list.append(biases)

            # Max_pool
            # Performs the max pooling on output of conv1
            max_pool_list.append(tf.nn.max_pool(con_list[x], ksize=self.mp_current_ksize_list[x],
                                                strides=self.mp_current_stride_list[x], padding='VALID',
                                                name='max_pool' + repr(x + 1)))

            # Batch Normalization1
            bn_list.append(tf.contrib.layers.batch_norm(max_pool_list[x], fused=True, data_format="NHWC", scope=scope))

        # Flatten the image matrix
        flattened = tf.reshape(bn_list[NUMBER_CONVOLUTION_LAYERS - 1], [BATCH_SIZE, -1])

        needed_nodes = self.nodes_needed()

        # Multilayer Neurons

        multilayer_list = []
        flattened_drp_list = []

        for x in range(NUMBER_MULTI_NEURON_LAYERS):
            with tf.variable_scope('MLN' + repr(x + 1)) as scope:
                if x == 0:
                    kernel = self.variable_with_weight_decay('weights',
                                                             shape=[needed_nodes, self.dl_current_nodes_list[x]],
                                                             stddev=self.dl_current_stdev_list[x],
                                                             wd=self.dl_current_wt_list[x])
                else:
                    kernel = self.variable_with_weight_decay('weights', shape=[self.dl_current_nodes_list[x - 1],
                                                                               self.dl_current_nodes_list[x]],
                                                             stddev=self.dl_current_stdev_list[x],
                                                             wd=self.dl_current_wt_list[x])

                biases = self.variable_on_cpu('biases', [self.dl_current_nodes_list[x]],
                                              tf.constant_initializer(self.dl_current_bias_list[x]))

                # Calculate the dropout
                keep_prob = tf.constant(self.dl_current_keepprob_list[x])

                # flattened_drp = None
                if x == 0:
                    flattened_drp_list.append(tf.nn.dropout(flattened, keep_prob=keep_prob))
                else:
                    flattened_drp_list.append(tf.nn.dropout(multilayer_list[x - 1], keep_prob=keep_prob))

                # Computes rectified linear: max(flattened, 0)
                multilayer_list.append(tf.nn.relu(tf.matmul(flattened_drp_list[x], kernel) + biases, name=scope.name))

                # Document MLN1 summary
                self.activation_summary(multilayer_list[x])
                saved_variable_list.append(kernel)
                saved_variable_list.append(biases)

        # linear layer(WX + b),
        # We don't apply sigmoid here because
        # tf.nn.sigmoid_cross_entropy_with_logits accepts the unscaled logits
        # and performs the sigmoid internally for efficiency.

        # SIGMOID_STANDARD_DEVIATION = 1 / ((IMAGE_HEIGHT - CNN1_FILTER_HEIGHT - CNN2_FILTER_HEIGHT + 1)
        #                                     * (IMAGE_WIDTH - CNN1_FILTER_WIDTH - CNN2_FILTER_WIDTH + 1))
        with tf.variable_scope('SMD') as scope:
            kernel = self.variable_with_weight_decay('weights',
                                                     shape=[self.dl_current_nodes_list[NUMBER_MULTI_NEURON_LAYERS - 1],
                                                            NUMBER_POSSIBLE_FACES],
                                                     stddev=self.smd_current_stdev,
                                                     wd=self.smd_current_wt)

            biases = self.variable_on_cpu('biases', [NUMBER_POSSIBLE_FACES],
                                          tf.constant_initializer(self.smd_current_bias))

            # Computes sigmoid of multilayer3
            s_ = tf.add(tf.matmul(multilayer_list[NUMBER_MULTI_NEURON_LAYERS - 1], kernel), biases, name=scope.name)

            # Document MLN3 summary
            self.activation_summary(s_)
            saved_variable_list.append(kernel)
            saved_variable_list.append(biases)

            return s_, saved_variable_list

    def train(self):
        if VERBOSE:
            print("Starting to train")
        start_prep_time = datetime.datetime.now().replace(microsecond=0)

        # if platform.system() == 'Windows':
        #     proc = psutil.Process(os.getpid())
        #     # proc.nice(psutil.HIGH_PRIORITY_CLASS)
        #     # proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        #     proc.nice(psutil.IDLE_PRIORITY_CLASS)

        # Train for a number of steps

        with tf.Graph().as_default():

            global Tensor_Name, Run
            Run, Tensor_Name = get_tag()
            # Get or create global_step from default tf.Graph
            # global_step = tf.contrib.framework.get_or_create_global_step()
            global_step = tf.train.get_or_create_global_step()

            # Get the filelist and labels
            # file_dict, labels = train_image_input()

            train_queue = tf.FIFOQueue(MAX_NUMBER_IN_TRAIN_QUEUE,
                                       dtypes=(FLOATING_POINT_PRECISION, FLOATING_POINT_PRECISION),
                                       shapes=[[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS],
                                               [BATCH_SIZE, NUMBER_POSSIBLE_FACES]], name="train_queue")

            # noinspection PyUnresolvedReferences
            # wx.Yield()

            validate_queue = tf.FIFOQueue(MAX_NUMBER_IN_VALIDATE_QUEUE,
                                          dtypes=(FLOATING_POINT_PRECISION, FLOATING_POINT_PRECISION),
                                          shapes=[[NUMBER_ACCURACY_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS],
                                                  [NUMBER_ACCURACY_IMAGES, NUMBER_POSSIBLE_FACES]],
                                          name="validate_queue")

            # wx.Yield()

            # Enqueue input data
            train_enqueue_op = train_queue.enqueue(self.train_image_input())

            # wx.Yield()

            validate_enqueue_op = validate_queue.enqueue(self.validate_image_input())

            # wx.Yield()

            # Create a training graph that starts by dequeuing a batch of pictures
            train_preprocessed_image, train_preprocessed_label = train_queue.dequeue()

            # wx.Yield()

            validate_preprocessed_image, validate_preprocessed_label = validate_queue.dequeue()

            # noinspection PyUnresolvedReferences
            # wx.Yield()

            # Take a sample of images and place into the Tensorboard
            tf.summary.image(Run + "sampled_images" + Tensor_Name, train_preprocessed_image)

            # wx.Yield()

            # Build a Graph that computes the logits predictions from the logit
            logits, variable_list = self.logit_ext(train_preprocessed_image)

            # wx.Yield()

            # Calculate loss
            loss = self.loss_(logits, train_preprocessed_label)

            # wx.Yield()

            # Build a Graph that trains the model with one batch of pictures and updates the model parameters
            train_op = self.train_op_(loss, global_step)

            # wx.Yield()

            # Create a queue runner that will run NUMBER_THREAD threads in parallel to enqueue examples
            train_qr = tf.train.QueueRunner(train_queue, [train_enqueue_op] * NUMBER_THREAD)

            # wx.Yield()

            validate_qr = tf.train.QueueRunner(validate_queue, [validate_enqueue_op] * NUMBER_THREAD)

            # wx.Yield()

            tf.train.add_queue_runner(train_qr)

            # wx.Yield()

            tf.train.add_queue_runner(validate_qr)

            # wx.Yield()

            # This following statements must be run before the self.interactive_sess is created
            # otherwise, tf.Graph is closed for variable initiation
            init_op = tf.global_variables_initializer()

            # wx.Yield()

            # Summary of all data for Tensorboard
            summary_op = tf.summary.merge_all()

            wx.Yield()

            # interactive_sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
            #                                          log_device_placement=True)
            # interactive_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

            # Limit only one cpu core to run tensorflow and leave other cores for other programs
            config = tf.ConfigProto(device_count={'CPU': 2})
            _sess = tf.Session(config=config)
            # _sess = tf.Session()

            # noinspection PyUnresolvedReferences
            wx.Yield()

            # interactive_sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            # The valuables needs to be inialized before the queue is started,
            # otherwise the queue may be emptied
            _sess.run(init_op)

            # wx.Yield()

            # Create a coordinator, launch the queue runner threads
            coord = tf.train.Coordinator()

            # noinspection PyUnresolvedReferences
            # wx.Yield()

            enqueue_threads = tf.train.start_queue_runners(_sess, coord=coord, start=True)

            # wx.Yield()

            best_image_accuracy = 0.0
            best_faces_accuracy = 0.0
            best_image_step = ""
            best_faces_step = ""
            print()
            prep_time = datetime.datetime.now().replace(microsecond=0) - start_prep_time
            print("prep time")
            print(prep_time)
            # start_timer = datetime.datetime.now().replace(microsecond=0)
            start_timer = datetime.datetime.now().replace(microsecond=0)

            time_elapsed = 0

            for step in range(int(self.num_steps_per_epoch)):
                wx.Yield()
                if coord.should_stop():
                    break

                # Adjust the step number
                true_step = step + 1

                # Save data into tensorboard every 20 steps
                if true_step % 20 == 0:
                    # wx.Yield()

                    if true_step % 100 == 0:
                        # Prepare to change folds
                        self.change_fold = True
                        train_queue.close()
                        validate_queue.close()
                        train_enqueue_op = train_queue.enqueue(self.train_image_input())
                        validate_enqueue_op = validate_queue.enqueue(self.validate_image_input())
                        # train_preprocessed_image, train_preprocessed_label = train_queue.dequeue()
                        # validate_preprocessed_image, validate_preprocessed_label = validate_queue.dequeue()
                        # tf.summary.image(Run + "sampled_images" + Tensor_Name, train_preprocessed_image)
                        # logits, variable_list = self.logit_ext(train_preprocessed_image)
                        # loss = self.loss_(logits, train_preprocessed_label)
                        # train_op = self.train_op_(loss, global_step)
                        train_qr = tf.train.QueueRunner(train_queue, [train_enqueue_op] * NUMBER_THREAD)
                        validate_qr = tf.train.QueueRunner(validate_queue, [validate_enqueue_op] * NUMBER_THREAD)
                        tf.train.add_queue_runner(train_qr)
                        tf.train.add_queue_runner(validate_qr)
                        # enqueue_threads = tf.train.start_queue_runners(_sess, coord=coord, start=True)

                    interval_step = int(true_step / self.num_steps_per_report) - 1
                    # create log writer object
                    writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR, graph=tf.get_default_graph())

                    _, summary, cur_loss = _sess.run([train_op, summary_op, loss])

                    # epoch = self.epoch_number - 1
                    
                    # save the loss in the self.np_stat
                    self.np_stats[self.epoch_number, LOSS, interval_step] = cur_loss

                    # write log
                    writer.add_summary(summary, step)

                    validate_y = tf.round(tf.sigmoid(self.mapit_ext(validate_preprocessed_image, variable_list)))
                    validate_y_ = tf.round(validate_preprocessed_label)

                    validate_y = tf.reshape(validate_y, [NUMBER_POSSIBLE_FACES * NUMBER_ACCURACY_IMAGES])
                    validate_y_ = tf.reshape(validate_y_, [NUMBER_POSSIBLE_FACES * NUMBER_ACCURACY_IMAGES])

                    tensorflow_category_correct_prediction = tf.equal(validate_y, validate_y_)

                    # The following two lines converts the true/false tensors into floating points,
                    # further reduce to a scalar mean.
                    tensorflow_category_accuracy = tf.reduce_mean(tf.cast(tensorflow_category_correct_prediction,
                                                                          tf.float32))

                    # Place calculated results into Tensorboard
                    tf.summary.scalar(Run + 'validate_category_accuracy' + Tensor_Name, tensorflow_category_accuracy)

                    validate_category_accuracy = _sess.run(tensorflow_category_accuracy)

                    self.np_stats[self.epoch_number, TENSORFLOW_FACE_ACCURACY, interval_step] = validate_category_accuracy

                    np_validate_y = _sess.run(validate_y)
                    np_validate_y_ = _sess.run(validate_preprocessed_label)

                    np_validate_y = np_validate_y.reshape(-1)
                    np_validate_y_ = np_validate_y_.reshape(-1)

                    self.calculate_accuracies_ext(interval_step, np_validate_y, np_validate_y_)

                    # print loss and accuracies

                    # Stamping the time
                    # run_timer = datetime.datetime.now().replace(microsecond=0) - start_timer
                    run_timer = datetime.datetime.now().replace(microsecond=0) - start_timer
                    # print("run time")
                    # print(run_timer)
                    # print("true_step")
                    # print(true_step)
                    if not self.epoch_time_established_flag:
                        # The number multiplying the prep_time is imperical to make sure the
                        # self.estimated_epoch_time is long enough to cover the remaining_time
                        if true_step == 20:
                            self.estimated_epoch_time = run_timer * 50 + prep_time * 40
                        elif true_step == 100:
                            self.estimated_epoch_time = run_timer * 10 + prep_time * 38
                        elif true_step == 200:
                            self.estimated_epoch_time = run_timer * 5 + prep_time * 34
                        elif true_step == 500:
                            self.estimated_epoch_time = run_timer * 2 + prep_time * 22
                    if true_step == 1000:
                        self.estimated_epoch_time = run_timer + prep_time * 3
                        self.epoch_time_established_flag = True

                    # remaining_time = self.estimated_epoch_time
                    if self.estimated_epoch_time > run_timer + prep_time:

                        remaining_time = self.estimated_epoch_time - run_timer - prep_time
                    else:
                        remaining_time = self.estimated_epoch_time + run_timer + prep_time

                    self.previous_epoch_time = self.estimated_epoch_time
                    # print("est. epoch time")
                    # print(self.estimated_epoch_time)

                    # # self.ongoing_holder_list[len(self.ongoing_holder_list) - 4].SetValue(str(estimated_epoch_time))
                    if self.previous_time_elapsed is not None:
                        time_elapsed = prep_time + run_timer + self.previous_time_elapsed
                    else:
                        time_elapsed = prep_time + run_timer
                    # time_remaining_for_epoch = run_timer * int((NUMBER_STEPS_PER_EPOCH - true_step) / true_step)

                    # print("time remaining for epoch")
                    # print(time_remaining_for_epoch)
                    # print("time elapsed")
                    # print(time_elapsed)
                    #
                    # print("prevous elapsed time")
                    # print(self.previous_time_elapsed)
                    # print("number of epochs")
                    # print(self.num_epoch)
                    # self.ongoing_holder_list[len(self.ongoing_holder_list) - 3].SetValue(str(time_elapsed))

                    self.time_to_complete = self.estimated_epoch_time * self.epoch_remaining + remaining_time

                    # print("epoch remaining")
                    # print(self.epoch_remaining)
                    # print("time to complete")
                    # print(self.time_to_complete)
                    
                    print_str1 = "TIME ELAPSED: {}   EPOCH(S) REMAINING: {}   EST. TIME FOR ONE EPOCH: {}"
                    print_str2 = "   EST. TIME TO COMPLETE: {}"
                    print_str = print_str1 + print_str2
                    print(print_str.format(time_elapsed, self.epoch_remaining, self.estimated_epoch_time,
                                           self.time_to_complete))
                    # ongoing_holder_index = len(self.ongoing_holder_list) - 4
                    # time_str = str(self.estimated_epoch_time)
                    # self.ongoing_holder_list[ongoing_holder_index].SetValue(time_str)
                    # self.ongoing_holder_list[ongoing_holder_index].Update()
                    # ongoing_holder_index = len(self.ongoing_holder_list) - 3
                    # time_str = str(time_elapsed)
                    # self.ongoing_holder_list[ongoing_holder_index].SetValue(time_str)
                    # self.ongoing_holder_list[ongoing_holder_index].Update()
                    # ongoing_holder_index = len(self.ongoing_holder_list) - 2
                    # time_str = str(time_to_complete)
                    # self.ongoing_holder_list[ongoing_holder_index].SetValue(time_str)
                    # self.ongoing_holder_list[ongoing_holder_index].Update()

                    # Do a formated print
                    print_str1 = "{} {:04d}/{:04d} TIME:{} LOSS:{:2.6f} "
                    print_str2 = "ACC'S.: pos.:{:.3f} neg.:{:.3f} t.:{:.3f} Tnrflw.:{:.3f} img.:{:.3f}"
                    print_str = print_str1 + print_str2
                    print(print_str.format(Run, true_step, NUMBER_STEPS_PER_EPOCH, run_timer, cur_loss,
                                           self.np_stats[self.epoch_number, POSITIVE_FACE_ACCURACY, interval_step],
                                           self.np_stats[self.epoch_number, NEGATIVE_FACE_ACCURACY, interval_step],
                                           self.np_stats[self.epoch_number, TOTAL_FACE_ACCURACY, interval_step],
                                           self.np_stats[self.epoch_number, TENSORFLOW_FACE_ACCURACY, interval_step],
                                           self.np_stats[self.epoch_number, IMAGE_ACCURACY, interval_step]), end="")
                    
                    # Do the printing of the sensitivities and specificities too, if they are short enough
                    if NUMBER_POSSIBLE_FACES <= 18:
                        print("\nsen.=", end="")
                        print_str = "{:.3f}"
                        for x in range(NUMBER_POSSIBLE_FACES):
                            print(print_str.format(self.np_stats[self.epoch_number,
                                                                 Sensitivity_dict["SENSITIVITY" + repr(x)],
                                                                 interval_step]), end=" ")

                        print("\nspe.=", end="")
                        for x in range(NUMBER_POSSIBLE_FACES):
                            print(print_str.format(self.np_stats[self.epoch_number,
                                                                 Specificity_dict["SPECIFICITY" + repr(x)],
                                                                 interval_step]), end=" ")

                    print()

                    # Save the best accuracies
                    if self.np_stats[self.epoch_number, IMAGE_ACCURACY, interval_step] > best_image_accuracy:
                        best_image_accuracy = self.np_stats[self.epoch_number, IMAGE_ACCURACY, interval_step]
                        best_image_step = true_step

                    if self.np_stats[self.epoch_number, TOTAL_FACE_ACCURACY, interval_step] > best_image_accuracy:
                        best_faces_accuracy = self.np_stats[self.epoch_number, TOTAL_FACE_ACCURACY, interval_step]
                        best_faces_step = true_step

                    enqueue_threads = tf.train.start_queue_runners(_sess, coord=coord, start=True)

                    if (step + 1) % NUMBER_STEPS_PER_EPOCH == 0:
                        # Add ops to save and restore all the variables.
                        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=3)
                        # Save the model and variables to disk.
                        save_path = saver.save(_sess, SAVE_MODEL_FILE,
                                               global_step=NUMBER_STEPS_PER_EPOCH)
                        print("Model saved in file: %s" % save_path)

                else:
                    # Plain vanilla training
                    # wx.Yield()
                    _sess.run(train_op)

            # When done, ask the threads to stop
            coord.request_stop()

            # Wait for all threads to actually do it
            coord.join(enqueue_threads)

            # Stamp the final time
            final_timer = datetime.datetime.now() - start_timer
            final_print_str1 = "Total time elapse for epoch: {}; Best IMAGE accuracy: {:.6f} at step {:04d};"
            final_print_str2 = "Best FACES accuracy: {:.6f} at step {:04d}"
            final_print_str = final_print_str1 + final_print_str2
            print(final_print_str.format(final_timer, best_image_accuracy, best_image_step,
                                         best_faces_accuracy, best_faces_step))
            
            self.previous_time_elapsed = time_elapsed
            self.epoch_remaining -= 1
            if self.epoch_remaining == -1:
                # set time to complete to zero
                self.time_to_complete = self.time_to_complete - self.time_to_complete

        if VERBOSE:
            print("One epoch training finished")

    @staticmethod
    def shrink_images(mode="train"):
        # Shrink the images and put them in corresponding directories for easy future usage. Only data to be put in by
        # human is in the TRAIN_DATA_DIR and TEST_DATA_DIR. The jpeg files in the those direction must have label attached
        # to the end of the file name before ".jped" . The format should of the label attachment should be a string of
        # 12 '0' and '1'. The positions of the '1' denote the presence of a certain predetermined baby or adult faces in
        # the picture. The predetermined baby or adult faces are arbitrary as far as this learning program is concern.
        # The current format is in the following:
        # baby1 baby2, baby3, father, monther, fraturnal grandpa, fraturnal grandma, maturnal grandpa, maturnal grandma,
        # uncle, aunt, babysitter, other adult

        # mode_str = ""
        # if  mode == "train":
        #     mode_str = "train"
        # else:
        #     mode_str = "test"

        if VERBOSE:
            # print("Start to shrink " + mode_str + " images")
            print("Start to shrink " + mode + " images")

        # size = IMAGE_HEIGHT, IMAGE_WIDTH
        # data_dir = ""
        # comp_data_dir = ""
        # rgb_data_dir = ""
        # label_dir = ""
        # Mode must be 'train' or 'test', otherwise will bomb out
        if mode == "train":
            data_dir = TRAIN_DATA_DIR
            comp_data_dir = COMPRESSED_TRAIN_DATA_DIR
            rgb_data_dir = RBG_TRAIN_DATA_DIR
            label_dir = TRAIN_LABEL_DIR
        elif 'test':
            data_dir = TEST_DATA_DIR
            comp_data_dir = COMPRESSED_TEST_DATA_DIR
            rgb_data_dir = RBG_TEST_DATA_DIR
            label_dir = TEST_LABEL_DIR
        # else:
        #     # noinspection PyUnreachableCode
        #     print("mode must be 'train' or 'test'")
        #     exit(400)

        if not os.path.exists(data_dir):
            print("No train data directory: ", data_dir, " and no baby picture files exist")
            exit(401)
        if not os.path.exists(comp_data_dir):
            print("No train data directory: ", comp_data_dir)
            print(comp_data_dir, "will be created")
            os.mkdir(comp_data_dir)
        if not os.path.exists(rgb_data_dir):
            print("No processed data directory: ", rgb_data_dir)
            print(rgb_data_dir, "will be created")
            os.mkdir(rgb_data_dir)

        # Fetch all files in the train data directories
        filenames = os.listdir(data_dir)
        shrink_filenames = os.listdir(comp_data_dir)

        if mode == "train":
            # Setting up global valuables for train_get_data()
            global Max_Number_File_in_Each_Fold, Max_Number_Train_Pictures, Max_Number_Files_in_Validate_Folds
            global Max_Training_Steps_in_Fold
            Max_Number_Train_Pictures = len(filenames)
            Max_Number_File_in_Each_Fold = math.floor(Max_Number_Train_Pictures / NUMBER_FOLDS)
            Max_Training_Steps_in_Fold = NUMBER_STEPS_PER_EPOCH / 5

            # Check for no existence of compressed jpeg in data and make one
        for file in filenames:
            found = False
            for shrink_file in shrink_filenames:
                if file == shrink_file:
                    found = True
                    break
            if not found:
                if len(file) >= 18:
                    if not file[-17] == "_" and not file[-18] == "_":
                        print("Poorly labeled file: ", file)
                        print(file, "will be ignored.")
                        break

                if file[-4:].lower() == ".jpg" or file[-5:].lower() == ".jpeg":
                    # Get the image, crop into a square and shrink it
                    im = None
                    try:
                        full_filename = os.path.join(data_dir, file)
                        im = Image.open(full_filename)
                    except IOError:
                        print("cannot open ", file)
                        exit(500)

                    # Get the average of the two dimensions. Then pad and crop the image
                    im_height, im_width = im.size[1], im.size[0]
                    image_dimension = int((im_width + im_height) / 2)

                    new_im = None
                    if not im_height == im_width:
                        if im_height > image_dimension:
                            margin = int((im_height - image_dimension) / 2)
                            new_image = im.crop((0, margin, im_width, im_height - margin))
                            new_im = Image.new("RGB", (image_dimension, image_dimension), color=(240, 240, 240))
                            new_im.paste(new_image, (margin, 0))
                        else:
                            margin = int((im_width - image_dimension) / 2)
                            new_image = im.crop((margin, 0, im_width - margin, im_height))
                            new_im = Image.new("RGB", (image_dimension, image_dimension), color=(240, 240, 240))
                            new_im.paste(new_image, (0, margin))
                    new_im.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

                    try:
                        compressed_file = os.path.join(comp_data_dir, file)
                        new_im.save(compressed_file)

                    except IOError:
                        print("Cannot save ", file)
                        exit(501)

                    # Read the color values and put them in a 3-dimension (channels, width, height) matrix
                    rgb_np_matrix = np.empty(shape=(IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS), dtype=np.uint8)

                    # The *_index are the starting point of the future shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
                    index = 0
                    for h in range(0, IMAGE_HEIGHT, 1):
                        for w in range(0, IMAGE_WIDTH, 1):
                            r, g, b = new_im.getpixel((w, h))
                            rgb_np_matrix[index] = np.uint8(r)
                            index += 1
                            rgb_np_matrix[index] = np.uint8(g)
                            index += 1
                            rgb_np_matrix[index] = np.uint8(b)
                            index += 1

                    rgb_np_matrix = np.float32(rgb_np_matrix)

                    # Stripe off the .jpeg or similar endings
                    full_rgb_filename = os.path.join(rgb_data_dir, file)
                    for x in range(0, -1 * len(full_rgb_filename), -1):
                        if full_rgb_filename[x] == ".":
                            full_rgb_filename = full_rgb_filename[0: x]
                            break

                    # Save the rgb_np_matrix as a npy file
                    try:
                        np.save(full_rgb_filename, rgb_np_matrix)
                    except IOError:
                        print("cannot save ", full_rgb_filename)
                        exit(502)

        # Process label and save it
        if not os.path.exists(label_dir):
            print("No label directory: ", label_dir, "and a new one will be created")
            os.mkdir(label_dir)

        file_list = []
        label_list = []

        rgb_filenames = os.listdir(rgb_data_dir)

        for file in rgb_filenames:
            # Slice the label from the backend of the file name
            label = []
            for y in range(0, NUMBER_POSSIBLE_FACES):
                label.append(file[y - 16])
            # Add the directory path to the npy file dictionary
            file_list.append(file)
            label_list.append(label)

        label_filename = os.path.join(label_dir, "label.txt")

        file_filename = os.path.join(label_dir, "file.txt")

        try:
            label_file = open(label_filename, "w")
            json.dump(label_list, label_file)
            label_file.close()
        except IOError:
            print("cannot save label_list into ", label_filename)
            exit(503)

        try:
            file_file = open(file_filename, "w")
            json.dump(file_list, file_file)
            file_file.close()
        except IOError:
            print("cannot save file_list into ", file_filename)
            exit(504)

        if VERBOSE:
            print("Finished with shrinking " + mode + " images")

    def train_image_input(self):
        # global Train_File_Dict, Validate_File_Dict, self.change_fold

        # Check for existence of fold file
        if not os.path.exists(TENSORBOARD_LOG_DIR):
            print("No train log directory exist and a new one will be created.")
            os.mkdir(TENSORBOARD_LOG_DIR)
        if not os.path.exists(WORKING_DIR):
            print("No tag directory exist and a new one will be created.")
            os.mkdir(WORKING_DIR)

        # Process to make train dictionary and validate dictionary
        if self.change_fold:  # need to find a place to make self.change_fold = True
            # label_list = []
            # file_list = []

            # Set up, train label directory, and label.txt
            if not os.path.exists(TRAIN_LABEL_DIR):
                print("No train label directory: ", TRAIN_LABEL_DIR)
                exit(506)

            # Read in label list
            if len(self.label_list) == 0:
                try:
                    label_filename = os.path.join(TRAIN_LABEL_DIR, "label.txt")
                    label_file = open(label_filename, 'r+')
                    self.label_list = json.load(label_file)
                    label_file.close()
                except IOError:
                    print("cannot open and read label.txt")
                    exit(507)

            # Read in file list
            if len(self.file_list) == 0:
                try:
                    file_filename = os.path.join(TRAIN_LABEL_DIR, "file.txt")
                    file_file = open(file_filename, 'r+')
                    self.file_list = json.load(file_file)
                    file_file.close()
                except IOError:
                    print("cannot open and read file.txt")
                    exit(508)

            # Pick three random folds out of fifteen
            fold_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
            self.num_folds = len(fold_list)

            fold1 = int(random.choice(fold_list))
            fold_list.__delitem__(int(fold1))
            fold2 = int(random.choice(fold_list))
            if fold2 > fold1:
                delete_fold2 = fold2 - 1
            else:
                delete_fold2 = fold2
            fold_list.__delitem__(int(delete_fold2))
            fold3 = int(random.choice(fold_list))

            self.max_num_file_in_each_fold = math.floor(len(self.file_list) / self.num_folds)

            for x in range(self.num_folds):
                # for x in range(fold_list_len):
                if x == fold1 or x == fold2 or x == fold3:
                    for y in range(int(self.max_num_file_in_each_fold) - 1):
                        index = int(x * self.max_num_file_in_each_fold + y)
                        self.validate_file_dict[(self.file_list[index])] = np.float32(self.label_list[index])
                else:
                    for y in range(int(self.max_num_file_in_each_fold) - 1):
                        index = int(x * self.max_num_file_in_each_fold + y)
                        self.train_file_dict[(self.file_list[index])] = np.float32(self.label_list[index])

            self.change_fold = False

        image_list = []
        labels_list = []
        # Get BATCH_SIZE random files with labels
        for x in range(BATCH_SIZE):
            random_file, label = random.choice(list(self.train_file_dict.items()))
            # Number_Steps_Used_in_Fold += 1

            # Check for existence of RGBtrain data directory
            if not os.path.exists(RBG_TRAIN_DATA_DIR):
                print("No train data directory: ", RBG_TRAIN_DATA_DIR)
                exit(505)

            # Set up random image list
            full_rgb_file = os.path.join(RBG_TRAIN_DATA_DIR, random_file)

            # Get the rgb data from file
            rgb_data = [IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS]
            try:
                rgb_data = np.load(full_rgb_file)
            except IOError:
                print("cannot open ", full_rgb_file)
                exit(507)

            # Preprocess image
            raw_image = tf.convert_to_tensor(rgb_data)
            distorted_image = tf.reshape(raw_image, [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])

            # Randomly flip the image horizontally
            if FLIP:
                distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Randomly change the brightness of the image
            if ALTER_BRIGHTNESS:
                distorted_image = tf.image.random_brightness(distorted_image, max_delta=MAXIMUM_BRIGHTNESS_DELTA)

            # Radomly change the contrast of the image
            if ALTER_CONTRAST:
                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower=LOWER_CONTRAST_CHANGE, upper=UPPER_CONTRAST_CHANGE)

            # Put the batched images in a list
            image_list.append(distorted_image)

            # Convert the labels into a tensor and place it in a list
            label_tensor = tf.convert_to_tensor(np.float32(label))
            labels_list.append(label_tensor)

        # Batch the image and label list
        random_image_list = tf.convert_to_tensor(image_list)
        random_labels_list = tf.convert_to_tensor(labels_list)

        # Test for time to change the folds
        # if Number_Steps_Used_in_Fold >= Max_Training_Steps_in_Fold:
        #     self.change_fold = True
        #     Number_Steps_Used_in_Fold = 0
        return random_image_list, random_labels_list

    def validate_image_input(self):
        # global Validate_File_Dict
        images_list = []
        labels_list = []

        for x in range(NUMBER_ACCURACY_IMAGES):
            # Get random validate file with label
            random_file, label = random.choice(list(self.validate_file_dict.items()))

            # Set up random image list
            full_rgb_file = os.path.join(RBG_TRAIN_DATA_DIR, random_file)

            # Get the rgb data from file
            rgb_data = [IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS]
            try:
                rgb_data = np.load(full_rgb_file)
            except IOError:
                print("cannot open ", full_rgb_file)
                exit(507)

            # Preprocess image
            raw_image = tf.convert_to_tensor(rgb_data)
            distorted_image = tf.reshape(raw_image, [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])

            # Randomly flip the image horizontally
            if FLIP:
                distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Randomly change the brightness of the image
            if ALTER_BRIGHTNESS:
                distorted_image = tf.image.random_brightness(distorted_image, max_delta=MAXIMUM_BRIGHTNESS_DELTA)

            # Radomly change the contrast of the image
            if ALTER_CONTRAST:
                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower=LOWER_CONTRAST_CHANGE, upper=UPPER_CONTRAST_CHANGE)

            # Convert the labels into a tensor and place it in a list
            label_tensor = tf.convert_to_tensor(np.float32(label))
            labels_list.append(label_tensor)

            # Place the images in a list
            images_list.append(distorted_image)

        images_list = tf.convert_to_tensor(images_list)
        labels_list = tf.convert_to_tensor(labels_list)

        return images_list, labels_list
    
    
# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
class ChangeColorDialog(wx.Dialog):
    def __init__(self, parent, color):
        super(ChangeColorDialog, self).__init__(parent, title="Change Ordering Color",
                                                style=wx.TAB_TRAVERSAL | wx.FRAME_FLOAT_ON_PARENT | wx.CAPTION)
        
        self.parent = parent
        self.primarycolor = color[0]
        self.secondarycolor = color[1]
        self.tertiarycolor = color[2]
        self.quaternarycolor = color[3]
        self.highercolor = color[4]

        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        self.primaryColorButton = wx.Button(self, wx.ID_ANY, u"PRIMARY", wx.DefaultPosition, wx.Size(90, 30),
                                            wx.NO_BORDER)
        self.primaryColorButton.SetBackgroundColour(self.primarycolor)
        # noinspection PyUnresolvedReferences
        self.secondaryColorButton = wx.Button(self, wx.ID_ANY, u"SECONDARY", wx.DefaultPosition, wx.Size(90, 30),
                                              wx.NO_BORDER)
        self.secondaryColorButton.SetBackgroundColour(self.secondarycolor)
        # noinspection PyUnresolvedReferences
        self.tertiaryColorButton = wx.Button(self, wx.ID_ANY, u"TERTIARY", wx.DefaultPosition, wx.Size(90, 30),
                                             wx.NO_BORDER)
        self.tertiaryColorButton.SetBackgroundColour(self.tertiarycolor)
        self.quaternaryColorButton = wx.Button(self, wx.ID_ANY, u"QUATERNARY", wx.DefaultPosition, wx.Size(90, 30),
                                               wx.NO_BORDER)
        self.quaternaryColorButton.SetBackgroundColour(self.quaternarycolor)
        self.higherColorButton = wx.Button(self, wx.ID_ANY, u"HIGHER", wx.DefaultPosition, wx.Size(90, 30),
                                           wx.NO_BORDER)
        self.higherColorButton.SetBackgroundColour(self.highercolor)

        # noinspection PyUnresolvedReferences
        self.okButton = wx.Button(self, wx.ID_OK, u"OK", wx.DefaultPosition, wx.Size(90, 30), 0)
        self.okButton.SetBackgroundColour(wx.Colour(225, 225, 225))
        self.okButton.Enable(False)
        # noinspection PyUnresolvedReferences
        self.defaultButton = wx.Button(self, wx.ID_OK, u"Default", wx.DefaultPosition, wx.Size(90, 30), 0)
        self.cancelButton = wx.Button(self, wx.ID_ANY, u"CANCEL", wx.DefaultPosition, wx.Size(90, 30), 0)
        self.cancelButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        self.init_ui()
        self.binding()
    
    def init_ui(self):
        # noinspection PyUnresolvedReferences
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.AddSpacer(40)
        # noinspection PyUnresolvedReferences
        sizer1.Add(self.okButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer1.Add(self.defaultButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sizer1.Add(self.cancelButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sizer2 = wx.BoxSizer(wx.VERTICAL)
        sizer2.Add(wx.StaticText(self, wx.ID_ANY, "Select to change color"), 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer2.Add(self.primaryColorButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sizer2.Add(self.secondaryColorButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer2.Add(self.tertiaryColorButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer2.Add(self.quaternaryColorButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer2.Add(self.higherColorButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer2.Add(sizer1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        
        self.SetSizerAndFit(sizer2)
        self.Center()
        self.Show()
    
    def binding(self):
        self.okButton.Bind(wx.EVT_LEFT_DOWN, self.on_ok)
        self.cancelButton.Bind(wx.EVT_LEFT_DOWN, self.on_cancel)
        self.defaultButton.Bind(wx.EVT_LEFT_DOWN, self.on_default)
        self.primaryColorButton.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        self.secondaryColorButton.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        self.tertiaryColorButton.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        # noinspection PyUnresolvedReferences
        self.quaternaryColorButton.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
        self.higherColorButton.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
    
    def on_left_click(self, event):
        self.okButton.Enable()
        self.primaryColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.primaryColorButton.Update()
        self.secondaryColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.secondaryColorButton.Update()
        # noinspection PyUnresolvedReferences
        self.tertiaryColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.tertiaryColorButton.Update()
        self.quaternaryColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.quaternaryColorButton.Update()
        self.higherColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.higherColorButton.Update()
        
        widget_id = event.GetEventObject()
        widget_id.SetWindowStyleFlag(wx.BORDER_SUNKEN)
        widget_id.Update()
        
        dlg = wx.ColourDialog(self)

        if dlg.ShowModal() == wx.ID_OK:
            color_data = dlg.GetColourData()
            new_color = color_data.GetColour()
            widget_id.SetBackgroundColour(new_color)
            if widget_id == self.primaryColorButton:
                self.primarycolor = new_color
            elif widget_id == self.secondaryColorButton:
                self.secondarycolor = new_color
            elif widget_id == self.tertiaryColorButton:
                self.tertiarycolor = new_color
            elif widget_id == self.quaternaryColorButton:
                self.quaternarycolor = new_color
            elif widget_id == self.higherColorButton:
                self.highercolor = new_color
            
    def get_color(self):
        color = [self.primarycolor, self.secondarycolor, self.tertiarycolor, self.quaternarycolor, self.highercolor]
        return color
    
    def on_default(self, event):
        self.okButton.Enable()
        self.primarycolor = Default_Primary_Hierarchy_Color
        self.secondarycolor = Default_Secondary_Hierarchy_Color
        self.tertiarycolor = Default_Tertiary_Hierarchy_Color
        self.quaternarycolor = Default_Quaternary_Hierarchy_Color
        self.highercolor = Default_Higher_Hierarchy_Color

        self.primaryColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.primaryColorButton.SetBackgroundColour(self.primarycolor)
        self.primaryColorButton.Update()
        self.secondaryColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.secondaryColorButton.SetBackgroundColour(self.secondarycolor)
        self.secondaryColorButton.Update()
        # noinspection PyUnresolvedReferences
        self.tertiaryColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.tertiaryColorButton.SetBackgroundColour(self.tertiarycolor)
        self.tertiaryColorButton.Update()
        # noinspection PyUnresolvedReferences
        self.quaternaryColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.quaternaryColorButton.SetBackgroundColour(self.quaternarycolor)
        self.quaternaryColorButton.Update()
        self.higherColorButton.SetWindowStyleFlag(wx.NO_BORDER)
        self.higherColorButton.SetBackgroundColour(self.highercolor)
        self.higherColorButton.Update()
    
    def on_ok(self, event):
        self.Close()
    
    def on_cancel(self, event):
        self.Close()
    
        
class ChangeUserSettingDialog(wx.Dialog):
    def __init__(self, parent, original_init, original_min, original_max, original_inc, title):
        super(ChangeUserSettingDialog, self).__init__(parent, title=title,
                                                      style=wx.TAB_TRAVERSAL | wx.FRAME_FLOAT_ON_PARENT | wx.CAPTION)
        
        self.original_init = original_init
        self.original_min = original_min
        self.original_max = original_max
        self.original_inc = original_inc
        
        self.previous_init = original_init
        self.previous_min = original_min
        self.previous_max = original_max
        self.previous_inc = original_inc

        self.init_deci = self.find_decimal_number(original_init)
        self.min_deci = self.find_decimal_number(original_min)
        self.max_deci = self.find_decimal_number(original_max)
        self.inc_deci = self.find_decimal_number(original_inc)

        self.init_value_formatted = ""
        self.min_value_formatted = ""
        self.max_value_formatted = ""
        self.inc_value_formatted = ""

        if self.init_deci == 0:
            label_str_init = "Initial Value:   %.0f  "
            self.init_value_formatted = "%.0f"
        elif self.init_deci == 1:
            label_str_init = "Initial Value:   %.1f  "
            self.init_value_formatted = "%.1f"
        elif self.init_deci == 2:
            label_str_init = "Initial Value:   %.2f  "
            self.init_value_formatted = "%.2f"
        elif self.init_deci == 3:
            label_str_init = "Initial Value:   %.3f  "
            self.init_value_formatted = "%.3f"
        elif self.init_deci == 4:
            label_str_init = "Initial Value:   %.4f  "
            self.init_value_formatted = "%.4f"
        elif self.init_deci == 5:
            label_str_init = "Initial Value:   %.5f  "
            self.init_value_formatted = "%.5"
        else:
            label_str_init = "Initial Value:   %.6f  "
            self.init_value_formatted = "%.6"

        if self.min_deci == 0:
            label_str_min = "Mimimal Value:   %.0f  "
            self.min_value_formatted = "%.0f"
        elif self.min_deci == 1:
            label_str_min = "Mimimal Value:   %.1f  "
            self.min_value_formatted = "%.1f"
        elif self.min_deci == 2:
            label_str_min = "Mimimal Value:   %.2f  "
            self.min_value_formatted = "%.2f"
        elif self.min_deci == 3:
            label_str_min = "Mimimal Value:   %.3f  "
            self.min_value_formatted = "%.3f"
        elif self.min_deci == 4:
            label_str_min = "Mimimal Value:   %.4f  "
            self.min_value_formatted = "%.4f"
        elif self.min_deci == 5:
            label_str_min = "Mimimal Value:   %.5f  "
            self.min_value_formatted = "%.5f"
        else:
            label_str_min = "Mimimal Value:   %.6f  "
            self.min_value_formatted = "%.6f"

        if self.max_deci == 0:
            label_str_max = "Maximal Value:   %.0f  "
            self.max_value_formatted = "%.0f"
        elif self.max_deci == 1:
            label_str_max = "Maximal Value:   %.1f  "
            self.max_value_formatted = "%.1f"
        elif self.max_deci == 2:
            label_str_max = "Maximal Value:   %.2f  "
            self.max_value_formatted = "%.2f"
        elif self.max_deci == 3:
            label_str_max = "Maximal Value:   %.3f  "
            self.max_value_formatted = "%.3f"
        elif self.max_deci == 4:
            label_str_max = "Maximal Value:   %.4f  "
            self.max_value_formatted = "%.4f"
        elif self.max_deci == 5:
            label_str_max = "Maximal Value:   %.5f  "
            self.max_value_formatted = "%.5f"
        else:
            label_str_max = "Maximal Value:   %.6f  "
            self.max_value_formatted = "%.6f"

        if self.inc_deci == 0:
            label_str_inc = "Increment Value:   %.0f  "
            self.inc_value_formatted = "%.0f"
        elif self.inc_deci == 1:
            label_str_inc = "Increment Value:   %.1f  "
            self.inc_value_formatted = "%.1f"
        elif self.inc_deci == 2:
            label_str_inc = "Increment Value:   %.2f  "
            self.inc_value_formatted = "%.2f"
        elif self.inc_deci == 3:
            label_str_inc = "Increment Value:   %.3f  "
            self.inc_value_formatted = "%.3f"
        elif self.inc_deci == 4:
            label_str_inc = "Increment Value:   %.4f  "
            self.inc_value_formatted = "%.4f"
        elif self.inc_deci == 5:
            label_str_inc = "Increment Value:   %.5f  "
            self.inc_value_formatted = "%.5f"
        else:
            label_str_inc = "Increment Value:   %.6f  "
            self.inc_value_formatted = "%.6f"

        self.lblinit = wx.StaticText(self, label=label_str_init % original_init, pos=(20, 20))
        self.init = wx.TextCtrl(self, value=self.init_value_formatted % original_init, pos=(110, 20),
                                size=(60, 20), name="init")
        self.lblmin = wx.StaticText(self, label=label_str_min % original_min, pos=(20, 20))
        self.min = wx.TextCtrl(self, value=self.min_value_formatted % original_min, pos=(110, 20),
                               size=(60, 20), name="min")
        self.lblmax = wx.StaticText(self, label=label_str_max % original_max, pos=(20, 60))
        self.max = wx.TextCtrl(self, value=self.max_value_formatted % original_max, pos=(110, 60),
                               size=(60, 20), name="max")
        self.lblinc = wx.StaticText(self, label=label_str_inc % original_inc, pos=(20, 100))
        # noinspection PyUnresolvedReferences
        self.inc = wx.TextCtrl(self, value=self.inc_value_formatted % original_inc, pos=(110, 100),
                               size=(60, 20), name="inc")
        self.cancelButton = wx.Button(self, wx.ID_ANY, u"CANCEL", wx.DefaultPosition, wx.Size(90, 30), 0)
        self.okButton = wx.Button(self, wx.ID_OK, u"OK", wx.DefaultPosition, wx.Size(90, 30), 0)
        self.okButton.SetBackgroundColour(wx.Colour(225, 225, 225))
        self.okButton.Enable(False)
        self.cancelButton.SetBackgroundColour(wx.Colour(185, 193, 195))
        self.init_ui()
        self.okButton.Bind(wx.EVT_LEFT_DOWN, self.on_ok)
        self.cancelButton.Bind(wx.EVT_LEFT_DOWN, self.on_cancel)
        self.init.Bind(wx.EVT_CHAR, self.test_for_key_being_pressed)
        self.init.Bind(wx.EVT_TEXT, self.on_text_changed)
        self.min.Bind(wx.EVT_CHAR, self.test_for_key_being_pressed)
        self.min.Bind(wx.EVT_TEXT, self.on_text_changed)
        self.max.Bind(wx.EVT_CHAR, self.test_for_key_being_pressed)
        self.max.Bind(wx.EVT_TEXT, self.on_text_changed)
        self.inc.Bind(wx.EVT_CHAR, self.test_for_key_being_pressed)
        # noinspection PyUnresolvedReferences
        self.inc.Bind(wx.EVT_TEXT, self.on_text_changed)

    def init_ui(self):
        sizer0 = wx.BoxSizer(wx.HORIZONTAL)
        sizer0.Add(self.lblinit, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sizer0.Add(self.init, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        # noinspection PyUnresolvedReferences
        sizer1.Add(self.lblmin, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer1.Add(self.min, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        # noinspection PyUnresolvedReferences
        sizer2.Add(self.lblmax, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer2.Add(self.max, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(self.lblinc, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer3.Add(self.inc, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer4.AddSpacer(40)
        sizer4.Add(self.okButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sizer4.Add(self.cancelButton, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        mainsizer = wx.BoxSizer(wx.VERTICAL)
        mainsizer.Add(sizer0, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        mainsizer.Add(sizer1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        mainsizer.Add(sizer2, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        mainsizer.Add(sizer3, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        mainsizer.Add(sizer4, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        self.SetSizerAndFit(mainsizer)
        self.Center()
        self.Show()

    @staticmethod
    def test_for_key_being_pressed(event):
        # Get the keycode pressed
        keycode = event.GetKeyCode()

        # Get rid of the key pressed if it is not 0 to 9
        if ASCII_0 <= keycode <= ASCII_9:
            event.Skip()
        # 8 is the backspace key
        elif keycode == ASCII_BACKSPACE:
            event.Skip()
        elif keycode == ASCII_ENTER:
            event.Skip()
        elif keycode == ASCII_DELETE:
            event.Skip()
        elif keycode == ASCII_INSERT:
            event.Skip()
        # test for the home, end, and arrow keys
        elif ASCII_HOME <= keycode <= ASCII_ARROW_DOWN:
            event.Skip()
        # test for pageup and pagedown keys
        elif ASCII_PAGEUP <= keycode <= ASCII_PAGEDOWN:
            event.Skip()
        elif keycode == ASCII_PERIOD:
            event.Skip()
        elif keycode == ASCII_MINUS:
            event.Skip()

        else:
            # Play a sound to indicate wrong button was pressed and eat up the key pressed by not giving event.Skip()
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            
    def on_text_changed(self, event):
        original_value = 0
        widget = None
        widget_id = event.GetEventObject()
        widget_name = widget_id.GetName()

        if widget_name == "init":
            widget = self.init
            original_value = self.original_init
        if widget_name == "min":
            widget = self.min
            original_value = self.original_min
        elif widget_name == "max":
            widget = self.max
            original_value = self.original_max
        elif widget_name == "inc":
            widget = self.inc
            original_value = self.original_inc

        entered_str = widget.GetValue()
        try:
            entered_value = float(entered_str)
        except:
            widget.ChangeValue(repr(original_value))
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            return

        init_value = float(self.init.GetValue())
        min_value = float(self.min.GetValue())
        max_value = float(self.max.GetValue())
        # inc_value = float(self.inc.GetValue())

        if widget_name == "init":
            _value = float(self.init.GetValue())
            if min_value > _value or _value > max_value:
                self.init.ChangeValue(self.init_value_formatted % self.previous_init)
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                return
            else:
                self.previous_init = _value
        elif widget_name == "min":
            _value = float(self.min.GetValue())
            if init_value < _value or _value > max_value:
                self.min.ChangeValue(self.min_value_formatted % self.previous_min)
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                return
            else:
                self.previous_min = _value
        elif widget_name == "max":
            _value = float(self.max.GetValue())
            if init_value > _value or _value < min_value:
                self.max.ChangeValue(self.max_value_formatted % self.previous_max)
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                return
            else:
                self.previous_max = _value
        elif widget_name == "inc":
            _value = float(self.inc.GetValue())
            if _value > max_value - min_value or _value < self.original_inc:
                self.inc.ChangeValue(self.inc_value_formatted % self.previous_inc)
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                return
            else:
                self.previous_inc = _value

        if entered_value is not None:
            if not entered_value == original_value:
                self.okButton.Enable()
                self.okButton.SetBackgroundColour(wx.Colour(185, 193, 195))

    @staticmethod
    def find_decimal_number(num_float):
        str_float = str(num_float)
        index = str_float.find('.')
        if index == -1:
            return 0
        # elif index == len(str_float) - 1:
        #     return 0
        else:
            return len(str_float) - index - 1

    def get_init(self):
        str_ = self.init.GetValue()
        return round(float(str_), self.init_deci)

    def get_min(self):
        str_ = self.min.GetValue()
        return round(float(str_), self.min_deci)

    def get_max(self):
        str_ = self.max.GetValue()
        return round(float(str_), self.max_deci)

    def get_inc(self):
        str_ = self.inc.GetValue()
        return round(float(str_), self.inc_deci)

    # noinspection PyUnusedLocal
    def on_ok(self, event):
        self.Close()

    # noinspection PyUnusedLocal
    def on_cancel(self, event):
        self.Close()

    # def SaveConnString(self, event):
    #     str_ = self.min.GetValue()
    #     self.min_value = round(float(str_), self.min_deci)
    #     str_ = self.max.GetValue()
    #     self.max_value = round(float(str_), self.max_deci)
    #     str_ = self.inc.GetValue()
    #     self.max_value = round(float(str_), self.inc_deci)


# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
class SettingsDialog(wx.Dialog):
    def __init__(self, parent):
        # Setting up a frame for the app
        # wx.Dialog.__init__(self, parent, id=wx.ID_ANY, pos=(200, 200),
        # 				  size=wx.Size(380, 300),
        # 				  style= wx.CAPTION | wx.TAB_TRAVERSAL | wx.STAY_ON_TOP)
        # noinspection PyUnresolvedReferences
        super(SettingsDialog, self).__init__(parent, title="Settings",
                                             style=wx.TAB_TRAVERSAL | wx.FRAME_FLOAT_ON_PARENT | wx.CAPTION)
        # wx.Frame.__init__(self, parent)

        # This is the data set the settings dialog is holding and passing back to the main program

        self.setting_dict = {"working dir": u"c:\\Users\\Chester\\AccuracyTesting",
                             "input dir": u"c:\\Users\\Chester\\PictureEntry\\Baby Willa Lau 6.15 to 8.15",
                             "tensorflow dir": u"c:\\Users\\Chester\\BabyPictureTrainLog",
                             "web name1": "Google",
                             "website1": "http://www.google.com",
                             "web name2": "DropBox",
                             "website2": "",
                             "web name3": "No Website Assigned",
                             "website3": "",
                             "save after number calculations": 40,
                             "larger screen": False,
                             "hover notes": True,
                             "baby1": "Baby1",
                             "baby2": "Baby2",
                             "baby3": "Baby3",
                             "unknown baby": "Unknown baby",
                             "dad": "Dad",
                             "mom": "Mom",
                             "fracturnal grandpa": "Fracturnal grandpa",
                             "fracturnal grandmom": "Fracturnal grandmom",
                             "maturnal grandpa": "Maturnal grandpa",
                             "maturnal grandmom": "Maturnal grandmom",
                             "baby sitter": "Baby sitter",
                             "unknown adult": "Unknown adult",
                             "no human face": "No human face"}

        self.user_parameter_dict = {}

        # Get all the parameters that was saved on hard drive in the working directory
        if self.load_parameters() == "error":
            return

        # Define all the widgets of setting dialog
        self.cancelButton = wx.Button(self, wx.ID_ANY, u"CANCEL", wx.DefaultPosition, wx.Size(90, 30), 0)
        self.okButton = wx.Button(self, wx.ID_ANY, u"OK", wx.DefaultPosition, wx.Size(90, 30), 0)
        # noinspection PyUnresolvedReferences
        self.checkbox2 = wx.CheckBox(self, wx.ID_ANY, u"Hover Notes",
                                     name="Hover note checkbox")
        self.checkbox1 = wx.CheckBox(self, wx.ID_ANY, u"Larger Screen",
                                     name="Larger screen checkbox")
        # noinspection PyUnresolvedReferences
        self.m_textCtrl6 = wx.TextCtrl(self, wx.ID_ANY, self.setting_dict["website3"],
                                       wx.DefaultPosition, (140, 20), 0, name="default website3")
        self.m_staticText6 = wx.StaticText(self, wx.ID_ANY, u"Website 3",
                                           wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT, name="label webname3")
        self.m_textCtrl5 = wx.TextCtrl(self, wx.ID_ANY, self.setting_dict["web name3"],
                                       wx.DefaultPosition, (125, 20), 0, name="default webname3")
        self.m_staticText5 = wx.StaticText(self, wx.ID_ANY, u"Web Name 3",
                                           wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT, name="label webname2")
        # noinspection PyUnresolvedReferences
        self.m_textCtrl4 = wx.TextCtrl(self, wx.ID_ANY, self.setting_dict["website2"],
                                       wx.DefaultPosition, (140, 20), 0, name="default website2")
        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        self.m_staticText4 = wx.StaticText(self, wx.ID_ANY, u"Website 2:",
                                           wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT, name="label website2")
        # noinspection PyUnresolvedReferences
        self.m_textCtrl3 = wx.TextCtrl(self, wx.ID_ANY, self.setting_dict["web name2"],
                                       wx.DefaultPosition, (125, 20), 0, name="default webname2")
        self.m_staticText3 = wx.StaticText(self, wx.ID_ANY, u"Web Name 2",
                                           wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT, name="label webname2")
        self.m_textCtrl2 = wx.TextCtrl(self, wx.ID_ANY, self.setting_dict["website1"],
                                       wx.DefaultPosition, (140, 20), 0, name="default website1")
        # noinspection PyUnresolvedReferences
        self.m_staticText2 = wx.StaticText(self, wx.ID_ANY, u"Website 1:",
                                           wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT, name="label website1")
        self.m_textCtrl1 = wx.TextCtrl(self, wx.ID_ANY, self.setting_dict["web name1"],
                                       wx.DefaultPosition, (125, 20), 0, name="default webname1")
        # noinspection PyUnresolvedReferences
        self.m_staticText1 = wx.StaticText(self, wx.ID_ANY, u"Web Name 1",
                                           wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT, name="label webname1")
        # noinspection PyUnresolvedReferences
        self.m_staticText7 = wx.StaticText(self, wx.ID_ANY, u"Save every",
                                           wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT, name="label save every")
        self.m_staticText8 = wx.StaticText(self, wx.ID_ANY, u"calculations",
                                           wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT, name="label calculation")
        self.m_number_calculations = wx.SpinCtrl(self, wx.ID_ANY, min=20, max=100, initial=40, size=(50, 20),
                                                 name="spin control")
        # noinspection PyUnresolvedReferences
        self.dirPicker2 = wx.DirPickerCtrl(self, wx.ID_ANY, self.setting_dict["tensorflow dir"],
                                           u"", wx.DefaultPosition, wx.DefaultSize,
                                           wx.DIRP_DIR_MUST_EXIST | wx.DIRP_SMALL | wx.DIRP_USE_TEXTCTRL,
                                           name="default tensorflow dir")
        self.dirPicker1 = wx.DirPickerCtrl(self, wx.ID_ANY, self.setting_dict["input dir"],
                                           u"", wx.DefaultPosition, wx.DefaultSize,
                                           wx.DIRP_DIR_MUST_EXIST | wx.DIRP_SMALL | wx.DIRP_USE_TEXTCTRL,
                                           name="default input dir")
        self.m_dirPicker3 = wx.DirPickerCtrl(self, wx.ID_ANY, self.setting_dict["working dir"], u"Select A Folder",
                                             wx.DefaultPosition, wx.DefaultSize,
                                             wx.DIRP_DIR_MUST_EXIST | wx.DIRP_SMALL | wx.DIRP_USE_TEXTCTRL,
                                             name="default working dir")

        # The self.evt is an event ID generated by DialogClosedEvent method
        # that was provided at the beginning of the declaration of this program
        self.evt = DialogClosedEvent()

        # Save the parent ID for later use. parent is the self.panel
        self.parent = parent

        # Set all the flags from the hover note for the settings.
        # Don't confuse this self.hoverNotesOn flag in this SettingDialog class
        # with the self.hoverNotesOn flag in the WorkingFrame class
        self.hoverNotesOn = self.setting_dict["hover notes"]

        # Main widget layout and binding go into here
        self.init_ui()

    def init_ui(self):
        # Place the working directory frame
        vsbsizer = wx.BoxSizer(wx.VERTICAL)
        sbsizer4 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"Working Directory"), wx.VERTICAL)
        # Create the working directory dirPickerCtrl
        sbsizer4.Add(self.m_dirPicker3, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        vsbsizer.Add(sbsizer4, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        # Place the default input directory frame
        sbsizer1 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"Default input Directory"), wx.VERTICAL)
        # Create the default input directory dirPickerCtrl
        sbsizer1.Add(self.dirPicker1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        vsbsizer.Add(sbsizer1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        # Place the default output directory frame
        sbsizer2 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"Default TensorFlow Data Directory"), wx.VERTICAL)
        # Create the default output directory dirPickerCtrl
        sbsizer2.Add(self.dirPicker2, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        vsbsizer.Add(sbsizer2, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        # Place the startup URL frame
        sbsizer3 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"Startup URL"), wx.VERTICAL)
        # Create the website1 name static text box
        # Create the text entry control for the website1 name
        # Set the maximum length for the website1 name
        self.m_textCtrl1.SetMaxLength(25)
        sbsizer5 = wx.BoxSizer(wx.HORIZONTAL)
        # noinspection PyUnresolvedReferences
        sbsizer5.Add(self.m_staticText1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sbsizer5.Add(self.m_textCtrl1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # Create the website1 label static text box
        # Create the text entry control for the website1 label
        # Set the maximum length for the website1 label
        self.m_textCtrl2.SetMaxLength(100)
        sbsizer6 = wx.BoxSizer(wx.HORIZONTAL)
        sbsizer6.Add(self.m_staticText2, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        sbsizer6.Add(self.m_textCtrl2, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        # Create the website2 name static text box
        # Create the text entry control for the website2 name
        # Set the maximum length for the website2 name
        self.m_textCtrl3.SetMaxLength(25)
        sbsizer7 = wx.BoxSizer(wx.HORIZONTAL)
        # noinspection PyUnresolvedReferences
        sbsizer7.Add(self.m_staticText3, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sbsizer7.Add(self.m_textCtrl3, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # Create the website2 label static text box
        # Create the text entry control for the website2 label
        # Set the maximum length for the website2 label
        self.m_textCtrl4.SetMaxLength(100)
        sbsizer8 = wx.BoxSizer(wx.HORIZONTAL)
        sbsizer8.Add(self.m_staticText4, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sbsizer8.Add(self.m_textCtrl4, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        # Create the website3 name static text box
        # Create the text entry control for the website3 name
        # Set the maximum length for the website3 name
        self.m_textCtrl5.SetMaxLength(25)
        sbsizer9 = wx.BoxSizer(wx.HORIZONTAL)
        # noinspection PyUnresolvedReferences
        sbsizer9.Add(self.m_staticText5, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sbsizer9.Add(self.m_textCtrl5, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # Create the website3 label static text box
        # Create the text entry control for the website3 label
        # Set the maximum length for the website3 label
        self.m_textCtrl6.SetMaxLength(100)
        sbsizer10 = wx.BoxSizer(wx.HORIZONTAL)
        sbsizer10.Add(self.m_staticText6, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sbsizer10.Add(self.m_textCtrl6, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        sbsizer3.Add(sbsizer5, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sbsizer3.Add(sbsizer6, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sbsizer3.Add(sbsizer7, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        sbsizer3.Add(sbsizer8, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        sbsizer3.Add(sbsizer9, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sbsizer3.Add(sbsizer10, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        vsbsizer.Add(sbsizer3, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        # Set up the save calculation spin control
        # noinspection PyUnresolvedReferences
        sbsizer12 = wx.BoxSizer(wx.HORIZONTAL)
        sbsizer12.Add(self.m_staticText7, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sbsizer12.Add(self.m_number_calculations, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        sbsizer12.Add(self.m_staticText8, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        # noinspection PyUnresolvedReferences
        vsbsizer.Add(sbsizer12, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        # Setting up all the checkboxes
        self.checkbox1.SetValue(self.setting_dict["larger screen"])
        vsbsizer.Add(self.checkbox1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        self.checkbox2.SetValue(self.setting_dict["hover notes"])
        # vsbsizer.Add(self.checkbox1, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)
        vsbsizer.Add(self.checkbox2, 0, wx.ALL | wx.ALIGN_RIGHT | wx.EXPAND, 5)

        # Setting up the OK and CANCEL buttons
        # noinspection PyUnresolvedReferences
        sbsizer11 = wx.BoxSizer(wx.HORIZONTAL)
        self.okButton.SetBackgroundColour(wx.Colour(10, 224, 10))
        self.cancelButton.SetBackgroundColour(wx.Colour(220, 10, 10))

        sbsizer11.AddSpacer(10)
        # noinspection PyUnresolvedReferences
        sbsizer11.Add(self.okButton, 0, wx.ALL, 5)
        sbsizer11.AddSpacer(15)
        sbsizer11.Add(self.cancelButton, 0, wx.ALL,  5)
        # ssbsizer11.AddSpacer(1)

        vsbsizer.Add(sbsizer11, 0, wx.ALL | wx.EXPAND, 5)

        self.SetSizerAndFit(vsbsizer)
        self.Center()

        # Binding all the checkboxes and the buttons
        self.checkbox1.Bind(wx.EVT_LEFT_DOWN, self.on_checkbox1_left_down)
        self.checkbox2.Bind(wx.EVT_LEFT_DOWN, self.on_checkbox2_left_down)
        self.okButton.Bind(wx.EVT_LEFT_DOWN, self.on_ok)
        self.cancelButton.Bind(wx.EVT_LEFT_DOWN, self.on_cancel)

    def set_value(self, label, value):
        # Simply setting the value into the dictionary
        self.setting_dict[label] = value

    def load_parameters(self):
        # Check for existence of working directory and file
        if not os.path.exists(self.setting_dict["working dir"]):
            os.mkdir(self.setting_dict["working dir"])
            _filename = os.path.join(self.setting_dict["working dir"], "data.txt")
            # Try to open data.txt. Create a new one if not existed
            _file = open(_filename, 'w+')
            json.dump(self.setting_dict, _file)

            # Close the data.txt for now
            _file.close()

        else:
            _filename = os.path.join(self.setting_dict["working dir"], "data.txt")
            if os.path.exists(_filename):
                # read in setting_data
                try:
                    _file = open(_filename, 'r')
                    # self.setting_dict.clear()
                    try:
                        self.setting_dict = json.load(_file)
                    except:
                        # json file is empty or corrupted. Replace it with the original template
                        _file.close()
                        _file = open(_filename, 'w+')
                        json.dump(self.setting_dict, _file)
                    _file.close()
                except IOError:
                    print("cannot open and read ", _filename)
                    return "error"
            else:
                # Create a new one if not existed
                _file = open(_filename, 'w+')
                # Close the data.txt for now
                _file.close()

        return ""

    def save_parameters(self):
        # Save in label list
        _filename = ""
        try:
            _filename = os.path.join(self.setting_dict["working dir"], "data.txt")
            _file = open(_filename, 'w')
            json.dump(self.setting_dict, _file)
            _file.close()
        except IOError:
            print("cannot open and write ", _filename)
            return "error"
        try:
            _filename = os.path.join(self.setting_dict["working dir"], "parameter_data.txt")
            _file = open(_filename, 'w')
            json.dump(self.user_parameter_dict, _file)
            _file.close()
        except IOError:
            print("cannot open and write ", _filename)
            return "error"
        return ""

    def on_checkbox1_left_down(self, event):
        # Setting the checkbox1 according to the user's desire
        if self.checkbox1.IsChecked():
            self.checkbox1.SetValue(False)
        else:
            self.checkbox1.SetValue(True)

    def on_checkbox2_left_down(self, event):
        # Setting the checkbox2 according to the user's desire
        if self.checkbox2.IsChecked():
            self.checkbox2.SetValue(False)
        else:
            self.checkbox2.SetValue(True)

    def on_ok(self, event):
        # Reset all the values from the user's desire to the settings dictionary
        del self.setting_dict["working dir"]
        self.setting_dict["working dir"] = self.m_dirPicker3.GetPath()
        del self.setting_dict["input dir"]
        self.setting_dict["input dir"] = self.dirPicker1.GetPath()
        del self.setting_dict["tensorflow dir"]
        self.setting_dict["tensorflow dir"] = self.dirPicker2.GetPath()
        del self.setting_dict["web name1"]
        self.setting_dict["web name1"] = self.m_textCtrl1.GetLineText(0)
        del self.setting_dict["website1"]
        self.setting_dict["website1"] = self.m_textCtrl2.GetLineText(0)
        del self.setting_dict["web name2"]
        self.setting_dict["web name2"] = self.m_textCtrl3.GetLineText(0)
        del self.setting_dict["website2"]
        self.setting_dict["website2"] = self.m_textCtrl4.GetLineText(0)
        del self.setting_dict["web name3"]
        self.setting_dict["web name3"] = self.m_textCtrl5.GetLineText(0)
        del self.setting_dict["website3"]
        self.setting_dict["website3"] = self.m_textCtrl6.GetLineText(0)
        del self.setting_dict["save after number calculations"]
        self.setting_dict["save after number calculations"] = self.m_number_calculations.GetValue()
        del self.setting_dict["larger screen"]
        self.setting_dict["larger screen"] = self.checkbox1.IsChecked()
        del self.setting_dict["hover notes"]
        self.setting_dict["hover notes"] = self.checkbox2.IsChecked()

        if self.save_parameters() == "error":
            return

        # This is the place to send the event created by DialogClosedEvent() method to self.panel of WorkingFrame class
        #  with the EVT_CUSTOM_DIALOG_CLOSED message
        wx.PostEvent(self.parent, self.evt)

        # Close the settings dialog
        self.Close()

    def on_cancel(self, event):
        # This is another place to send the event created by DialogClosedEvent() method
        # to self.panel of WorkingFrame class
        #  with the EVT_CUSTOM_DIALOG_CLOSED message
        wx.PostEvent(self.parent, self.evt)

        # Close the settings dialog
        self.Close()
    
       
class PlotResult(scrolled.ScrolledPanel):
    # def __init__(self, parent, num_steps_per_epoch, num_steps_per_report):
    def __init__(self, parent):
        super(PlotResult, self).__init__(parent)
        
        # self.subplot_list = [None, None, None, None, None, None, None]
        # self.num_epoch = -1
        # self.plot_tensor = None
        # self.name_list = []
        # self.num_steps_per_epoch = num_steps_per_epoch
        
        self.parent = parent
        self.plot_tensor = None
        self.name_list = []
        self.choice_string_list = []
        self.steps_per_report = 0
        self.num_epoch = self.num_graph = self.num_data = 0
        
        self.scale = "linear"

        self.init_ui()
    
    def init_ui(self):
        file_list = os.listdir(WORKING_DIR)
        refined_stat_file_list = [x for x in file_list if x.startswith("accuracy_data")]
        dlg = wx.SingleChoiceDialog(self.parent, "", "Select Data File To Be Plotted", refined_stat_file_list)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        
        np_data_file = dlg.GetStringSelection()
        full_np_data_file = os.path.join(WORKING_DIR, np_data_file)
        
        try:
            self.plot_tensor = np.load(full_np_data_file)
        except:
            wx.MessageDialog(self, "Cannot read %s\n\n" % np_data_file,
                             "Data File Cannot Be Read", wx.ICON_EXCLAMATION | wx.OK).ShowModal()
            exit(500)

        (self.num_epoch, self.num_graph, self.num_data) = np.shape(self.plot_tensor)

        time_str = np_data_file[13: -4]
        name_list_file = "stat_name_list" + time_str + ".txt"
        try:
            # Try to open data.txt. Create a new one if not existed
            full_name_list_file = os.path.join(WORKING_DIR, name_list_file)
            stat_name_file = open(full_name_list_file, 'r')
            self.name_list = eval(stat_name_file.readline())
            stat_name_file.close()
            # if not name_list:
            #     name_list = [LOSS, IMAGE_ACCURACY, POSITIVE_FACE_ACCURACY, NEGATIVE_FACE_ACCURACY,
            #                       TOTAL_FACE_ACCURACY, TENSORFLOW_IMAGE_ACCURACY, TENSORFLOW_FACE_ACCURACY]
        except:
            wx.MessageDialog(self, "Cannot read %s\n\n" % name_list_file,
                             "Data File Cannot Be Read", wx.ICON_EXCLAMATION | wx.OK).ShowModal()
            exit(500)
            
        self.choice_string_list = self.convert_name_list_back_string_list()
        
        steps_per_report_file = "steps_per_report" + time_str + ".txt"
        try:
            # Try to open data.txt. Create a new one if not existed
            full_steps_per_report_file = os.path.join(WORKING_DIR, steps_per_report_file)
            steps_file = open(full_steps_per_report_file, 'r')
            self.steps_per_report = eval(steps_file.readline())
            steps_file.close()
            # if not name_list:
            #     name_list = [LOSS, IMAGE_ACCURACY, POSITIVE_FACE_ACCURACY, NEGATIVE_FACE_ACCURACY,
            #                       TOTAL_FACE_ACCURACY, TENSORFLOW_IMAGE_ACCURACY, TENSORFLOW_FACE_ACCURACY]
        except:
            wx.MessageDialog(self, "Cannot read %s\n\n" % name_list_file,
                             "Data File Cannot Be Read", wx.ICON_EXCLAMATION | wx.OK).ShowModal()
            exit(500)
            
        self.plot_single_data()
       
    def convert_name_list_back_string_list(self):
        choice_list = []
        for item in self.name_list:
            if item == LOSS:
                choice_list.append("Loss")
            elif item == IMAGE_ACCURACY:
                choice_list.append("Image Accuracy")
            elif item == POSITIVE_FACE_ACCURACY:
                choice_list.append("Positive Face Accuracy")
            elif item == NEGATIVE_FACE_ACCURACY:
                choice_list.append("Negative Face Accuracy")
            elif item == TOTAL_FACE_ACCURACY:
                choice_list.append("Total Face Accuracy")
            elif item == TENSORFLOW_FACE_ACCURACY:
                choice_list.append("Tensorflow Face Accuracy")
            elif item == SEN1:
                choice_list.append("Sensitivity Baby 1")
            elif item == SEN2:
                choice_list.append("Sensitivity Baby 2")
            elif item == SEN3:
                choice_list.append("Sensitivity Baby 3")
            elif item == SEN4:
                choice_list.append("Sensitivity Baby 4")
            elif item == SEN5:
                choice_list.append("Sensitivity Dad")
            elif item == SEN6:
                choice_list.append("Sensitivity Mom")
            elif item == SEN7:
                choice_list.append("Sensitivity Fraternal GrandPa")
            elif item == SEN8:
                choice_list.append("Sensitivity Fraternal Grandom")
            elif item == SEN9:
                choice_list.append("Sensitivity Maternal GrandPa")
            elif item == SEN10:
                choice_list.append("Sensitivity Maternal Grandom")
            elif item == SEN11:
                choice_list.append("Sensitivity Baby Sitter")
            elif item == SEN12:
                choice_list.append("Sensitivity Unknow Adult")
            elif item == SP1:
                choice_list.append("Specificity Baby 1")
            elif item == SP2:
                choice_list.append("Specificity Baby 2")
            elif item == SP3:
                choice_list.append("Specificity Baby 3")
            elif item == SP4:
                choice_list.append("Specificity Baby 4")
            elif item == SP5:
                choice_list.append("Specificity Dad")
            elif item == SP6:
                choice_list.append("Specificity Mom")
            elif item == SP7:
                choice_list.append("Specificity Fraternal GrandPa")
            elif item == SP8:
                choice_list.append("Specificity Fraternal Grandom")
            elif item == SP9:
                choice_list.append("Specificity Maternal GrandPa")
            elif item == SP10:
                choice_list.append("Specificity Maternal Grandom")
            elif item == SP11:
                choice_list.append("Specificity Baby Sitter")
            elif item == SP12:
                choice_list.append("Specificity Unknow Adult")
        return choice_list
           
    def plot_single_data(self):
        dlg = wx.SingleChoiceDialog(self.parent, "Select one item", "Data to be Plotted", self.choice_string_list)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        data_type_selection = dlg.GetSelection()
        title = dlg.GetStringSelection()
        np_x, np_y, np_z = self.get_data(data_type_selection)
        graph_type_selection = self.pick_graphtype()
        if graph_type_selection == THREE_D_WIRE_FRAME:
            self.plot_3dwireframe(np_x, np_y, np_z, title)
        elif graph_type_selection == THREE_D_SURFACE:
            self.plot_3dsurface(np_x, np_y, np_z, title)
        elif graph_type_selection == THREE_D_CONTOUR_PLOT:
            self.plot_3dcontour(np_x, np_y, np_z, title)
        elif graph_type_selection == THREE_D_WIRE_GRAPH_BY_EPOCH_NUMBER:
            self.plot_3dwire_epoch_number(np_x, np_y, np_z, title)
        elif graph_type_selection == THREE_D_WIRE_GRAPH_BY_STEPS:
            self.plot_3dwire_steps(np_x, np_y, np_z, title)
        elif graph_type_selection == THREE_D_BAR:
            self.plot_3dbar(np_z, title)
        elif graph_type_selection == THREE_D_POLYGON:
            self.plot_3dpolygon(np_z, title)
        # elif graph_type_selection == TWO_D_BAR:
        #     self.plot_2dbar(np_x, np_y, np_z, title)
        elif graph_type_selection == TWO_D_LINES:
            self.plot_2dlines(np_z, title)
        elif graph_type_selection == TWO_D_LOG:
            self.plot_2dlog(np_z, title)
        elif graph_type_selection == TWO_D_BROKEN_Y_AXIS:
            self.plot_2dbrokenaxis(np_z, title)
   
    def plot_2comparision_data(self):
        dlg = wx.MultiChoiceDialog(self.parent, "Select two items", "Data to be Plotted", 2, self.name_list)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        
    def plot_3comparision_data(self):
        dlg = wx.MultiChoiceDialog(self.parent, "Select three items", "Data to be Plotted", 3, self.name_list)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        
    def plot_4comparision_data(self):
        dlg = wx.MultiChoiceDialog(self.parent, "Select four items", "Data to be Plotted", 4, self.name_list)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
    
    def pick_graphtype(self):
        graph_type_list = ["3D wire_frame", "3D surface", "3D contour plot", "3D wire graph by epoch number",
                           "3D wire graph by steps", "3D bar", "3D polygon", "2D lines", "2D log", "2D broken y-axis"]
        dlg = wx.SingleChoiceDialog(self.parent, "Select one graph type", "Graph type", graph_type_list)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return 0
        else:
            return dlg.GetSelection()

        # THREE_D_WIRE_FRAME = 0
        # THREE_D_SURFACE = 1
        # THREE_D_CONTOUR_PLOT = 2
        # THREE_D_WIRE_GRAPH_BY_EPOCH_NUMBER = 3
        # THREE_D_WIRE_GRAPH_BY_STEPS = 4
        # THREE_D_BAR = 5
        # THREE_D_POLYGON = 6
        # TWO_D_BAR = 7
        # TWO_D_LINES = 8
        # TWO_D_CUMULATIVE = 9
        # TWO_D_LOG = 10
        # TWO_D_LOGIT = 11
        # TWO_D_BROKEN_Y_AXIS = 12
        
    def get_data(self, selection):
        (num_epoch, num_graph, num_data) = np.shape(self.plot_tensor)
        np_x = np.arange(0, num_data, 1.0)
        np_y = np.arange(0, num_epoch, 1.0)
        # plt.grid(True)
        # plt.style.use('ggplot')
        np_z = np.zeros((1, num_data), np.float32)
        for z in range(num_epoch):
            np_temp_z = self.plot_tensor[z, selection: selection + 1, :]
            np_z = np.concatenate((np_z, np_temp_z), axis=0)
        np_z = np.delete(np_z, 0, axis=0)
        
        return np_x, np_y, np_z
    
    def plot_3dwireframe(self, x, y, np_z, title):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=.06, right=.90, bottom=.06, top=.98)
        np_x, np_y = np.meshgrid(x, y)
        ax.plot_wireframe(np_x, np_y, np_z, rstride=1, cstride=1)
        self.set_labels(title, np_z, ax, "3d")
        
        plt.show()
    
    def plot_3dsurface(self, x, y, np_z, title):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Make data.
        np_x, np_y = np.meshgrid(x, y)
        # Plot the surface.
        surf = ax.plot_surface(np_x, np_y, np_z, cmap='coolwarm', linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        self.set_labels(title, np_z, ax, "3d")
        
        plt.show()

    def plot_3dcontour(self, x, y, np_z, title):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax = fig.add_subplot(111, projection='3d')
        np_x, np_y = np.meshgrid(x, y)
        ax.plot_surface(np_x, np_y, np_z, rstride=8, cstride=8, alpha=0.4)
        ax.contour(np_x, np_y, np_z, zdir='z', offset=-0, cmap='coolwarm')
        ax.contour(np_x, np_y, np_z, zdir='y', offset=-self.num_epoch * -1, cmap='coolwarm')
        ax.contour(np_x, np_y, np_z, zdir='x', offset=0, cmap='coolwarm')
        self.set_labels(title, np_z, ax, "3d")
        
        plt.show()
    
    def plot_3dwire_epoch_number(self, x, y, np_z, title):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax = fig.add_subplot(111, projection='3d')
        np_x, np_y = np.meshgrid(x, y)
        ax.plot_wireframe(np_x, np_y, np_z, rstride=1, cstride=0)
        self.set_labels(title, np_z, ax, "3d")
        
        plt.show()
    
    def plot_3dwire_steps(self, x, y, np_z, title):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax = fig.add_subplot(111, projection='3d')
        np_x, np_y = np.meshgrid(x, y)
        ax.plot_wireframe(np_x, np_y, np_z, rstride=0, cstride=1)
        self.set_labels(title, np_z, ax, "3d")
       
        plt.show()
    
    def plot_3dbar(self, np_z, title):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colors_list = [cm.jet(w) for w in np.random.rand(self.num_epoch)]
        xs = np.arange(0, self.num_data, 1.0)
        yticks = [w for w in range(self.num_epoch)]
        
        for c, k in zip(colors_list, yticks):
            ys = np_z[k]
            ax.bar(xs, ys, zs=k, zdir='y', color=colors_list[k], alpha=0.8)
        self.set_labels(title, np_z, ax, "3d")
        
        plt.show()
    
    def plot_3dpolygon(self, np_z, title):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #
        # def cc(arg):
        #     return mcolors.to_rgba(arg, alpha=0.7)
        #
        xs = np.arange(0, self.num_data, 1.0)
        verts = []
        zs = [float(z) for z in range(self.num_epoch)]
        for z in zs:
            ys = np_z[int(z)]
            ys[0], ys[-1] = 0, 0
            verts.append(list(zip(xs, ys)))
        facecolors_list = [cm.jet(w) for w in np.random.rand(self.num_epoch)]
        poly = PolyCollection(verts, facecolors=facecolors_list)
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs, zdir='y')
        self.set_labels(title, np_z, ax, "3d")
        
        plt.show()
    
    # def plot_2dbar(self, x, y, np_z, title):
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    #     fig.subplots_adjust(left=.06, right=.90, bottom=.06, top=.98)
    #
    #     colors_list = [cm.jet(w) for w in np.random.rand(self.num_epoch)]
    #     epoch_list = ["epoch " + repr(w + 1) for w in range(self.num_epoch)]
    #     for s in range(self.num_data):
    #         for w in range(self.num_epoch):
    #             ax.bar(s, np_z[w], 0.4, color=colors_list[w], orientation='vertical', label=epoch_list[w])
    #     self.set_labels(title, np_z, ax, "2d")
    #     plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., shadow=True, fontsize='small')
    #     plt.show()
    
    def plot_2dlines(self, np_z, title):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        fig.subplots_adjust(left=.06, right=.90, bottom=.06, top=.98)
        colors_list = [cm.jet(w) for w in np.random.rand(self.num_epoch)]
        epoch_list = ["epoch " + repr(w + 1) for w in range(self.num_epoch)]
        for w in range(self.num_epoch):
            plt.plot(np_z[w], lw=2.5, color=colors_list[w], label=epoch_list[w])
        self.scale = "linear"
        self.set_labels(title, np_z, ax, "2d")
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., shadow=True, fontsize='small')
        plt.show()
    
    # def plot_2dcumulative(self, x, y, z):
    #     pass
        
    def plot_2dlog(self, np_z, title):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        fig.subplots_adjust(left=.06, right=.90, bottom=.06, top=.98)
        colors_list = [cm.jet(w) for w in np.random.rand(self.num_epoch)]
        epoch_list = ["epoch " + repr(w + 1) for w in range(self.num_epoch)]
        for w in range(self.num_epoch):
            plt.plot(np_z[w], lw=2.5, color=colors_list[w], label=epoch_list[w])
        self.scale = "log"
        self.set_labels(title, np_z, ax, "2d")
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., shadow=True, fontsize='small')
        plt.show()
    
    # def plot_2dlogit(self, x, y, np_z, title):
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    #     fig.subplots_adjust(left=.06, right=.90, bottom=.06, top=.98)
    #     colors_list = [cm.jet(w) for w in np.random.rand(self.num_epoch)]
    #     epoch_list = ["epoch " + repr(w + 1) for w in range(self.num_epoch)]
    #     for w in range(self.num_epoch):
    #         plt.plot(np_z[w], lw=2.5, color=colors_list[w], label=epoch_list[w])
    #     self.scale = "logit"
    #     self.set_labels(title, np_z, ax, "2d")
    #     plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., shadow=True, fontsize='small')
    #     plt.show()
    
    def plot_2dbrokenaxis(self, np_z, title):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="row", figsize=(10, 9.0))
        epoch_list = ["epoch " + repr(w + 1) for w in range(self.num_epoch)]
        for w in range(self.num_epoch):
            # plt.plot(np_z[w], lw=2.5, color=colors_list[w], label=epoch_list[w])
            ax1.plot(np_z[w])
            ax2.plot(np_z[w], label=epoch_list[w])
        self.scale = "linear"
        self.set_labels(title, np_z, ax2, "2d")
        
        np_max = np.max(np_z)
        np_min = np.min(np_z)
        np_var = np.var(np_z)
        np_mean = np.mean(np_z)
        
        ax1.set_ylim(np_mean + np_var, np_max)
        ax2.set_ylim(np_min, np_mean)
        
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        d = .015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0., shadow=True, fontsize='small')
        plt.show()
    
    def set_labels(self, title, np_z, ax, dimensions):
        mini_value = np_z.min()
        max_value = np_z.max()
        pad_value = (max_value - mini_value) / 10
        max_value += pad_value
        mini_value -= pad_value / 4
        if dimensions == '3d':
            # floor_value = math.floor(mini_value)
            # ceiling_value = math.ceil(max_value)
            # inc_value = 1.0
            # if ceiling_value - floor_value <= 2:
            #     inc_value = 0.25
            plt.xticks(np.arange(0, self.num_data + 1, 5.0), fontsize='x-small')
            plt.yticks(np.arange(0, self.num_epoch + 1, 1.0), fontsize='x-small')
            # plt.zticks(np.arange(floor_value, ceiling_value, inc_value), fontsize='small')
            ax.set_xlabel('Period----Steps / Number of Steps Per Report(%d)' % NUMBER_OF_STEPS_PER_REPORT,
                          fontsize='x-small')
            ax.set_xlim3d(0, self.num_data + 1)
            ax.set_ylabel('Epoch', fontsize='x-small')
            ax.set_ylim3d(0, self.num_epoch + 1)
            ax.set_zlabel(title, fontsize='medium')
            ax.set_zlim3d(mini_value, max_value)
        elif dimensions == '2d':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if self.scale == "linear":
                plt.yscale("linear")
            elif self.scale == "log":
                plt.yscale("log")
                self.scale = "linear"
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            floor_value = math.floor(mini_value)
            ceiling_value = math.ceil(max_value)
            inc_value = 1.0
            if ceiling_value - floor_value <= 2:
                inc_value = 0.25
            plt.yticks(np.arange(floor_value, ceiling_value, inc_value), fontsize='small')
            plt.xticks(np.arange(0, self.num_data + 1, 1.0), fontsize='small')
            ax.set_xlabel('Period----Steps / Number of Steps Per Report(%d)' % NUMBER_OF_STEPS_PER_REPORT)
            ax.set_xlim(0, self.num_data)
            ax.set_ylabel(title)
            ax.set_ylim(mini_value, max_value)
            
           
class WorkingFrame(wx.Frame):
    def __init__(self, parent, title):
        # parent is the main window. It should be None

        # title is the name appearing at the top of the program. It can be any string

        # Setting up a frame for the app

        x, y = wx.DisplaySize()
        # noinspection PyUnresolvedReferences
        super(WorkingFrame, self).__init__(parent, title=title, pos=(0, 0),
                                           size=(x, y - 40), style=wx.DEFAULT_FRAME_STYLE & ~wx.CLOSE_BOX)

        _select_dialog = ParameterPanel(self)
        _select_dialog.SetBackgroundColour(Default_Background_Color)
        # _select_dialog.SetAutoLayout(1)
        # _select_dialog.SetupScrolling()
        self.Show()


# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyAttributeOutsideInit
class SingleApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super(SingleApp, self).__init__(redirect=False, filename=None, useBestVisual=False, clearSigInt=True)

    # noinspection PyAttributeOutsideInit
    def OnInit(self):
        # Find the user id and make a name for the process, allowing one instance per user
        self.name = "SingleApp-%s" % wx.GetUserId()
        # Get the instance handle and check if the instance is running?
        # noinspection PyAttributeOutsideInit
        self.instance = wx.SingleInstanceChecker(self.name)
        if self.instance.IsAnotherRunning():
            _str1 = "A process for the Select Hyperparameter is already running.\n\n"
            _str2 = "Vision Neuronetwork and Tensorflow are CPU and memory intensive processes.\n\n"
            _str3 = "Cannot start a second Select Hyperparameter process."
            _str = _str1 + _str2 + _str3
            # Give a message and exit
            wx.MessageDialog(None, _str, "UNABLE TO START A SECOND PROCESS!", wx.ICON_EXCLAMATION | wx.OK).ShowModal()

            return False
        frame = WorkingFrame(None, "Select Hyperparameters")
        return True


def main():
    # app = wx.App()
    # WorkingFrame(None, "Select Hyperparameters")

    # Allow only one instance of the Select Hyperparameter to be running per user at one time
    app = SingleApp(redirect=False)
    # Start the app
    app.MainLoop()


if __name__ == '__main__':
    main()
