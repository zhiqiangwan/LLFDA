import sys
sys.path.append('../')
import os
import glob
import math
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras import backend as K
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512_Siamese_analyze_contrastive_loss import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator_paired_input import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

# %matplotlib inline

# Set a few configuration parameters.
img_height = 512
img_width = 512
# model_mode indicates the way the pretrained model was created.
# In training model, Model_Build should == 'Load_Model'. decode_detections will be called in the Evaluator.
# However, decode_detections is run on CPU and is very slow.
# In inference model, Model_Build should == 'New_Model_Load_Weights'.
# DecodeDetections will be called when build the model. DecodeDetections is writen in tensorflow and is run GPU.
# It seems that the result under inference model is slightly better than that under training model.
# Maybe DecodeDetections and decode_detections are not exactly the same.
model_mode = 'inference'  # 'training'#
assert model_mode == 'inference'

model_path = '../trained_weights/SSD512_City_to_foggy0_01_resize_400_800/current/pool12_loss_weights_0_000005/epoch-55_loss-3.9684_val_loss-5.2015.h5'

batch_size = 8

DatasetName = 'City_to_foggy0_01_paired_input'
processed_dataset_path = './processed_dataset_h5/' + DatasetName
if not os.path.exists(processed_dataset_path):
    os.makedirs(processed_dataset_path)

if len(glob.glob(os.path.join(processed_dataset_path, '*.h5'))):
    Dataset_Build = 'Load_Dataset'
else:
    Dataset_Build = 'New_Dataset'


# The anchor box scaling factors used in the original SSD512
scales_coco = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
scales = scales_coco
top_k = 200
confidence_thresh = 0.01
nms_iou_threshold = 0.5


if DatasetName == 'City_to_foggy0_01_paired_input':
    resize_image_to = (400, 800)

    # Our model will produce predictions for these classes.
    classes = ['background',
               'person', 'rider', 'car', 'truck',
               'bus', 'train', 'motorcycle', 'bicycle']
    train_classes = classes
    train_include_classes = 'all'
    # Number of positive classes, 8 for domain Cityscapes, 20 for Pascal VOC, 80 for MS COCO, 1 for SIM10K
    n_classes = len(classes) - 1

else:
    raise ValueError('Undefined dataset name.')


# 1: Build the Keras model

K.clear_session()  # Clear previous models from memory.

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

# model.output = `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
# In inference mode, the predicted locations have been converted to absolute coordinates.
# In addition, we have performed confidence thresholding, per-class non-maximum suppression, and top-k filtering.
model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer= [[1.0, 2.0, 0.5],
                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                          [1.0, 2.0, 0.5],
                                          [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 128, 256, 512],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=confidence_thresh,
                iou_threshold=nms_iou_threshold,
                top_k=top_k,
                nms_max_output_size=400)

# 2: Load the trained weights into the model
model.load_weights(model_path, by_name=True)

if Dataset_Build == 'New_Dataset':
    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

    # Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

    train_dataset = DataGenerator(dataset='train', load_images_into_memory=False, hdf5_dataset_path=None)

    Cityscapes_images_dir = '../../datasets/Cityscapes/JPEGImages'
    Cityscapes_target_images_dir = '../../datasets/CITYSCAPES_beta_0_01/JPEGImages'

    # The directories that contain the annotations.
    Cityscapes_annotation_dir = '../../datasets/Cityscapes/Annotations'

    # The paths to the image sets.
    Cityscapes_train_source_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_source.txt'
    Cityscapes_train_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_target.txt'

    # images_dirs, image_set_filenames, and annotations_dirs should have the same length
    train_dataset.parse_xml(images_dirs=[Cityscapes_images_dir,
                                         Cityscapes_target_images_dir],
                            image_set_filenames=[Cityscapes_train_source_image_set_filename,
                                                 Cityscapes_train_target_image_set_filename],
                            annotations_dirs=[Cityscapes_annotation_dir,
                                              Cityscapes_annotation_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
    # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
    # option in the constructor, because in that cas the images are in memory already anyway. If you don't
    # want to create HDF5 datasets, comment out the subsequent two function calls.

    # After create these h5 files, if you have resized the input image, you need to reload these files. Otherwise,
    # the images and the labels will not change.

    train_dataset.create_hdf5_dataset(file_path=os.path.join(processed_dataset_path, 'dataset_train.h5'),
                                      resize=resize_image_to,
                                      variable_image_size=True,
                                      verbose=True)

    train_filenames = []
    with open(Cityscapes_train_source_image_set_filename, 'r') as f:
        train_filenames.extend([os.path.join(Cityscapes_images_dir, line.strip()) for line in f])
    with open(Cityscapes_train_target_image_set_filename, 'r') as f:
        train_filenames.extend([os.path.join(Cityscapes_target_images_dir, line.strip()) for line in f])

    train_dataset = DataGenerator(dataset='train',
                                  load_images_into_memory=False,
                                  hdf5_dataset_path=os.path.join(processed_dataset_path, 'dataset_train.h5'),
                                  filenames=train_filenames,
                                  filenames_type='text',
                                  images_dir=Cityscapes_images_dir)

elif Dataset_Build == 'Load_Dataset':
    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

    # The directories that contain the images.
    Cityscapes_images_dir = '../../datasets/Cityscapes/JPEGImages'
    Cityscapes_target_images_dir = '../../datasets/CITYSCAPES_beta_0_01/JPEGImages'

    # The paths to the image sets.
    Cityscapes_train_source_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_source.txt'
    Cityscapes_train_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_target.txt'

    train_filenames = []
    with open(Cityscapes_train_source_image_set_filename, 'r') as f:
        train_filenames.extend([os.path.join(Cityscapes_images_dir, line.strip()) for line in f])
    with open(Cityscapes_train_target_image_set_filename, 'r') as f:
        train_filenames.extend([os.path.join(Cityscapes_target_images_dir, line.strip()) for line in f])

    train_dataset = DataGenerator(dataset='train',
                                  load_images_into_memory=False,
                                  hdf5_dataset_path=os.path.join(processed_dataset_path, 'dataset_train.h5'),
                                  filenames=train_filenames,
                                  filenames_type='text',
                                  images_dir=Cityscapes_images_dir)

else:
    raise ValueError('Undefined Dataset_Build. Dataset_Build should be New_Dataset or Load_Dataset.')

# Make predictions:
# 1: Set the generator for the predictions.

# For the test generator:
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# First convert the input image to 3 channels and size img_height X img_width
# Also, convert the groundtruth bounding box
# Remember, if you want to visualize the predicted box on the original image,
# you need to apply the corresponding reverse transformation.
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

test_generator = train_dataset.generate(batch_size=batch_size,
                                        shuffle=False,
                                        transformations=[convert_to_3_channels,
                                                         resize],
                                        label_encoder=None,
                                        returns={'processed_images',
                                                 'filenames',
                                                 'inverse_transform',
                                                 'original_images',
                                                 'original_labels'},
                                        keep_images_without_gt=False)

n_images = train_dataset.get_dataset_size() // 2
n_batches = int(math.ceil(n_images / batch_size))
print("Number of images in the dataset:\t{:>6}".format(n_images))

contrastive_loss = [0] * 9

for i in range(n_batches):
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(test_generator)
    y_pred = model.predict(batch_images)
    for k, batch_item in enumerate(y_pred):
        contrastive_loss[k] += np.mean(np.sum(np.square(batch_item), axis=1))



