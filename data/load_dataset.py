from __future__ import print_function
from scipy import misc
import os
import numpy as np
import sys

def load_test_data(dped_dir, test_size):

    test_directory_GT = dped_dir +  'test_datasets/GT/'

    NUM_TEST_IMAGES = 100
    all_I = np.zeros((test_size, 512*512*3))
    all_I_bicubic = np.zeros((test_size, 512*512*3))

    TEST_IMAGES = np.random.choice(np.arange(800, 800+NUM_TEST_IMAGES), test_size, replace=False)

    i = 0
    for img in TEST_IMAGES:
        if os.path.exists(test_directory_GT + '0' + str(img) + '.png'):
            I = np.asarray(misc.imread(test_directory_GT + '0' + str(img) + '.png'))
            I_bicubic = misc.imresize(I, 0.25, interp="bicubic")
            I_bicubic = misc.imresize(I_bicubic, 4.0, interp="bicubic")

            I = np.float16(I)/255
            I_bicubic = np.float16(I_bicubic)/255

            shape = I.shape
            x_start = np.random.randint(0, shape[0]-512)
            y_start = np.random.randint(0, shape[1]-512)
            I = I[x_start:x_start+512, y_start:y_start+512, :]
            I_bicubic = I_bicubic[x_start:x_start+512, y_start:y_start+512, :]


            # I = I[np.newaxis, :]
            # I_bicubic = I_bicubic[np.newaxis, :]
            I = I.reshape(512*512*3)
            I_bicubic = I_bicubic.reshape(512*512*3)

            all_I[i, :] = I
            all_I_bicubic[i, :] = I_bicubic

            i += 1

    return all_I_bicubic, all_I


def load_batch(dped_dir, train_size):

    train_directory_phone = dped_dir +  'train_datasets/GT/'

    NUM_TRAINING_IMAGES = 800
    all_I = np.zeros((train_size, 512*512*3))
    all_I_bicubic = np.zeros((train_size, 512*512*3))

    TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), train_size, replace=False)

    i = 0
    for img in TRAIN_IMAGES:
        zero_number = 4 - len(str(img))
        zero = '0' * zero_number
        if os.path.exists(train_directory_phone + zero + str(img) + '.png'):
            I = np.asarray(misc.imread(train_directory_phone + zero + str(img) + '.png'))
            I_bicubic = misc.imresize(I, 0.25, interp="bicubic")
            I_bicubic = misc.imresize(I_bicubic, 4.0, interp="bicubic")

            I = np.float16(I)/255
            I_bicubic = np.float16(I_bicubic)/255

            shape = I.shape
            x_start = np.random.randint(0, shape[0]-512)
            y_start = np.random.randint(0, shape[1]-512)
            I = I[x_start:x_start+512, y_start:y_start+512, :]
            I_bicubic = I_bicubic[x_start:x_start+512, y_start:y_start+512, :]


            I = I.reshape(512*512*3)
            I_bicubic = I_bicubic.reshape(512*512*3)

            all_I[i, :] = I
            all_I_bicubic[i, :] = I_bicubic

            i += 1


    return all_I_bicubic, all_I
