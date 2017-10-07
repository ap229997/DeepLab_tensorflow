
# coding: utf-8

# In[12]:


# evaluate on single image
from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels


# In[3]:


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


# In[4]:


# initialise all parameters with default values
SAVE_DIR = 


# In[4]:



def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("--model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()


# In[18]:


def load(saver, sess, ckpt_path):
    
    saver.restore(sess, ckpt_path)
    print ("got model parameters")


# In[11]:


def main():
    args = get_arguments()
    
    # prepare image
    img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    
    img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
    img = tf.cast(tf.concat(2,[img_b, img_g, img_r]), dtype=tf.float32)
    
    img -= IMG_MEAN
    
    net = DeepLabLFOVModel()
    
    trainable = tf.trainable_variables()
    
    pred = net.preds(tf.expand_dims(img, dim=0))
    
    # tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()
    
    sess.run(init)
    
    # load_weights
    saver = tf.train.Saver(var_list=trainable)
    load(saver, sess, args.model_weights)
    
    preds = sess.run([pred])
    
    msk = decode_labels(np.array(preds)[0, 0, :, :, 0])
    im = Image.fromarray(msk)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    im.save(args.save_dir+'mask.png')
    
    print ('done')
    
if __name__ == '__name__':
    main()

