
# coding: utf-8

# In[2]:


import os
import numpy as np
import tensorflow as tf


# In[3]:


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


# In[4]:


def read_labeled_image_list(data_dir, data_list):
    # returns full paths of images and masks
    f = open(data_list, 'r')
    images = []
    masks = []
    
    for line in f:
        image, mask = line.strip("\n").split(' ')
        images.append(data_dir+image)
        masks.append(data_dir+mask)
        
    return images, masks


# In[6]:


def read_images_from_disk(input_queue, input_size, random_scale):
    
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    label = tf.image.decode_png(label_contents, channels=1)
    
    if input_size is not None:
        h, w = input_size
        if random_scale:
            scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
            h_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(img)[0]), scale))
            w_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(img)[1]), scale))
            new_shape = tf.squeeze(tf.pack([h_new, w_new]), squeeze_dims=[1])
            
            img = tf.image.resize_images(img, new_shape)
            label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
            label = tf.squeeze(label, squeeze_dims=[0])
            
        img = tf.image.resize_image_with_crop_or_pad(img, h, w)
        label = tf.image.resize_image_with_crop_or_pad(label, h, w)
        
    # RGB to BGR
    img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
    img = tf.cast(tf.concat(2,[img_b, img_g, img_r]), dtype=tf.float32)
    
    img -= IMG_MEAN
    
    return img, label


# In[9]:


class ImageReader(object):
    
    def __init__(self, data_dir, data_list, input_size, random_scale, coord):
        
        self.data_dir = data_dir # path to images and mask directory
        self.data_list = data_list # path to list containing file locations
        self.intput_size = input_size
        self.coord = coord # tf queue coordinator
        
        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        
        # form queue to pass as argument to read_images_from_disk
        self.queue = tf.train_slice_input_producer([self.images, self.labels], shuffle=input_size is not None)
        
        # read images from disk
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale)
        
    def dequeue(self, num_elements):
        # pack images and labels into a batch
        
        image_batch, label_batch = tf.train_batch([self.image, self.label], num_elements)
        
        return image_batch, label_batch

