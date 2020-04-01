# -*- coding: utf-8 -*-
"""
Created on 06 Feb 2020 16:26:17

@author: jiahuei
"""
import os
import sys
import time
import numpy as np
import urllib.request as urllib
from pprint import pprint
from datetime import datetime

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# import tensorflow_hub as tfhub

up_dir = os.path.dirname
CURR_DIR = up_dir(os.path.realpath(__file__))
BASE_DIR = up_dir(up_dir(CURR_DIR))
COMMON = os.path.join(BASE_DIR, 'common')
sys.path.insert(1, BASE_DIR)
from common.ops import pretrained_cnn as cops

NO_BATCHNORM = ['vgg_16', 'vgg_19']
decode_predictions = tf.keras.applications.imagenet_utils.decode_predictions
# Set random seeds
rand_seed = 2
np.random.seed(rand_seed)
tf.random.set_seed(rand_seed)


class Config:
    rand_seed = 2
    kernel_initializer = 'xavier_uniform'
    cnn_feat_map_name = None
    cnn_name = None
    cnn_trainable = False
    dense_trainable = False


class CNNModel(tf.keras.Model):
    def __init__(self, config, optimizer=None, name='CNNModel'):
        super().__init__(name=name)
        c = self._config = config
        self.default_input_size = cops.get_cnn_default_input_size(c.cnn_name)
        _mssg = '>>> CNN: {}    CNN trainable: {}    Dense trainable: {}'.format(
            c.cnn_name, c.cnn_trainable, c.dense_trainable)
        if c.cnn_name in cops.TFHUB_CNN_LIST_TF2:
            self.encoder_cnn = cops.TFHubCNN(
                cnn_name=c.cnn_name,
                cnn_feat_map_name=[c.cnn_feat_map_name],
                include_top=True,
                trainable=c.cnn_trainable)
            _mssg += '    Type: TF-Hub'
        else:
            self.encoder_cnn = cops.KerasCNN(
                cnn_name=c.cnn_name,
                cnn_feat_map_name=[c.cnn_feat_map_name],
                input_shape=self.default_input_size,
                include_top=True,
                trainable=c.cnn_trainable)
            _mssg += '    Type: TF-Keras'
        print(_mssg)
        self.loss_layer = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        if c.dense_trainable:
            self.dense_layer = tf.keras.layers.Dense(
                units=1000,
                activation=None,
                use_bias=True,
                name='projection_layer')
        if optimizer is not None:
            assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
            self.optimizer = optimizer
    
    # noinspection PyMethodOverriding
    def call(self, inputs, training):
        c = self._config
        assert isinstance(inputs, list)
        assert len(inputs) == 1
        images = self.preprocess_image(inputs[0])
        preds, endpoints = self.encoder_cnn.forward(images=images, training=training)
        assert len(endpoints) == 1, 'More than one feature map / layer returned'
        # pprint(self.encoder_cnn.get_updates_for(images))
        if c.dense_trainable:
            fm = endpoints[c.cnn_feat_map_name]
            assert len(fm.shape) == 4, 'Feature map is not a rank-4 tensor'
            vec = tf.nn.softmax(self.dense_layer(tf.math.reduce_mean(fm, axis=[1, 2])))
            preds = (preds + vec) / 2.
        return preds, endpoints
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def classify(self, image_string):
        preds, _ = self.__call__(inputs=[image_string], training=False)
        return preds
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def optimise_step(self, image_string):
        with tf.GradientTape() as tape:
            preds, endpoints = self.__call__(inputs=[image_string], training=True)
            loss = self.loss_layer(y_true=[0], y_pred=preds)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # pprint([t.name for t in trainable_variables])
        return loss
    
    def preprocess_image(self, image_string):
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.central_crop(image, central_fraction=0.875)
        image = tf.expand_dims(image, 0)
        image = tf.compat.v1.image.resize_bilinear(image, self.default_input_size[:2], align_corners=False)
        # image = tf.image.resize(image, input_size)
        return image


def test_cnn(config):
    print('')
    
    # Create model
    opt = tf.keras.optimizers.Adam(0.5)  # Use a large LR so that change in tvar is easier to detect
    cnn = CNNModel(config=conf, optimizer=opt)
    
    # Run the net once to build / compile
    pred0 = cnn.classify(img_string).numpy()
    if config.dense_trainable is False and config.cnn_trainable is False:
        pprint(decode_predictions(pred0))
    
    # Collect variables before infer runs
    v0 = [t.numpy() for t in cnn.variables]
    cnn_tv0 = [t.numpy() for t in cnn.encoder_cnn.trainable_variables]
    bn0 = [t.numpy() for t in cnn.non_trainable_variables]
    dense_tv0 = [t.numpy() for t in cnn.dense_layer.trainable_variables] if hasattr(cnn, 'dense_layer') else []
    
    # Run a few more times
    for i in range(2):
        _ = cnn.classify(img_string).numpy()
    pred1 = cnn.classify(img_string).numpy()
    assert np.all(pred0 == pred1), 'Predictions changed between inference runs'
    # pprint(decode_predictions(pred1))
    
    # Collect variables after infer runs
    v1 = [t.numpy() for t in cnn.variables]
    cnn_tv1 = [t.numpy() for t in cnn.encoder_cnn.trainable_variables]
    bn1 = [t.numpy() for t in cnn.non_trainable_variables]
    dense_tv1 = [t.numpy() for t in cnn.dense_layer.trainable_variables] if hasattr(cnn, 'dense_layer') else []
    
    # Train the net
    for i in range(5):
        _ = cnn.optimise_step(img_string)
    
    # Collect variables after training runs
    v2 = [t.numpy() for t in cnn.variables]
    cnn_tv2 = [t.numpy() for t in cnn.encoder_cnn.trainable_variables]
    bn2 = [t.numpy() for t in cnn.non_trainable_variables]
    dense_tv2 = [t.numpy() for t in cnn.dense_layer.trainable_variables] if hasattr(cnn, 'dense_layer') else []
    
    print('')
    
    # Assert that no variable changes between inference runs
    if config.cnn_trainable and config.cnn_name in NO_BATCHNORM:
        assert len(bn0) == 0 and len(bn1) == 0 and len(bn2) == 0, 'BN / Non-trainable vars list should be empty'
    else:
        assert len(bn0) > 0 and len(bn1) > 0 and len(bn2) > 0, 'BN / Non-trainable vars list should not be empty'
    vars_change_infer = any([np.any(x != y) for x, y in zip(v0, v1)])
    cnn_tvars_change_infer = any([np.any(x != y) for x, y in zip(cnn_tv0, cnn_tv1)])
    bn_vars_change_infer = any([np.any(x != y) for x, y in zip(bn0, bn1)])
    dense_tvars_change_infer = any([np.any(x != y) for x, y in zip(dense_tv0, dense_tv1)])
    assert vars_change_infer is False, 'Some model vars changed after inference runs'
    assert cnn_tvars_change_infer is False, 'Some CNN tvars changed after inference runs'
    assert bn_vars_change_infer is False, 'Some BN / Non-trainable vars changed after inference runs'
    assert dense_tvars_change_infer is False, 'Some Dense tvars changed after inference runs'
    
    cnn_tvars_change_train = any([np.any(x != y) for x, y in zip(cnn_tv2, cnn_tv1)])
    bn_vars_change_train = any([np.any(x != y) for x, y in zip(bn2, bn1)])
    
    if config.cnn_trainable:
        assert len(cnn_tv0) > 0 and len(cnn_tv1) > 0 and len(cnn_tv2) > 0, \
            'CNN tvars list should not be empty'
        # Assert that CNN, BN, Dense variables change after training
        assert cnn_tvars_change_train is True, 'CNN tvars unchanged after training'
        if config.cnn_name not in NO_BATCHNORM:
            assert bn_vars_change_train is True, 'BN / Non-trainable vars unchanged after training'
    else:
        assert len(cnn_tv0) == 0 and len(cnn_tv1) == 0 and len(cnn_tv2) == 0, \
            'CNN tvars list should be empty'
        # Assert that only Dense variables change after training
        assert cnn_tvars_change_train is False, 'Some CNN tvars changed after training'
        assert bn_vars_change_train is False, 'Some BN / Non-trainable vars changed after training'
    
    if config.dense_trainable:
        assert len(dense_tv0) > 0 and len(dense_tv1) > 0 and len(dense_tv2) > 0, \
            'Dense tvars list should not be empty'
        dense_tvars_change_train = any([np.any(x != y) for x, y in zip(dense_tv2, dense_tv1)])
        assert dense_tvars_change_train is True, 'Dense tvars unchanged after training'
    else:
        assert len(dense_tv0) == 0 and len(dense_tv1) == 0 and len(dense_tv2) == 0, \
            'Dense tvars list should be empty'
    
    if not hasattr(cnn, 'dense_layer') and not config.cnn_trainable:
        assert len(v0) > 0 and len(v1) > 0 and len(v2) > 0, 'Model vars list should not be empty'
        assert config.dense_trainable is False, '`dense_trainable` should be False'
        assert config.cnn_trainable is False, '`cnn_trainable` should be False'
        # Assert that no variable changes after training
        vars_change_infer = any([np.any(x != y) for x, y in zip(v2, v1)])
        assert vars_change_infer is False, 'Some model vars changed after training'
    del cnn


def run_test(config):
    try:
        test_cnn(config)
        print('>>> PASSED.')
    except AssertionError as e:
        print('>>> FAILED: {}'.format(e))


# url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
url = 'https://upload.wikimedia.org/wikipedia/commons/b/b7/Beagle_Faraon.JPG'
img_string = urllib.urlopen(url).read()

# One from each family
cnn_list = [('inception_v1', 'Mixed_4d'),
            ('resnet_v2_50', 'block4'),
            ('inception_resnet_v2', 'MaxPool_5a_3x3'),
            ('nasnet_mobile', 'normal_concat_8'),
            ('mobilenet_v1_025_128', 'Conv2d_11_depthwise'),
            ('vgg_16', 'block4_conv3'),
            ('densenet_121', 'conv5_block7_concat'),
            ('efficientnet_b0', 'block5c_project_conv')]

conf = Config()
for cnn_name, fmap_name in cnn_list[5:6]:
    conf.cnn_name = cnn_name
    conf.cnn_feat_map_name = fmap_name
    # conf.dense_trainable = conf.cnn_trainable = False
    # run_test(conf)
    
    # dense_trainable == False: Test CNN only, output probabilities should match
    # dense_trainable == True: Test CNN as feature extractor and under fine-tuning cases
    for dense_trainable in [False, True]:
        conf.dense_trainable = dense_trainable
        for cnn_trainable in [False, True]:
            conf.cnn_trainable = cnn_trainable
            run_test(conf)
