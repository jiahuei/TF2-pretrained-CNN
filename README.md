# Pretrained CNN Models in TensorFlow 2

Simple Python 3 wrapper for `keras_applications` and `tensorflow_hub` CNN models.

- Allows extraction of any number of intermediate layers (feature maps) from the CNN. 
- The models are implemented as `tf.keras.layers.Layer` instances.


## References

* [Keras Applications](https://github.com/keras-team/keras-applications/tree/976050c468ff949bcbd9b9cf64fe1d5c81db3f3a)
* [TensorFlow Hub](https://www.tensorflow.org/hub/api_docs/python/hub)


## List of CNNs:

For more details, refer to `ops/pretrained_cnn.py`.

* VGG 16 and 19
* Inception V1 to V3
* ResNet V1 and V2
* Inception-ResNet-V2
* NASNet Large and Mobile
* PNASNet Large
* DenseNet
* EfficientNet B0 to B7
* MobileNet V1 and V2