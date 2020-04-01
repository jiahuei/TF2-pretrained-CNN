# Pretrained CNN Models in TensorFlow 2

Simple Python 3 wrapper for `keras_applications` and `tensorflow_hub` CNN models.

- Allows extraction of any number of intermediate layers (feature maps) from the CNN.
- The CNNs can be fine-tuned. Currently Batch Norm layers are updated when fine-tuning.
- The models are implemented as `tf.keras.layers.Layer` instances.


## References

* [Keras Applications](https://github.com/keras-team/keras-applications/tree/0bb8618db8d764e85159b898688c269312fa386b)
* [TensorFlow Hub](https://www.tensorflow.org/hub/api_docs/python/hub)


## List of CNNs:

For more details, refer to `ops/pretrained_cnn.py`.

* VGG 16 and 19
* Inception V1
* Inception V2
* Inception V3
* ResNet V1 variants
* ResNet V2 variants
* Inception-ResNet-V2
* NASNet Large and Mobile
* PNASNet Large
* DenseNet variants
* EfficientNet B0 to B7
* MobileNet V1 variants
* MobileNet V2 variants
