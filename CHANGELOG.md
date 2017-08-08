
# Experiments

## `tensorflow` experiments:

* `exp.tf_001`:
  * SoftMax Regression on MNIST data using `tensorflow`
  * Reaches ~_92%_ of test accuracy.

* `exp.tf_002`:
  * A 4-layer ConvNet (~LeCunnNet) on MNIST data with dropout using `tensorflow`
  * Reaches ~_99.2%_ of test accuracy.

* `exp.tf_003`:
  * A 4-layer ConvNet (~LeCunnNet) on MNIST data with dropout using `tensorflow`
  * Reaches ~99% of test accuracy.
  * Same as `exp.tf_002` but with `tf.nn.convolution` rather than `tf.nn.conv2d`
  * Reaches the same ~_99.2%_ of test accuracy.

* `exp.tf_004`:
  * tensorflow-mechanics tutorial
  * A 2-layer FC net applied on MNIST with structured code using tensorflow.
  * Throughs a `Bus Error` if the command `summary_writer.flush()` is included (see `FIXME:`).
    * MAYBE: you should open a `tensorboard` on that log directory to read it.
  * Saves checkpoints (but does not save training summary)
  * Reaches ~_98.1%_ of evaluation accuracy.