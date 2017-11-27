import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[64, 784])

mnist.test._images[0]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# Reshape the image
x_image = tf.reshape(x, [64, 28, 28, 1])
# Weights for first convolution
W_conv1 = weight_variable([3, 3, 1, 8])
b_conv1 = bias_variable([8])
# Convolution and max pooling output size: 14x14x8
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# Weights for second convolution
W_conv2 = weight_variable([3, 3, 8, 4])
b_conv2 = bias_variable([4])
# Convolution and max pooling output size: 7x7x4
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#  Weights for third convolution
W_conv3 = weight_variable([3, 3, 4, 2])
b_conv3 = bias_variable([2])
# Convolution size: 7x7x2
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# Transposed convolution wieghts
d_conv1 = weight_variable([2, 2, 4, 2])
# Output shape
out_shape1 = [64, 14, 14, 4]
# Filter size
filter_size = [2, 2, 4, 2]
# Stride size
stride  = [1, 2, 2, 1]
# First deconvolution size :14x14x4
t_conv1 = tf.nn.conv2d_transpose(h_conv3,d_conv1,out_shape1,stride, padding='SAME', data_format = 'NHWC')
# Weight for convolution
W_conv4 = weight_variable([3, 3, 4, 4])
b_conv4 = bias_variable([4])
# Convolution output size 14x14x4
h_conv4 = tf.nn.relu(conv2d(t_conv1, W_conv4) + b_conv4)
# Weights for deconvolution
d_conv2 = weight_variable([2, 2, 8, 4])
# Output shape
out_shape1 = [64, 28, 28, 8]
# Last deconvolution
t_conv2 = tf.nn.conv2d_transpose(h_conv4,d_conv2,out_shape1,stride, padding='SAME', data_format = 'NHWC')
# Weights for convlution
W_conv5 = weight_variable([3,3,8,8])
b_conv5 = bias_variable([8])
# Convolution
h_conv5 = tf.nn.relu(conv2d(t_conv2, W_conv5) + b_conv5)
# Weights for last convolution
W_conv6 = weight_variable([3,3,8,1])
b_conv6 = bias_variable([1])
# Last convolution
h_conv7 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)



loss = tf.reduce_mean(tf.square(h_conv7 - x_image))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
loss_vector4 = []
loss_vector3 = []
loss_vector2 = []


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(500):
    batch = mnist.train.next_batch(64)
    train_step.run(feed_dict={x: batch[0]})
    loss_value = loss.eval(feed_dict = {x: batch[0]})
    loss_vector4.append(loss_value)
    if i % 100 == 0:
      print("Batch: " + str(i) + " Loss value:" + str(loss_value))
    if i == 499:
      input1 = batch[0][0]
      first_image = np.array(input1, dtype='float')
      pixels1 = first_image.reshape((28, 28))
      input2 = batch[0][1]
      second_image = np.array(input2, dtype='float')
      pixels2 = second_image.reshape((28, 28))
      output = h_conv7.eval(feed_dict={x: batch[0]})
      first_output = output[0]
      fir_output = np.array(first_output, dtype='float')
      pixels_out1 = fir_output.reshape((28, 28))
      second_output = output[1]
      sec_output = np.array(second_output, dtype='float')
      pixels_out2 = sec_output.reshape((28, 28))
      fig1 = plt.figure(1)
      plt.imshow(pixels1, cmap='gray')
      fig1.savefig('11')
      fig2 = plt.figure(1)
      plt.imshow(pixels2, cmap='gray')
      fig2.savefig('12')
      fig3 = plt.figure(1)
      plt.imshow(pixels_out1, cmap='gray')
      fig3.savefig('13')
      fig4 = plt.figure(1)
      plt.imshow(pixels_out2, cmap='gray')
      fig4.savefig('14')

loss = tf.reduce_mean(tf.square(h_conv7 - x_image))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(500):
    batch = mnist.train.next_batch(64)
    train_step.run(feed_dict={x: batch[0]})
    loss_value = loss.eval(feed_dict = {x: batch[0]})
    loss_vector3.append(loss_value)
    if i % 100 == 0:
      print("Batch: " + str(i) + " Loss value:" + str(loss_value))
    if i == 499:
      input1 = batch[0][0]
      first_image = np.array(input1, dtype='float')
      pixels1 = first_image.reshape((28, 28))
      input2 = batch[0][1]
      second_image = np.array(input2, dtype='float')
      pixels2 = second_image.reshape((28, 28))
      output = h_conv7.eval(feed_dict={x: batch[0]})
      first_output = output[0]
      fir_output = np.array(first_output, dtype='float')
      pixels_out1 = fir_output.reshape((28, 28))
      second_output = output[1]
      sec_output = np.array(second_output, dtype='float')
      pixels_out2 = sec_output.reshape((28, 28))
      fig1 = plt.figure(1)
      plt.imshow(pixels1, cmap='gray')
      fig1.savefig('21')
      fig2 = plt.figure(1)
      plt.imshow(pixels2, cmap='gray')
      fig2.savefig('22')
      fig3 = plt.figure(1)
      plt.imshow(pixels_out1, cmap='gray')
      fig3.savefig('23')
      fig4 = plt.figure(1)
      plt.imshow(pixels_out2, cmap='gray')
      fig4.savefig('24')

loss = tf.reduce_mean(tf.square(h_conv7 - x_image))
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(500):
    batch = mnist.train.next_batch(64)
    train_step.run(feed_dict={x: batch[0]})
    loss_value = loss.eval(feed_dict = {x: batch[0]})
    loss_vector2.append(loss_value)
    if i % 100 == 0:
      print("Batch: " + str(i) + " Loss value:" + str(loss_value))
    if i == 499:
      input1 = batch[0][0]
      first_image = np.array(input1, dtype='float')
      pixels1 = first_image.reshape((28, 28))
      input2 = batch[0][1]
      second_image = np.array(input2, dtype='float')
      pixels2 = second_image.reshape((28, 28))
      output = h_conv7.eval(feed_dict={x: batch[0]})
      first_output = output[0]
      fir_output = np.array(first_output, dtype='float')
      pixels_out1 = fir_output.reshape((28, 28))
      second_output = output[1]
      sec_output = np.array(second_output, dtype='float')
      pixels_out2 = sec_output.reshape((28, 28))
      fig1 = plt.figure(1)
      plt.imshow(pixels1, cmap='gray')
      fig1.savefig('31')
      fig2 = plt.figure(1)
      plt.imshow(pixels2, cmap='gray')
      fig2.savefig('32')
      fig3 = plt.figure(1)
      plt.imshow(pixels_out1, cmap='gray')
      fig3.savefig('33')
      fig4 = plt.figure(1)
      plt.imshow(pixels_out2, cmap='gray')
      fig4.savefig('34')

epochs = range(len(loss_vector2))

fig5 = plt.figure()
plt.plot(epochs,loss_vector2)
plt.plot(epochs,loss_vector3)
plt.plot(epochs,loss_vector4)
plt.title('Loss function')
plt.legend(['0.1','0.01','0.001'], loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig5.savefig('Loss.jpg')



