a = h_conv5.eval(feed_dict={x: batch[0]})
a = a[0]
first_image = np.array(a, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()