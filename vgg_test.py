import tensorflow as tf
import vgg_lib
import cifar_read
import vgg_parameter

import time


directory = "oss://hualangdeeplearning.oss-cn-shanghai-internal.aliyuncs.com/cifar-10-batches-bin/"
y_batch, x_batch = cifar_read.get_train(directory, vgg_parameter.BATCH_SIZE)

print("y_batch", y_batch.get_shape())

# xs = tf.placeholder(tf.float32, vgg_parameter.TRAIN_SIZE)
# ys = tf.placeholder(tf.float32, vgg_parameter.OUTPUT_SIZE)


prediction = vgg_lib.vgg16(x_batch)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_batch, tf.int64), tf.argmax(prediction, 1)), tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    while True:
        tf.train.start_queue_runners(sess)
        saver.restore(sess, "oss://hualangdeeplearning.oss-cn-shanghai-internal.aliyuncs.com/cifar10_aliyun/save/train_save")
        accum = sess.run( accuracy)

        print("time:", time.time())


        print("accuracy:", accum)

        time.sleep(10)
