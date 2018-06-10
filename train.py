import tensorflow as tf
import vgg_lib
import cifar_read
import vgg_parameter
#from vgg_cifar10_aliyun import vgg_lib
#from vgg_cifar10_aliyun import cifar_read
#from vgg_cifar10_aliyun import vgg_parameter
import os
import time



directory = "oss://hualangdeeplearning.oss-cn-shanghai-internal.aliyuncs.com/cifar-10-batches-bin/"
#directory = "../cifar-10-batches-bin/"
y_batch, x_batch = cifar_read.get_train(directory, vgg_parameter.BATCH_SIZE, 1)

print("y_batch", y_batch.get_shape())



prediction = vgg_lib.vgg16(x_batch)
print("prediction", prediction.get_shape())

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels = y_batch))


accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_batch, tf.int64), tf.argmax(prediction, 1)),tf.float32))
globle_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(1e-3, globle_step,  decay_rate=0.96, decay_steps=2000, staircase=True)
#learning_rate1 = tf.train.exponential_decay(1e-3, globle_step,  decay_rate=0.95, decay_steps=2000, staircase=True)
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy, globle_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()



with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "oss://hualangdeeplearning.oss-cn-shanghai-internal.aliyuncs.com/cifar10_aliyun/save/train_save")

    for i in list(range(30000000)):
        tf.train.start_queue_runners(sess)


        _, predict,  accum,  = sess.run([train_step, prediction,  accuracy])
        if i % 1 == 0:

                print("time:", time.time())
                print("step: %d"%i)
                #print("learning rate:", l)
                print("accuracy:", accum)
        if i % 50 == 0:
                saver.save(sess, "oss://hualangdeeplearning.oss-cn-shanghai-internal.aliyuncs.com/cifar10_aliyun/save/train_save")

    
