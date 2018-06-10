import tensorflow as tf
import os


#path = "oss://hualangdeeplearning.oss-cn-shanghai-internal.aliyuncs.com/cifar-10-batches-bin/"
image_width = 32
image_height = 32
image_depth = 3
label_length = 1
image_length = image_height * image_width * image_depth
record_length = label_length + image_length
def read_cifar10(filequeue):
    reader= tf.FixedLengthRecordReader(record_length)
    key,value = reader.read(filequeue)
    record = tf.decode_raw(value,tf.uint8)

    label = tf.cast(tf.slice(record,[0], [label_length]), tf.int32)

    image = tf.slice(record, [label_length], [image_length])
    #print(tf.shape(image))
    image = tf.cast(tf.reshape(image, [image_depth, image_width, image_height]), tf.float32)
    image = tf.transpose(image, [1,2,0])

    distort_image = tf.random_crop(image, [32,32,3])
    distort_image = tf.image.random_flip_up_down(distort_image)
    distort_image = tf.image.random_flip_left_right(distort_image)
    distort_image = tf.image.random_hue(distort_image,max_delta=0.05)
    distort_image = tf.image.random_brightness(distort_image, max_delta=63)
    distort_image = tf.image.random_contrast(distort_image,lower=0.2, upper=1.8)
    distort_image = tf.image.random_saturation(distort_image,lower=0.0,upper=2.0)

    image = tf.image.per_image_standardization(distort_image)

    return (label, image)

def get_train(directory, batch_size, if_train = None):
    if if_train is not None:
        file = [os.path.join(directory,"data_batch_%d.bin" %i) for i in list(range(1, 6))]
    else:
        file = [os.path.join(directory, "test_batch.bin")]

    filequeue = tf.train.string_input_producer(file)
    label, image = read_cifar10(filequeue)
    min_dequeue_size = 5000
    capacity = 5000 + 3 * batch_size
    label, image = tf.train.shuffle_batch([label, image], batch_size, capacity=capacity, min_after_dequeue=min_dequeue_size)
    #label = tf.reshape(label, [batch_size])
    #label = tf.one_hot(label,10,on_value=1, axis = 1)

    label = tf.reshape(label, [batch_size])



    return label, image

if __name__ == "__main__":

    #label, image = read_cifar10(filequeue)
    label, image = get_train(path, 10)




    with tf.Session() as sess:
        for i in list(range(10)):
            tf.train.start_queue_runners(sess)
            label_print = sess.run(label)
            print(label_print)
            print(label_print.shape)
            #image_print = sess.run(image)
            #print(image_print)
            #image_print = sess.run(image)
            #print(len(image_print))