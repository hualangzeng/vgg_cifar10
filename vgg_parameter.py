
BATCH_SIZE = 200
IMAGE_SIZE = [32,32,3]
TRAIN_SIZE= [BATCH_SIZE, 32,32,3]
OUTPUT_SIZE = [BATCH_SIZE, 1]
IMAGE_CHANNEL_NUM = 1

conv1_weight_size = [3,3,3,64]
conv1_bias_size = [64]
conv1_stride_size = [1,1,1,1]
conv1_padding_value = "SAME"

conv2_weight_size = [3,3,64,64]
conv2_bias_size = [64]
conv2_stride_size = [1,1,1,1]
conv2_padding_value = "SAME"

pooling3_k_size = [1, 2, 2, 1]
pooling3_stride_size = [1, 2,2,1]
pooling3_padding_value = "VALID"

conv4_weight_size = [3,3,64,128]
conv4_bias_size = [128]
conv4_stride_size = [1,1,1,1]
conv4_padding_value = "SAME"

conv5_weight_size = [3,3,128,128]
conv5_bias_size = [128]
conv5_stride_size = [1,1,1,1]
conv5_padding_value = "SAME"

pooling6_k_size = [1, 2, 2, 1]
pooling6_stride_size = [1, 2,2,1]
pooling6_padding_value = "VALID"

conv7_weight_size = [3,3,128,256]
conv7_bias_size = [256]
conv7_stride_size = [1,1,1,1]
conv7_padding_value = "SAME"

conv8_weight_size = [3,3,256,256]
conv8_bias_size = [256]
conv8_stride_size = [1,1,1,1]
conv8_padding_value = "SAME"

conv9_weight_size = [3,3,256,256]
conv9_bias_size = [256]
conv9_stride_size = [1,1,1,1]
conv9_padding_value = "SAME"

pooling10_k_size = [1, 2, 2, 1]
pooling10_stride_size = [1, 2,2,1]
pooling10_padding_value = "VALID"

conv11_weight_size = [3,3,128,256]
conv11_bias_size = [256]
conv11_stride_size = [1,1,1,1]
conv11_padding_value = "SAME"

conv12_weight_size = [3,3,256,256]
conv12_bias_size = [256]
conv12_stride_size = [1,1,1,1]
conv12_padding_value = "SAME"

conv13_weight_size = [3,3,256,256]
conv13_bias_size = [256]
conv13_stride_size = [1,1,1,1]
conv13_padding_value = "SAME"

pooling14_k_size = [1, 2, 2, 1]
pooling14_stride_size = [1, 2,2,1]
pooling14_padding_value = "VALID"

conv15_weight_size = [8,8,64,128]
conv15_bias_size = [128]
conv15_stride_size = [1,1,1,1]
conv15_padding_value = "VALID"
conv15_reshape = [BATCH_SIZE, 128]

fc_16_weight_size = [128,128]
fc_16_bias_size = [128]

fc_17_weight_size = [384, 192]
fc_17_bias_size = [192]

fc_18_weight_size = [192, 10]
fc_18_bias_size = [10]








