import tensorflow as tf
import numpy as np

def convResidual(x, im_length,in_length, channel_dim, filter_size, scale):
	as_image = tf.reshape(x, [-1, im_length,im_length,in_length])
	
	W1 = tf.Variable(tf.random_normal([1, 1, in_length, channel_dim],mean = 0.0, stddev= 0.1 / np.sqrt(filter_size*filter_size*in_length)))
	b1 = tf.Variable(tf.constant(0.01, shape=[channel_dim]))
	
	W2 = tf.Variable(tf.random_normal([1, 1, channel_dim, 128], mean = 0.0, stddev= 0.1 / np.sqrt(filter_size*filter_size*256)))
	b2 = tf.Variable(tf.constant(0.01, shape=[128]))
	
	W3 = tf.Variable(tf.random_normal([3, 3, 128, 128], mean = 0.0, stddev= 0.1 / np.sqrt(3*3*128)))
	b3 = tf.Variable(tf.constant(0.01, shape=[128]))
	
	W4 = tf.Variable(tf.random_normal([1, 1, 128, channel_dim], mean = 0.0, stddev= 0.1 / np.sqrt(filter_size*filter_size*128)))
	b4 = tf.Variable(tf.constant(0.01, shape=[channel_dim]))

	conv1 = tf.nn.conv2d(input=as_image,filter=W1,  strides=[1, scale, scale, 1], padding="SAME")
	conv12 = tf.nn.relu(tf.add(conv1,b1))
	conv12 = tf.reshape(conv12,[-1, im_length, im_length, channel_dim])
	
	conv2 = tf.nn.conv2d(input=conv12,filter=W2,  strides=[1, scale, scale, 1], padding="SAME")
	conv22 = tf.nn.relu(tf.add(conv2,b2))
	conv22 = tf.reshape(conv22,[-1,im_length,im_length,128])
	
	conv3 = tf.nn.conv2d(input=conv22,filter=W3,  strides=[1, scale, scale, 1], padding="SAME")
	conv32 = tf.nn.relu(tf.add(conv3,b3))
	conv32 = tf.reshape(conv32,[-1,im_length,im_length,128])
	
	conv4 = tf.nn.conv2d(input=conv32,filter=W4,  strides=[1, scale, scale, 1], padding="SAME")
	conv42 = tf.nn.relu(tf.add(conv4,b4))
	conv42 = tf.reshape(conv42,[-1,im_length,im_length,channel_dim])

	net = tf.add(conv42,conv12)
	output = tf.reshape(net, [-1, im_length, im_length, channel_dim])

	return output

def convOneByOne(x, im_length,in_length, channel_dim, filter_size, scale):
	as_image = tf.reshape(x, [-1, im_length,im_length,in_length])
	W1 = tf.Variable(tf.random_normal([filter_size, filter_size, in_length, channel_dim], mean = 0.0, stddev= 0.1 / np.sqrt(filter_size*filter_size*in_length)))
	b1 = tf.Variable(tf.constant(0.01, shape=[channel_dim]))
	conv1 = tf.nn.conv2d(input=as_image,filter=W1,  strides=[1, scale, scale, 1], padding="SAME")
	conv12 = tf.add(conv1,b1)
	output = tf.reshape(conv1, [-1, im_length, im_length, channel_dim])
	return output

def maxPoolingOne(x):
	avg_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return avg_pool

def maxPoolingTwo(x):
	avg_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return avg_pool

def maxPoolingThree(x):
	avg_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return avg_pool

def maxPoolingFour(x):
	avg_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return avg_pool

def avgPoolingFinal(x):
	avg_pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
	return avg_pool

def maxPoolingFinal(x):
	max_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
	return max_pool

def hourglass(input_layer):
	convolveOne = convResidual(input_layer, im_length=128,in_length=3,channel_dim=256,filter_size=1,scale =1)
	convolveOneHalf = convResidual(convolveOne, im_length=128,in_length=256,channel_dim=256,filter_size=1,scale =1)
	afterPoolOne = maxPoolingOne(convolveOneHalf) #64,64,10

	convolveTwo = convResidual(afterPoolOne,im_length=64, in_length=256,channel_dim=256,filter_size=1,scale =1)
	convolveTwoHalf = convResidual(convolveTwo, im_length=64,in_length=256,channel_dim=256,filter_size=1,scale =1)
	afterPoolTwo = maxPoolingTwo(convolveTwoHalf) #32,32,10

	convolveThree = convResidual(afterPoolTwo, im_length=32,in_length=256,channel_dim=256,filter_size=1,scale =1)
	convolveThreeHalf = convResidual(convolveThree, im_length=32,in_length=256,channel_dim=256,filter_size=1,scale =1)
	afterPoolThree = maxPoolingThree(convolveThreeHalf) #16,16,10

	convolveFour = convResidual(afterPoolThree, im_length=16,in_length=256,channel_dim=256,filter_size=1,scale =1)
	convolveFourHalf = convResidual(convolveFour, im_length=16,in_length=256,channel_dim=256,filter_size=1,scale =1)
	afterPoolFour = maxPoolingFour(convolveFourHalf) #8,8,10

	convolveFive = convResidual(afterPoolFour, im_length=8,in_length=256,channel_dim=256,filter_size=1,scale =1)
	convolveFiveHalf = convResidual(convolveFive, im_length=8,in_length=256,channel_dim=256,filter_size=1,scale =1)
	afterPoolFive = maxPoolingOne(convolveFiveHalf) #4,4,10

	resizedFive = tf.image.resize_nearest_neighbor(afterPoolFive,[8,8])
	addToFive = convResidual(convolveFiveHalf, im_length=8,in_length=256,channel_dim=256,filter_size=1,scale =1)
	addedFive = tf.add(resizedFive,addToFive)
	convolveAfterFive = convResidual(addedFive, im_length=8,in_length=256,channel_dim=256,filter_size=1,scale =1)
	
	resizedSix = tf.image.resize_nearest_neighbor(convolveAfterFive,[16,16])
	addToSix = convResidual(convolveFourHalf, im_length=16,in_length=256,channel_dim=256,filter_size=1,scale =1)
	addedSix = tf.add(resizedSix,addToSix)
	convolveSix = convResidual(addedSix, im_length=16,in_length=256,channel_dim=256,filter_size=1,scale =1)

	resizedSeven = tf.image.resize_nearest_neighbor(convolveSix,[32,32])
	addToSeven = convResidual(convolveThreeHalf, im_length=32,in_length=256,channel_dim=256,filter_size=1,scale =1)
	addedSeven = tf.add(resizedSeven,addToSeven)
	convolveSeven = convResidual(addedSeven, im_length=32,in_length=256,channel_dim=256,filter_size=1,scale =1)

	resizedEight = tf.image.resize_nearest_neighbor(convolveSeven,[64,64])
	addToEight = convResidual(convolveTwoHalf, im_length=64,in_length=256,channel_dim=256,filter_size=1,scale =1)
	addedEight = tf.add(resizedEight,addToEight)
	convolveEight = convResidual(addedEight, im_length=64,in_length=256,channel_dim=256,filter_size=1,scale =1)

	resizedNine = tf.image.resize_nearest_neighbor(convolveEight,[128,128])
	addToNine = convResidual(convolveOneHalf, im_length=128,in_length=256,channel_dim=256,filter_size=1,scale =1)
	addedNine = tf.add(resizedNine,addToNine)
	convolveNine = convResidual(addedNine, im_length=128,in_length=256,channel_dim=256,filter_size=1,scale =1)

	finish = convOneByOne(convolveNine,im_length=128,in_length=256,channel_dim=3,filter_size=1,scale=1)
	return finish
	
def cnn():
	input_layer = tf.placeholder(tf.float32, shape=[None, 128,128,3])

	hourOne = hourglass(input_layer)
	input2 = convOneByOne(tf.reshape(input_layer,[-1,128,128,3]),im_length=128,in_length=3,channel_dim=3,filter_size=1,scale=1)
	preHour2 = tf.add(hourOne,input2)

	hour2 = hourglass(preHour2)
	afterhour1 = convOneByOne(tf.reshape(hourOne,[-1,128,128,3]),im_length=128,in_length=3,channel_dim=3,filter_size=3,scale=1)
	preHour3 = tf.add(hour2,afterhour1)

	hour3 = hourglass(preHour3)
	afterhour2 = convOneByOne(tf.reshape(hour2,[-1,128,128,3]),im_length=128,in_length=3,channel_dim=3,filter_size=1,scale=1)
	
	finish = tf.add(afterhour2,hour3)
	pred_layer = tf.reshape(finish, [-1, 128, 128,3])

	return input_layer, pred_layer








