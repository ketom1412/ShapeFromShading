import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from model.CnnTwelve import cnn
from model.modelThirteen import cnn
from utils.config import get, is_file_prefix
from data_scripts.getImageDataBetter import read_data_sets
import scipy.misc

if __name__ == '__main__':
	sess = tf.InteractiveSession()
	input_layer, prediction_layer = cnn()
	saver = tf.train.Saver()
	saver.restore(sess, get('TRAIN.CNN.CHECKPOINT'))

	io.use_plugin('freeimage')

	# faces = read_data_sets(one_hot = False)
	for i in range(0,2000):
	#for i in range(0, 200):
		if(i % 100 == 0):
			print("doing image {}".format(i))
		strFilenameImage = get('DATA.TestedImages')+ str(int(i)) + ".png"
		encodedImage = (scipy.misc.imread(strFilenameImage)/255-0.5)*2
		prediction = prediction_layer.eval(feed_dict={input_layer: np.expand_dims(encodedImage,axis=0)})
		scipy.misc.imsave('../eecs442challenge/Submit2/{}.png'.format(i), prediction[0,:,:,:])
		



