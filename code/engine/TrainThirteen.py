import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from utils.config import get, is_file_prefix
from data_scripts.getImageDataBetter import read_data_sets
from model.modelThirteen import cnn

def get_weights(saver, sess):
	''' load model weights if they were saved previously '''
	if is_file_prefix('TRAIN.CNN.CHECKPOINT'):
		saver.restore(sess, get('TRAIN.CNN.CHECKPOINT'))
		print('Yay! I restored weights from a saved model!')
	else:
		print('OK, I did not find a saved model, so I will start training from scratch!')

def report_training_progress(batch_index, input_layer, loss_func, faces):
	''' Update user on training progress '''
	if batch_index % 5:
		return
	print('starting batch number %d \033[100D\033[1A' % batch_index)
	if batch_index % 25:
		return
	print("batch index{}".format(batch_index))
	print("about to calculate loss")
	error = loss_func.eval(
		feed_dict={input_layer: (faces.validation.getValidation(faces.validation.images[:12],"images")/255.0-0.5)*2,\
						masks: faces.validation.getValidation(faces.validation.masks[:12],"masks"), \
				   true_normals: (faces.validation.getValidation(faces.validation.normals[:12],"normals")/255.0-0.5)*2})
	print('\n \t rmse is about %f' % error)

def train_cnn(input_layer, prediction_layer, loss_func, optimizer, faces):
	''' Train CNN '''
	try:
		for batch_index in range(get('TRAIN.CNN.NB_STEPS')):
			report_training_progress(batch_index, input_layer, loss_func, faces)
			batch_images, batch_masks, batch_normals = faces.train.next_batch(get('TRAIN.CNN.BATCH_SIZE'))
			optimizer.run(feed_dict={input_layer: batch_images, masks: batch_masks, true_normals: batch_normals})
	
	except KeyboardInterrupt:
		print('OK, I will stop training even though I am not finished.')

if __name__ == '__main__':

	print('building model...')
	sess = tf.InteractiveSession()  # start talking to tensorflow backend
	input_layer, prediction_layer = cnn()  # fetch model layers
	masks = tf.placeholder(tf.float32, shape=[None, 128, 128])
	true_normals = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])

	print("over here you you{}".format(tf.shape(prediction_layer)))
	rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(prediction_layer, true_normals))))
	optimizer = tf.train.AdamOptimizer(learning_rate=get('TRAIN.CNN.LEARNING_RATE')).minimize(
		rmse)  # TODO: define the training step
	sess.run(tf.global_variables_initializer())  # initialize some globals
	saver = tf.train.Saver()  # prepare to save model
	# load model weights if they were saved previously
	get_weights(saver, sess)

	print('loading data...')
	faces = read_data_sets(one_hot=False)

	print('training...')
	train_cnn(input_layer, prediction_layer, rmse, optimizer, faces)

	print('saving trained model...\n')
	saver.save(sess, get('TRAIN.CNN.CHECKPOINT'))





	