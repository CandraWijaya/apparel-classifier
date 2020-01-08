import matplotlib.pyplot as plt

def gettraingraph(model, type, serial, traintype):
	plt.plot(model.history.history[type])
	plt.plot(model.history.history['val_' + type])

	if type == 'acc':
		plt.title('model accuracy')
		plt.ylabel('accuracy')
	elif type == 'loss':
		plt.title('model loss')
		plt.ylabel('loss')

	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('graph/' + type + '_cats_dogs_' + traintype + '_' + str(serial) + '.png', transparent = False, bbox_inches = 'tight')
	plt.show()