import numpy as np
from keras.layers import *
from keras.models import *
from keras import losses
from keras import callbacks
import pandas as pd

def createModel(shape):
    model = Sequential()
    model.add(Dense(50, input_shape=(30,3)))
    model.add(LSTM(50, input_shape=(30,3)))
    #model.add(Dropout(0.5))
    model.add(Dense(2))
    #model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Activation('softmax'))
    model.compile(optimizer="adam", loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def loadModel(model_name):
    # load json and create model
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+'.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    return loaded_model

def testModel(X, y, model):
    for i in range(len(X)):
        res = model.predict(np.array([X[i]]))
        print("PREDICTION1: "+str(np.argmax(res[0])))
        print("REAL: "+str(y[i]))

embeddings_dict = {}
with open("glove.6B.50d.txt", 'r', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
		
with open("senti_binary.train", "r") as fr:
	DATA_x = []
	DATA_y = []
	tot_w = 0
	no_w = 0
	for l in fr.readlines():
		el = l.split('\t')
		new_x = []
		for w in el[0].split(" "):
			tot_w += 1
			if w.lower() in embeddings_dict.keys():
				new_x.append(embeddings_dict[w.lower()])
			else:
				no_w += 1
				#METTERE VETTORE NULLO
				#print(w + " NON C'E'")

		mean_new_x = []
		for sent in new_x:			
			sent = np.array(sent)
			mean_new_x.append(sent.mean())
		if len(mean_new_x) < 30:
			for i in range(30-len(mean_new_x)):
				mean_new_x.append(0.0)
		if len(mean_new_x) > 30:
			mean_new_x = mean_new_x[:30]

		mean_new_x_3d = []
		for i in range(len(mean_new_x)):
			ts = []
			if i == 0:
				ts = [0, mean_new_x[i], mean_new_x[i+1]]
			elif i == len(mean_new_x)-1:
				ts = [mean_new_x[i-1], mean_new_x[i], 0]
			else:
				ts = [mean_new_x[i-1], mean_new_x[i], mean_new_x[i+1]]
			mean_new_x_3d.append(ts)
		DATA_x.append(mean_new_x_3d)
		y = [0,0]
		y[int(el[1])] = 1
		DATA_y.append(y)

	DATA_x = np.array(DATA_x)
	print(DATA_x.shape)
	DATA_y = np.array(DATA_y)
	print(DATA_y.shape)

	model = createModel(DATA_x.shape)
	es = callbacks.EarlyStopping(monitor='loss',
                                  min_delta=0,
                                  patience=10,
                                  verbose=0, mode='auto')
	print("ALL SET")
	model.fit(x=DATA_x, y=DATA_y, epochs=10)

	testModel(DATA_x[:10], DATA_y[:10], model)

	# serialize model to JSON
	model_json = model.to_json()
	with open("models\\model_sent1.json", "w") as json_file:
		json_file.write(model_json)
    # serialize weights to HDF5
	model.save_weights("models\\model_sent1.h5")
	print("Saved model to disk")


