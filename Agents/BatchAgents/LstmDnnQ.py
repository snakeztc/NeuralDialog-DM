from Utils.config import *
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.layers.core import TimeDistributedMerge

class LstmDnnQ(BatchAgent):

    def init_model(self):
        print "Creating model"
        embed_size = rnnDqnConfig["embedding"]
        pooling_type = rnnDqnConfig["pooling"]
        use_pool = pooling_type != None

        model = Sequential()
        model.add(Embedding(self.domain.nb_words+1, embed_size, mask_zero=(not use_pool)))

        if rnnDqnConfig["recurrent"] == "LSTM":
            model.add(LSTM(rnnDqnConfig["first_hidden"], return_sequences=use_pool))
        else:
            model.add(GRU(rnnDqnConfig["first_hidden"], return_sequences=use_pool))
        model.add(Dropout(0.2))

        if use_pool:
            model.add(TimeDistributedMerge(pooling_type))

        model.add(Dense(rnnDqnConfig["second_hidden"], init='lecun_uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))

        model.add(Dense(self.domain.actions_num, init='lecun_uniform'))
        model.add(Activation('linear'))

        opt = RMSprop(clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        print "Model created"
        print model.summary()
        return model

