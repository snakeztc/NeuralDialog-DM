from Utils.config import generalConfig, wordDqnConfig, model_dir
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop


class LstmDnnQ(BatchAgent):
    mode_path = model_dir+"best-lstm.h5"

    def init_model(self):
        print "Creating model"
        embed_size = wordDqnConfig["embedding"]
        pooling_type = wordDqnConfig["pooling"]
        use_pool = pooling_type != None

        model = Sequential()
        model.add(Embedding(self.domain.nb_words+1, embed_size, mask_zero=(not use_pool)))

        if wordDqnConfig["recurrent"] == "LSTM":
            model.add(LSTM(wordDqnConfig["first_hidden"], return_sequences=use_pool))
        else:
            model.add(GRU(wordDqnConfig["first_hidden"], return_sequences=use_pool))
        model.add(Dropout(0.2))

        if use_pool:
            model.add(TimeDistributedMerge(pooling_type))

        model.add(Dense(wordDqnConfig["second_hidden"], init='lecun_uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))

        model.add(Dense(self.domain.actions_num, init='lecun_uniform'))
        model.add(Activation('linear'))

        opt = RMSprop(clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        print "Model created"
        print model.summary()
        return model


