from Utils.config import generalConfig, turnDqnConfig, model_dir
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.optimizers import RMSprop
from keras.layers.core import TimeDistributedMerge, TimeDistributedDense


class TurnLstmDnnQ(BatchAgent):
    mode_path = model_dir+"best-turn-lstm.h5"

    def init_model(self):
        print "Creating model"
        embed_size = turnDqnConfig["embedding"]
        pooling_type = turnDqnConfig["pooling"]
        use_pool = pooling_type != None

        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(None, self.representation.state_features_num)))

        model.add(TimeDistributedDense(embed_size, input_dim=self.representation.state_features_num))

        if turnDqnConfig["recurrent"] == "LSTM":
            model.add(LSTM(turnDqnConfig["first_hidden"], input_dim=embed_size, return_sequences=use_pool))
        else:
            model.add(GRU(turnDqnConfig["first_hidden"], input_dim=embed_size, return_sequences=use_pool))
        model.add(Dropout(0.2))

        if use_pool:
            model.add(TimeDistributedMerge(pooling_type))

        model.add(Dense(turnDqnConfig["second_hidden"], init='lecun_uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))

        model.add(Dense(self.domain.actions_num, init='lecun_uniform'))
        model.add(Activation('linear'))

        opt = RMSprop(clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        print "Model created"
        print model.summary()
        return model

