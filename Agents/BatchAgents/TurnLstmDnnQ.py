from Utils.config import generalConfig, turnDqnConfig, model_dir
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dropout, Masking, Input, Dense, TimeDistributed
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot


class TurnLstmDnnQ(BatchAgent):
    mode_path = model_dir+"best-turn-lstm.h5"

    def init_model(self):
        return self.get_graph_model()

    def get_graph_model(self):
        print "Creating model"
        embed_size = turnDqnConfig["embedding"]

        loss = {}
        outputs = []

        model_input = Input(shape=(None, self.representation.state_features_num), name='input')
        input_mask = Masking(mask_value=0.0, input_shape=(None, self.representation.state_features_num),
                             name='input_mask')(model_input)
        turn_embed = TimeDistributed(Dense(embed_size, input_dim=self.representation.state_features_num),
                                     name='turn_embed')(input_mask)
        dialog_embed = LSTM(turnDqnConfig["recurrent_size"], input_dim=embed_size, return_sequences=False,
                            name='dialog_embed')(turn_embed)
        dropped_dialog = Dropout(turnDqnConfig["dropout"])(dialog_embed)

        # add the policy networks
        for p_name in self.domain.policy_names:
            str_name = self.domain.policy_str_name[p_name]
            l1 = Dense(turnDqnConfig["l1-"+str_name], activation='tanh', name="l1-"+str_name)(dropped_dialog)
            dl1 = Dropout(turnDqnConfig["dropout"], name="l1dp-"+str_name)(l1)
            o = Dense(self.domain.policy_action_num[p_name], activation='linear', name=str_name)(dl1)
            loss[str_name] = "mse"
            outputs.append(o)

        model = Model(input=model_input, output=outputs)
        opt = RMSprop(clipvalue=1.0)
        model.compile(optimizer=opt, loss=loss)
        plot(model, to_file='model.png')
        print model.summary()
        print "Model created"
        return model






