from Utils.config import generalConfig, hybridDqnConfig, model_dir
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dropout, Masking, Input, Dense, TimeDistributed
from keras.optimizers import RMSprop
from Domains.Domain import Domain


class HybridLstmDnnQ(BatchAgent):
    mode_path = model_dir+"best-turn-lstm.h5"

    def init_model(self):
        return self.get_graph_model()

    def get_graph_model(self):
        print "Creating model"
        embed_size = hybridDqnConfig["embedding"]

        loss = {}
        outputs = []

        model_input = Input(shape=(None, self.representation.state_features_num), name='input')
        input_mask = Masking(mask_value=0.0, input_shape=(None, self.representation.state_features_num),
                             name='input_mask')(model_input)
        turn_embed = TimeDistributed(Dense(embed_size, input_dim=self.representation.state_features_num),
                                     name='turn_embed')(input_mask)
        dialog_embed = LSTM(hybridDqnConfig["recurrent_size"], input_dim=embed_size, return_sequences=False,
                            name='dialog_embed')(turn_embed)
        dropped_dialog = Dropout(hybridDqnConfig["dropout"])(dialog_embed)

        # add the policy networks
        for p_idx, p_name in zip(self.domain.policy_names, self.domain.policy_str_name):
            l1 = Dense(hybridDqnConfig["l1-"+p_name], activation='tanh', name="l1-"+p_name)(dropped_dialog)
            dl1 = Dropout(hybridDqnConfig["dropout"], name="l1dp-"+p_name)(l1)
            o = Dense(self.domain.policy_action_num[p_idx], activation='linear', name=p_name)(dl1)
            loss[p_name] = "mse"
            outputs.append(o)

        # add all supervised signals
        spl_l1 = Dense(hybridDqnConfig["l1-spl-share"], activation='tanh', name="l1-spl-share")(dropped_dialog)
        dspl_l1 = Dropout(hybridDqnConfig["dropout"], name="l1-spl-drop")(spl_l1)

        for idx, s_idx in enumerate(self.domain.spl_indexs):
            s_name = self.domain.spl_str_name[idx]
            s_modal = self.domain.spl_modality[idx]
            s_type = self.domain.spl_type[idx]
            s_loss = 'categorical_crossentropy' if s_type == Domain.categorical else 'mse'
            s_act = 'softmax' if s_type == Domain.categorical else 'linear'

            o = Dense(s_modal, activation=s_act, name=s_name)(dspl_l1)
            loss[s_name] = s_loss
            outputs.append(o)

        model = Model(input=model_input, output=outputs)
        opt = RMSprop(clipvalue=1.0)
        model.compile(optimizer=opt, loss=loss)
        print model.summary()
        print "Model created"
        return model






