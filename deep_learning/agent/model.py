from logging import getLogger

import os
import json
import hashlib

from deep_learning.config import Config
from deep_learning.agent.model_api import ChessAPI
# from config import Config

import tensorflow as tf

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


logger = getLogger(__name__)
"""
Model used to make predictions on a chess game

Model updated to TF 2.0 from https://github.com/Zeta36/chess-alpha-zero

attributes:
    config: the Config class with parameters to use
"""
class ChessModel:
    def __init__(self, config: Config):
        self.config = config
        self.channels_first = True
        self.data_format = "channels_first"
        self.Input_Layer_Dim = Input((18,8,8))
        self.api = None

    def get_pipes(self, num = 1):
        """
        Creates a list of pipes on which observations of the game state will be listened for. Whenever
        an observation comes in, returns policy and value network predictions on that pipe.

        :param int num: number of pipes to create
        :return str(Connection): a list of all connections to the pipes that were created
        """
        if self.api is None:
            self.api = ChessAPI(self)
            self.api.start()
        return [self.api.create_pipe() for _ in range(num)]


    def build_model(self):

        mc = self.config.model
        in_x = x = self.Input_Layer_Dim # used as the input first layer of the model
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding='same',
                                    data_format=self.data_format, use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                                    name='input_convolutional-{}-{}'.format(str(mc.cnn_first_filter_size), str(mc.cnn_filter_num)))(x)
        x = BatchNormalization(axis=1, name='input_batchnorm')(x) # used for forward training instead of gradient descent
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_remaining_block(x, i+1)

        res_out = x

        # chain the res_out through the policy and value neural nets
        x = Conv2D(filters=mc.p_filter_size, kernel_size=mc.p_kernel_size, data_format=self.data_format,
                    use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="policy_conv-{}-{}".format(mc.p_kernel_size, mc.p_filter_size))(res_out)
        x = BatchNormalization(axis=1, name='policy_batchnorm')(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name='policy_flatten')(x)


        policy_out = Dense(self.config.num_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        x = Conv2D(filters=mc.v_filter_size, kernel_size=mc.v_kernel_size, data_format=self.data_format,
                   use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="value_conv-{}-{}".format(mc.v_kernel_size, mc.v_filter_size))(res_out)
        x = BatchNormalization(axis=1, name="value_bactchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)

        value_out = Dense(1,kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="Chesster")


    def _build_remaining_block(self, x, index):
        mc = self.config.model
        in_x = x
        indexed_name="res{}".format(str(index))
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
           data_format=self.data_format, use_bias=False, kernel_regularizer=l2(mc.l2_reg),
           name=indexed_name+"_conv1-{}-{}".format(str(mc.cnn_filter_size),str(mc.cnn_filter_num)))(x)
        x = BatchNormalization(axis=1, name=indexed_name+"_batchnorm1")(x)
        x = Activation("relu",name=indexed_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format=self.data_format, use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=indexed_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=indexed_name+"_add")([in_x, x])
        x = Activation("relu", name=indexed_name+"_relu2")(x)
        return x

    @staticmethod
    def fetch_digest(weights_path):
        if os.path.exists(weights_path):
            m = hashlib.sha256()
            with open(weights_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def save(self, config_path, weights_path):
        """
        Saves the model.
            arguments:
                config_path: path to configuration file
                weight_path: path to the weights
        """
        logger.debug("saved model to {}".format(config_path))
        with open(config_path, 'wt') as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weights_path)
        self.digest = self.fetch_digest(weights_path)
        logger.debug("Saved model digest {}".format(self.digest))

    def load(self, config_path, weights_path):
        if os.path.exists(config_path) and os.path.exists(weights_path):
            logger.debug("loading model from {}".format(config_path))
            with open(config_path, 'rt') as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weights_path)
            self.model._make_predict_function()
            self.digest = self.fetch_digest(weights_path)
            logger.debug("loaded model digest = {}".format(self.digest))
            return True
        else:
            logger.debug("model no existy..")
            return False



if __name__=="__main__":
    model = ChessModel(Config())

    model.build_model()
