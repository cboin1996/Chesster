from logging import getLogger


# from chess_zero.agent.api_chess import ChessModelAPI
# from deep_learning.config import Config
from config import Config

import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, Input, BatchNormalization, Activation, Add, Flatten
from tensorflow.keras.regularizers import l2



"""
Model used to make predictions on a chess game

Model updated to TF 2.0 from https://github.com/Zeta36/chess-alpha-zero

attributes:
    config: the Config class with parameters to use
"""
class ChessModel:
    def __init__(self, config: Config):
        self.config = config

    def build_model(self):
        mc = self.config.model
        in_x = x = Input((18,8,8)) # used as the input first layer of the model
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size,
                                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                                    name='input_convolutional-{}-{}'.format(str(mc.cnn_first_filter_size), str(mc.cnn_filter_num)))(x)
        x = BatchNormalization(axis=1, name='input_batchnorm')(x) # used for forward training instead of gradient descent
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_remaining_block(x, i+1)

        res_out = x

        # chain the res_out through the policy and value neural nets
        x = Conv2D(filters=mc.p_filter_size, kernel_size=mc.p_kernel_size, data_format="channels_first",
                    use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="policy_conv-{}-{}".format(mc.p_kernel_size, mc.p_filter_size))(res_out)
        x = BatchNormalization(axis=1, name='policy_batchnorm')(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name='policy_flatten')(x)


        policy_out = Dense(self.config.num_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        x = Conv2D(filters=mc.v_filter_size, kernel_size=mc.v_kernel_size, data_format="channels_first",
                   use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="value_conv-{}-{}".format(mc.v_kernel_size, mc.v_filter_size))(res_out)
        x = BatchNormalization(axis=1, name="value_bactchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)

        value_out = Dense(1,kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = tf.keras.Model(in_x, [policy_out, value_out], name="Chesster")


    def _build_remaining_block(self, x, index):
        mc = self.config.model
        in_x = x
        indexed_name="res{}".format(str(index))
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
           data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
           name=indexed_name+"_conv1-{}-{}".format(str(mc.cnn_filter_size),str(mc.cnn_filter_num)))(x)
        x = BatchNormalization(axis=1, name=indexed_name+"_batchnorm1")(x)
        x = Activation("relu",name=indexed_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=indexed_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=indexed_name+"_add")([in_x, x])
        x = Activation("relu", name=indexed_name+"_relu2")(x)
        return x





if __name__=="__main__":
    model = ChessModel(Config())

    model.build_model()
