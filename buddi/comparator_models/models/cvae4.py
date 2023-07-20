# general imports
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Softmax, ReLU, ELU, LeakyReLU
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, kl_divergence
from tensorflow.keras.datasets import mnist
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import euclidean
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from tqdm import tnrange, tqdm_notebook

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# programming stuff
import time
import os
import pickle
from pathlib import Path

# disable eager execution
# https://github.com/tensorflow/tensorflow/issues/47311#issuecomment-786116401
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def instantiate_model(
        n_x = 7000,
        n_drug = 2,
        n_label = 2,
        n_tech = 2,
        n_z = 266, # same as BuDDI4 64+64+64+64+10
        encoder_dim1 = 784,
        encoder_dim2 = 512,
        decoder_dim1 = 784,
        decoder_dim2 = 512,
        batch_size = 500,
        n_epoch = 100,
        beta_kl = 0.01,
        activ = 'relu',
        optim = Adam(learning_rate=0.0005)):


    def null_f(args):
        return args



    ####################
    ### Encoder
    ####################

    # declare the Keras tensor we will use as input to the encoder
    X = Input(shape=(n_x,))
    Drug = Input(shape=(n_drug,))
    Label = Input(shape=(n_label,))
    Bulk = Input(shape=(n_tech,))

    inputs = concat([X, Label, Bulk, Drug])

    # set up encoder network
    # this is an encoder with 512 hidden layer
    # Dense is a functor, with given initializations (activation and hidden layer dimension)
    # After initialization, the functor is returned and inputs is used as an arguement
    encoder_1 = Dense(encoder_dim1, activation=activ, name="encoder_1")(inputs)
    encoder_2 = Dense(encoder_dim2, activation=activ, name="encoder_2")(encoder_1)


    # now from the hidden layer, you get the mu and sigma for 
    # the latent space

    mu_slack = Dense(n_z, activation='linear', name = "mu_slack")(encoder_2)
    l_sigma_slack = Dense(n_z, activation='linear', name = "sigma_slack")(encoder_2)


    ####################
    ### Latent Space
    ####################

    # now we need the sampler from mu and sigma
    def sample_z(args):
        mu, l_sigma, n_z = args
        eps = K.random_normal(shape=(batch_size, n_z), mean=0., stddev=1.)
        return mu + K.exp(l_sigma / 2) * eps


    # Sampling latent space
    z_slack = Lambda(sample_z, output_shape = (n_z, ), name="z_samp_slack")([mu_slack, l_sigma_slack, n_z])

    z_concat = concat([z_slack, Label, Bulk, Drug])


    ####################
    ### Decoder
    ####################
    def null_f(args):
        return args

    ###### DECODER
    # set up decoder network
    # this is a decoder with 512 hidden layer
    decoder_hidden1 = Dense(decoder_dim1, activation=activ, name = "decoder_h1")
    decoder_hidden2 = Dense(decoder_dim2, activation=activ, name = "decoder_h2")


    # final reconstruction
    decoder_out = Dense(n_x, activation='sigmoid', name = "decoder_out")

    dh_p1 = decoder_hidden1(z_concat)
    dh_p2 = decoder_hidden2(dh_p1)
    outputs = decoder_out(dh_p2)


    d_in = Input(shape=(n_z+n_label+n_tech+n_drug,))
    d_h1 = decoder_hidden1(d_in)
    d_h2 = decoder_hidden2(d_h1)
    d_out = decoder_out(d_h2)



    ###### Loss functions where you need access to internal variables
    def vae_loss(y_true, y_pred):
        recon = K.sum(mean_squared_error(y_true, y_pred), axis=-1)
        kl_slack = beta_kl * K.sum(K.exp(l_sigma_slack) + K.square(mu_slack) - 1. - l_sigma_slack, axis=-1)
        return recon + kl_slack

    def recon_loss(y_true, y_pred):
        return K.sum(mean_squared_error(y_true, y_pred), axis=-1)


    ##### link it all together
    cvae = Model([X, Label, Bulk, Drug], outputs)
    encoder = Model([X, Label, Bulk, Drug], [mu_slack, z_slack])


    decoder = Model(d_in, d_out)

    ####################
    ### compile
    ####################   

    cvae.compile(optimizer=optim, loss=vae_loss, metrics = [recon_loss])


    return (cvae, encoder, decoder)

