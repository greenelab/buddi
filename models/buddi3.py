"""
This is the script that contains the model generation and training code for BuDDI, 
which is called by the interface buddi.py
buddi3 has 3 latent spaces other than the slack space. 
Currently it is assumed that the perturbation space is missing, but 
since each latent space other than cell-type proportion is interchangable, 
you can remove any latent space you like.

Future work will make the latent space removed more general.

"""


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

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



def fit_model(known_prop_vae, unknown_prop_vae, encoder_unlab, encoder_lab, decoder, classifier,
              X_unknown_prop, label_unknown_prop, bulk_unknown_prop, 
              X_known_prop, Y_known_prop, label_known_prop, bulk_known_prop, 
              epochs, batch_size):
    
    assert np.maximum(len(X_unknown_prop), len(X_known_prop)) > batch_size, \
            ("batch size is too big", len(X_unknown_prop), batch_size, len(X_known_prop))
    
    start = time.time()
    history = []
    history_val = []
    meta_hist = []

    # make test train split
    # This creates a train-validation set. This 20% of the data is only used for 
    # understanding how well BuDDI is training and for QC purposes.
    # in the future I will add a flag to toggle is this is wanted or not
    # for now, it is reccommended to train buddi several times and 
    # compare the different models to one another
    unkp_idx_train = np.random.choice(range(X_unknown_prop.shape[0]), np.ceil(X_unknown_prop.shape[0]*0.8).astype(int), replace=False)
    kp_idx_train = np.random.choice(range(X_known_prop.shape[0]), np.ceil(X_known_prop.shape[0]*0.8).astype(int), replace=False)

    unkp_idx_test = np.setdiff1d(range(X_unknown_prop.shape[0]), unkp_idx_train)
    kp_idx_test = np.setdiff1d(range(X_known_prop.shape[0]), kp_idx_train)

    # now we need to make the known and unknown samples the same size
    # and a size compatible with the batch size
    train_size_samp = np.maximum(len(unkp_idx_train), len(kp_idx_train))
    train_size_samp_batch = np.ceil(train_size_samp/batch_size)
    train_size_samp = batch_size*train_size_samp_batch
    train_size_samp = train_size_samp.astype(int)

    test_size_samp = batch_size

    print(f"test_size_samp: {test_size_samp}")
    print(f"train_size_samp: {train_size_samp}")

    unkp_idx_train = np.random.choice(unkp_idx_train, train_size_samp, replace=True)
    kp_idx_train = np.random.choice(kp_idx_train, train_size_samp, replace=True)
    unkp_idx_test = np.random.choice(unkp_idx_test, test_size_samp, replace=True)
    kp_idx_test = np.random.choice(kp_idx_test, test_size_samp, replace=True)

    X_unknown_prop_test = X_unknown_prop[unkp_idx_test]
    label_unknown_prop_test = label_unknown_prop[unkp_idx_test]
    bulk_unknown_prop_test = bulk_unknown_prop[unkp_idx_test]
    X_known_prop_test = X_known_prop[kp_idx_test]
    Y_known_prop_test = Y_known_prop[kp_idx_test]
    label_known_prop_test = label_known_prop[kp_idx_test]
    bulk_known_prop_test = bulk_known_prop[kp_idx_test]


    X_unknown_prop = X_unknown_prop[unkp_idx_train]
    label_unknown_prop = label_unknown_prop[unkp_idx_train]
    bulk_unknown_prop = bulk_unknown_prop[unkp_idx_train]
    X_known_prop = X_known_prop[kp_idx_train]
    Y_known_prop = Y_known_prop[kp_idx_train]
    label_known_prop = label_known_prop[kp_idx_train]
    bulk_known_prop = bulk_known_prop[kp_idx_train]


    for epoch in range(epochs):

        unlabeled_index = np.arange(len(X_unknown_prop))
        np.random.shuffle(unlabeled_index)

        labeled_index = np.arange(len(X_known_prop))
        np.random.shuffle(labeled_index)

        batches = len(X_unknown_prop) // batch_size
        for i in range(batches):
            # Labeled
            index_range =  labeled_index[i * batch_size:(i+1) * batch_size]
            loss = known_prop_vae.train_on_batch([X_known_prop[index_range], Y_known_prop[index_range]],
                                                    [X_known_prop[index_range], Y_known_prop[index_range], label_known_prop[index_range],  bulk_known_prop[index_range]])

            # Unlabeled
            index_range =  unlabeled_index[i * batch_size:(i+1) * batch_size]
            loss += [unknown_prop_vae.train_on_batch(X_unknown_prop[index_range],
                                                        [X_unknown_prop[index_range], label_unknown_prop[index_range], bulk_unknown_prop[index_range]])]


            history.append(loss)

            # validation loss
            # Labeled
            loss_val = known_prop_vae.test_on_batch([X_known_prop_test, Y_known_prop_test],
                                                    [X_known_prop_test, Y_known_prop_test, label_known_prop_test, bulk_known_prop_test])

            # Unlabeled
            loss_val += [unknown_prop_vae.test_on_batch(X_unknown_prop_test,
                                                        [X_unknown_prop_test, label_unknown_prop_test, bulk_unknown_prop_test])]

            history_val.append(loss_val)


            # track spearman corr
            y_true = Y_known_prop_test
            y_est = classifier.predict(X_known_prop_test, batch_size=batch_size)
            spr_err = [spearmanr(y_true[idx].astype(float), y_est[idx].astype(float))[0]
                            for idx in range(0, y_est.shape[0])]
            meta_hist.append(spr_err)



    done = time.time()
    elapsed = done - start
    print("Elapsed: ", elapsed)
    print("Epoch: ", epoch)

    return [history, meta_hist, history_val]

def instantiate_model(n_x,
                    n_y,
                    n_label,
                    n_bulk,
                    n_label_z = 64,
                    encoder_dim = 512,
                    decoder_dim = 512,
                    class_dim1 = 512,
                    class_dim2 = 256,
                    batch_size = 500,
                    n_epoch = 100,
                    alpha_rot = 100, #100000
                    alpha_bulk = 100, #10000
                    alpha_prop = 100, #100
                    beta_kl_slack = 0.1, # 10 ###
                    beta_kl_rot = 100, # 100 ###
                    beta_kl_bulk = 100, # 100 ###
                    activ = 'relu',
                    optim = Adam(learning_rate=0.0005)):


    def null_f(args):
        return args



    ####################
    ### Encoder
    ####################

    # declare the Keras tensor we will use as input to the encoder
    X = Input(shape=(n_x,))
    Y = Input(shape=(n_y,))
    label = Input(shape=(n_label,))
    props = Input(shape=(n_y,))
    z_in = Input(shape=(n_y+n_label_z+n_label_z+n_label_z,))


    inputs = X

    # set up encoder network
    # this is an encoder with encoder_dim hidden layer
    encoder_s = Dense(encoder_dim, activation=activ, name="encoder_slack")(inputs)
    encoder_r = Dense(encoder_dim, activation=activ, name="encoder_rot")(inputs)
    encoder_b = Dense(encoder_dim, activation=activ, name="encoder_bulk")(inputs)

    # now from the hidden layer, you get the mu and sigma for 
    # the latent space

    mu_slack = Dense(n_label_z, activation='linear', name = "mu_slack")(encoder_s)
    l_sigma_slack = Dense(n_label_z, activation='linear', name = "sigma_slack")(encoder_s)

    mu_rot = Dense(n_label_z, activation='linear', name = "mu_rot")(encoder_r)
    l_sigma_rot = Dense(n_label_z, activation='linear', name = "sigma_rot")(encoder_r)

    mu_bulk = Dense(n_label_z, activation='linear', name = "mu_bulk")(encoder_b)
    l_sigma_bulk = Dense(n_label_z, activation='linear', name = "sigma_bulk")(encoder_b)



    ####################
    ### Proportion Estimator 
    ####################

    # set up labeled classifier
    #defining the architecture of the classifier

    class_hidden1 = Dense(class_dim1, activation=activ, name = "cls_h1")
    class_hidden2 = Dense(class_dim2, activation=activ, name="cls_h2")
    class_out = Dense(n_y, activation='softmax', name="cls_out")

    classifier_h1 = class_hidden1(inputs)
    classifier_h2 = class_hidden2(classifier_h1)
    Y_cls = class_out(classifier_h2)

    ####################
    ### Latent Space
    ####################

    # now we need the sampler from mu and sigma
    def sample_z(args):
        mu, l_sigma, n_z = args
        eps = K.random_normal(shape=(batch_size, n_z), mean=0., stddev=1.)
        return mu + K.exp(l_sigma / 2) * eps


    # Sampling latent space
    z_slack = Lambda(sample_z, output_shape = (n_label_z, ), name="z_samp_slack")([mu_slack, l_sigma_slack, n_label_z])
    z_rot = Lambda(sample_z, output_shape = (n_label_z, ), name="z_samp_rot")([mu_rot, l_sigma_rot, n_label_z])
    z_bulk = Lambda(sample_z, output_shape = (n_label_z, ), name="z_samp_bulk")([mu_bulk, l_sigma_bulk, n_label_z])

    z_concat_lab = concat([z_slack, Y, z_rot, z_bulk])
    z_concat_unlab = concat([z_slack, Y_cls, z_rot, z_bulk])



    ####################
    ### Decoder
    ####################
    def null_f(args):
        return args

    ###### DECODER
    # set up decoder network
    # this is a decoder with 512 hidden layer
    decoder_hidden = Dense(decoder_dim, activation=activ, name = "decoder_h1")

    # final reconstruction
    decoder_out = Dense(n_x, activation='sigmoid', name = "decoder_out")

    d_in = Input(shape=(n_label_z+n_y+n_label_z+n_label_z,))
    d_h1 = decoder_hidden(d_in)
    d_out = decoder_out(d_h1)

    # set up the decoder part that links to the encoder
    # labeled decoder
    h_lab = decoder_hidden(z_concat_lab)
    outputs_lab = decoder_out(h_lab)

    # unlabeled decoder
    h_unlab = decoder_hidden(z_concat_unlab)
    outputs_unlab = decoder_out(h_unlab)


    ###### Rotations classifier
    # this is the rotation we try to estimate
    rot_h1 = ReLU(name = "rot_h1")
    rot_h2 = Dense(n_label, activation='linear', name = "rot_h2")
    rot_softmax = Softmax(name = "mu_rot_pred")
    decoder_sigma_r = Lambda(null_f, name = "l_sigma_rot_pred")


    rot_1_out = rot_h1(z_rot)
    rot_2_out = rot_h2(rot_1_out)
    rotation_outputs = rot_softmax(rot_2_out)
    sigma_outputs_r = decoder_sigma_r(l_sigma_rot)



    ###### Bulk classifier
    # this is the bulk or sc we try to estimate
    bulk_h1 = ReLU(name = "bulk_h1")
    bulk_h2 = Dense(n_bulk, activation='linear', name = "bulk_h2")
    bulk_softmax = Softmax(name = "mu_bulk_pred")
    decoder_sigma_d = Lambda(null_f, name = "l_sigma_bulk_pred")


    bulk_1_out = bulk_h1(z_bulk)
    bulk_2_out = bulk_h2(bulk_1_out)
    bulk_outputs = bulk_softmax(bulk_2_out)
    sigma_outputs_d = decoder_sigma_d(l_sigma_bulk)



    ###### Loss functions where you need access to internal variables
    
    def vae_loss(y_true, y_pred):
        recon = K.sum(mean_squared_error(y_true, y_pred), axis=-1)
        kl_rot = beta_kl_rot * K.sum(K.exp(l_sigma_rot) + K.square(mu_rot) - 1. - l_sigma_rot, axis=-1)
        kl_bulk = beta_kl_bulk * K.sum(K.exp(l_sigma_bulk) + K.square(mu_bulk) - 1. - l_sigma_bulk, axis=-1)
        kl_slack = beta_kl_slack * K.sum(K.exp(l_sigma_slack) + K.square(mu_slack) - 1. - l_sigma_slack, axis=-1)
        return recon + kl_rot + kl_bulk+ kl_slack


    def recon_loss(y_true, y_pred):
        return K.sum(mean_squared_error(y_true, y_pred), axis=-1)


    def prop_loss(y_true, y_pred):
        return K.sum(mean_absolute_error(y_true, y_pred), axis=-1) * alpha_prop

    def bulk_loss(y_true, y_pred):
        return K.sum(mean_absolute_error(y_true, y_pred), axis=-1) * alpha_bulk


    def class_loss(y_true, y_pred):
        recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)*alpha_rot
        return recon



    ##### link it all together
    classifier = Model(X, Y_cls)

    known_prop_vae = Model([X, Y], [outputs_lab, Y_cls, rotation_outputs, bulk_outputs])
    unknown_prop_vae = Model(X, [outputs_unlab, rotation_outputs, bulk_outputs])

    known_prop_vae.compile(optimizer=optim, loss=[vae_loss, prop_loss, class_loss, bulk_loss]) #, metrics = [KL_loss, recon_loss])
    unknown_prop_vae.compile(optimizer=optim, loss=[vae_loss, class_loss, bulk_loss]) #, metrics = [KL_loss, recon_loss])

    encoder_unlab = Model(X, [z_slack, mu_slack, l_sigma_slack, z_rot, mu_rot, l_sigma_rot, z_bulk, mu_bulk, l_sigma_bulk])

    encoder_lab = Model([X, Y], [z_slack, mu_slack, l_sigma_slack, z_rot, mu_rot, l_sigma_rot, z_bulk, mu_bulk, l_sigma_bulk])


    decoder = Model(d_in, d_out)


    return (known_prop_vae, unknown_prop_vae, encoder_unlab, encoder_lab, decoder, classifier)
