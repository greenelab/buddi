# import the VAE code
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from buddi.preprocessing import sc_preprocess
from buddi.plotting import validation_plotting as vp
from comparator_models.models import vae4

# general imports
import warnings
from typing import Any
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import rankdata
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# programming stuff
import time
import os
import pickle
from pathlib import Path
from argparse import ArgumentParser


# ===============================================================
# === training
# ===============================================================

@dataclass
class VAETrainParameters:
    """
    Parameters for constructing a Buddi model.

    n_label_z: dimension of latent code for each non-y latent space

    TODO: fix format, populate with all parameters
    """
    n_z: int = 266 # same as vae4 64+64+64+64+10
    encoder_dim1: int = 784
    encoder_dim2: int = 512
    decoder_dim1: int = 784
    decoder_dim2: int = 512
    batch_size: int = 500
    n_epoch: int = 100
    beta_kl: float = 0.01
    activ: str = 'relu'
    adam_learning_rate: float = 0.0005

@dataclass
class VAETrainResults:
    vae: Any
    encoder: Any
    decoder: Any
    loss_fig: Any
    output_folder: Path

default_params = VAETrainParameters()



def make_loss_df(vae_hist):

    # write out the loss for later plotting
    # unpack the loss values
    val_recon_loss = vae_hist.history['val_recon_loss']
    train_recon_loss = vae_hist.history['recon_loss']



    # make into a dataframe
    loss_df = pd.DataFrame(data=val_recon_loss, columns=['val_recon_loss'])
    loss_df['batch'] = [*range(len(val_recon_loss))]
    loss_df['train_recon_loss'] = train_recon_loss


    # add the log to make it easier to plot
    loss_df["log_val_recon_loss"] = np.log10(loss_df["val_recon_loss"]+1)
    loss_df["log_train_recon_loss"] = np.log10(loss_df["train_recon_loss"]+1)




    return loss_df


def _make_loss_fig(loss_df, ax, title, loss_to_plot):
    ## plot loss
    g = sns.lineplot(
        x="batch", y=loss_to_plot,
        data=loss_df,
        legend="full",
        alpha=0.3, ax= ax
    )
    
    title = f"{title} Final Loss sum: {np.round(loss_df[loss_to_plot].iloc[-1], 3)}"
    ax.set_title(title)
    return g



def make_loss_fig(loss_df):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    _make_loss_fig(loss_df, ax=axs[0], title=f"Training Recon Loss", loss_to_plot="train_recon_loss")
    _make_loss_fig(loss_df, ax=axs[1], title=f"Validation Recon Loss", loss_to_plot="val_recon_loss")


    fig.suptitle("Loss curves", fontsize=14)

    fig.show()



def plot_reconstruction_vae(encoder, decoder,
        X_temp, Y_temp, label_num,
        batch_size=500):


    # now use the encoder to get the latent spaces
    mu_slack = encoder.predict(X_temp, batch_size=batch_size)


    # and decode
    decoded_outputs = decoder.predict(mu_slack, batch_size=batch_size)

    # combine the true output and the reconstruction
    X_dup = np.vstack([X_temp, decoded_outputs])


    Y_dup = np.append(Y_temp, Y_temp)

    label_dup = np.append(label_num, label_num)
    source_dup = np.asarray(np.append([0]*label_num.shape[0], [1]*label_num.shape[0]))

    fig, axs = plt.subplots(1, 3, figsize=(30,5))

    plot_df = vp.get_pca_for_plotting(np.asarray(X_dup))
    vp.plot_pca(plot_df, color_vec=Y_dup, ax=axs[0], title="PCA of orig+reconstruction -- Cell Type color")
    vp.plot_pca(plot_df, color_vec=label_dup, ax=axs[1], title="PCA of orig+reconstruction -- Sample Type color")
    vp.plot_pca(plot_df, color_vec=source_dup, ax=axs[2], title="PCA of orig+reconstruction -- Source Type color")


    fig.suptitle("Reconstructed and Original Training Data", fontsize=14)
    axs[1].legend([],[], frameon=False)

    return fig

def calc_VAE_perturbation_kang(X_full, Y_full, meta_df, encoder, decoder, 
                           scaler, batch_size, genes_ordered, top_lim=100):
    
    # get the cell type specific single cell reference
    idx_sc_ref = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.samp_type == "sc_ref")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.cell_prop_type == "cell_type_specific")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.sample_id == "1015")
    idx_sc_ref = np.where(idx_sc_ref)[0]


    sc_ref_meta_df = meta_df.iloc[idx_sc_ref]
    print(sc_ref_meta_df.cell_prop_type.value_counts())


    X_sc_ref = np.copy(X_full)
    X_sc_ref = X_sc_ref[idx_sc_ref,]

    ## get the transformation vector
    proj_vec = get_pert_transform_vec_VAE(X_full, meta_df, encoder, decoder, batch_size)



    # get the CTRL encodings
    mu_slack, z_slack = encoder.predict(X_sc_ref, batch_size=batch_size)

    # add the latent proj
    encoded_0_1 = z_slack + proj_vec

    # decode
    decoded_0_1 = decoder.predict(encoded_0_1, batch_size=batch_size)
    decoded_0_0 = decoder.predict(z_slack, batch_size=batch_size)

    decoded_0_1 = scaler.inverse_transform(decoded_0_1)
    decoded_0_0 = scaler.inverse_transform(decoded_0_0)

    final_meta_df = sc_ref_meta_df
    final_meta_df["isTraining"] = "Test"


    top_genes = {}
    de_genes_all = None
    for curr_cell_type in Y_full.columns:


        # this is for the "projected" expression
        curr_idx = np.where(final_meta_df.Y_max == curr_cell_type)[0]
        proj_ctrl = decoded_0_0[curr_idx]
        proj_stim = decoded_0_1[curr_idx]

        # take the median for nomalization

        proj_ctrl = np.median(rankdata(proj_ctrl, axis=1), axis=0)
        proj_stim = np.median(rankdata(proj_stim, axis=1), axis=0)
        proj_log2FC = np.abs(proj_stim-proj_ctrl)

        # make into DF
        proj_log2FC_df = pd.DataFrame(proj_log2FC, index=genes_ordered)

        intersect_proj = proj_log2FC_df.loc[genes_ordered][0]
        top_proj_genes = intersect_proj.index[np.argsort(np.abs(intersect_proj))].tolist()[::-1][0:top_lim]

        top_genes[curr_cell_type] = top_proj_genes


    return (final_meta_df, decoded_0_0, decoded_0_1, top_genes)



def get_pert_transform_vec_VAE(X_full, meta_df, encoder, decoder, batch_size):

    # get the stimulated bulks
    idx_stim_train = np.logical_and(meta_df.samp_type == "bulk", meta_df.isTraining == "Train")
    idx_stim_train = np.logical_and(idx_stim_train, meta_df.stim == "STIM")
    idx_stim_train = np.where(idx_stim_train)[0]
    idx_stim_train = np.random.choice(len(idx_stim_train), 500, replace=True)

    # get the ctrl bulks
    idx_ctrl_train = np.logical_and(meta_df.samp_type == "bulk", meta_df.isTraining == "Train")
    idx_ctrl_train = np.logical_and(idx_ctrl_train, meta_df.stim == "CTRL")
    idx_ctrl_train = np.random.choice(len(idx_ctrl_train), 500, replace=True)

    X_ctrl = X_full[idx_ctrl_train]
    mu_slack, z_slack = encoder.predict(X_ctrl, batch_size=batch_size)
    train_ctrl = z_slack

    X_stim = X_full[idx_stim_train]
    mu_slack, z_slack = encoder.predict(X_stim, batch_size=batch_size)
    train_stim = z_slack


    train_stim_med = np.median(train_stim, axis=0)
    train_ctrl_med = np.median(train_ctrl, axis=0)

    proj_train = train_stim_med - train_ctrl_med

    return proj_train

def calc_VAE_perturbation(X_full, Y_full, meta_df, encoder, decoder, 
                           scaler, batch_size, genes_ordered, top_lim=100):
    
    # get the cell type specific single cell reference
    idx_sc_ref = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.samp_type == "sc_ref")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.cell_prop_type == "cell_type_specific")
    idx_sc_ref = np.where(idx_sc_ref)[0]


    sc_ref_meta_df = meta_df.iloc[idx_sc_ref]
    print(sc_ref_meta_df.cell_prop_type.value_counts())


    X_sc_ref = np.copy(X_full)
    X_sc_ref = X_sc_ref[idx_sc_ref,]

    ## get the transformation vector
    proj_vec = get_pert_transform_vec_VAE(X_full, meta_df, encoder, decoder, batch_size)



    # get the CTRL encodings
    mu_slack, z_slack = encoder.predict(X_sc_ref, batch_size=batch_size)

    # add the latent proj
    encoded_0_1 = z_slack + proj_vec

    # decode
    decoded_0_1 = decoder.predict(encoded_0_1, batch_size=batch_size)
    decoded_0_0 = decoder.predict(z_slack, batch_size=batch_size)

    decoded_0_1 = scaler.inverse_transform(decoded_0_1)
    decoded_0_0 = scaler.inverse_transform(decoded_0_0)

    final_meta_df = sc_ref_meta_df
    final_meta_df["isTraining"] = "Test"


    top_genes = {}
    de_genes_all = None
    for curr_cell_type in Y_full.columns:


        # this is for the "projected" expression
        curr_idx = np.where(final_meta_df.Y_max == curr_cell_type)[0]
        proj_ctrl = decoded_0_0[curr_idx]
        proj_stim = decoded_0_1[curr_idx]

        # take the median for nomalization

        proj_ctrl = np.median(rankdata(proj_ctrl, axis=1), axis=0)
        proj_stim = np.median(rankdata(proj_stim, axis=1), axis=0)
        proj_log2FC = np.abs(proj_stim-proj_ctrl)

        # make into DF
        proj_log2FC_df = pd.DataFrame(proj_log2FC, index=genes_ordered)

        intersect_proj = proj_log2FC_df.loc[genes_ordered][0]
        top_proj_genes = intersect_proj.index[np.argsort(np.abs(intersect_proj))].tolist()[::-1][0:top_lim]

        top_genes[curr_cell_type] = top_proj_genes


    return (final_meta_df, decoded_0_0, decoded_0_1, top_genes)



def train_vae(res_data_path, exp_id, use_vae4,
                X_vae,
                params: VAETrainParameters=default_params):
    
    # set seeds
    from numpy.random import seed
    seed(1)
    from tensorflow.random import set_seed
    set_seed(2)

    n_x = X_vae.shape[1]
    n_z = 266



    ##################################################
    #####. Train Model first pass
    ##################################################
    if use_vae4:
        vae, encoder, decoder = vae4.instantiate_model(
            n_x=n_x,
            n_z=params.n_z,
            encoder_dim1 = params.encoder_dim1, 
            encoder_dim2 = params.encoder_dim2, 
            decoder_dim1 = params.decoder_dim1, 
            decoder_dim2 = params.decoder_dim2, 
            batch_size = params.batch_size, 
            n_epoch = params.n_epoch,  
            beta_kl = params.beta_kl, 
            activ = params.activ, 
            optim = tf.keras.optimizers.legacy.Adam(learning_rate=params.adam_learning_rate)
        )


    # make test train split
    idx_train = np.random.choice(range(X_vae.shape[0]), np.ceil(X_vae.shape[0]*0.8).astype(int), replace=False)
    idx_test = np.setdiff1d(range(X_vae.shape[0]), idx_train)

    # now we need to make the size compatible with the batch size
    train_size_samp = len(idx_train)
    train_size_samp_batch = np.ceil(train_size_samp/params.batch_size)
    train_size_samp = params.batch_size*train_size_samp_batch
    train_size_samp = train_size_samp.astype(int)

    # same for test
    test_size_samp = len(idx_test)
    test_size_samp_batch = np.ceil(test_size_samp/params.batch_size)
    test_size_samp = params.batch_size*test_size_samp_batch
    test_size_samp = test_size_samp.astype(int)

    idx_train = np.random.choice(idx_train, train_size_samp, replace=True)
    idx_test = np.random.choice(idx_test, test_size_samp, replace=True)

    vae_hist = vae.fit(X_vae[idx_train], 
                      X_vae[idx_train], verbose = 0, batch_size=params.batch_size, epochs=params.n_epoch,
                      validation_data=(X_vae[idx_test], 
                                        X_vae[idx_test]))
                                        
                                        
    loss_df = make_loss_df(vae_hist)

    # plot loss
    loss_fig = make_loss_fig(loss_df)


    vae.save(f"{res_data_path}/{exp_id}_vae")
    encoder.save(f"{res_data_path}/{exp_id}_encoder")
    decoder.save(f"{res_data_path}/{exp_id}_decoder")

    


    return VAETrainResults(
        vae=vae,
        encoder=encoder,
        decoder=decoder,
        loss_fig=loss_fig,
        output_folder=res_data_path
    )




# # ===============================================================
# # === entrypoint
# # ===============================================================

# if __name__ == "__main__":
#     # read in arguments
#     parser = ArgumentParser()
#     parser.add_argument("-res", "--res_data_path", dest="res_data_path",
#                         help="path to write DIVA results")
#     parser.add_argument("-exp", "--exp_id",
#                         dest="exp_id",
#                         help="ID for results")
#     parser.add_argument("-n", "--num_genes",
#                         dest="num_genes", type=int,
#                         help="Number of features (genes) for VAE")
#     parser.add_argument("-hd", "--hyp_dict",
#                         dest="hyp_dict", type=int,
#                         help="Dictionary of hyperparameters")

#     args = parser.parse_args()

#     params = BuddiParameters(
#         n_label_z=args
#     )
