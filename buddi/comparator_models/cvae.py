# import the VAE code
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from buddi.preprocessing import sc_preprocess
from buddi.plotting import validation_plotting as vp
from comparator_models.models import cvae4, cvae3

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
class CVAETrainParameters:
    """
    Parameters for constructing a Buddi model.

    n_label_z: dimension of latent code for each non-y latent space

    TODO: fix format, populate with all parameters
    """
    n_z: int = 266 # same as cvae4 64+64+64+64+10
    encoder_dim1: int = 784
    encoder_dim2: int = 512
    decoder_dim1: int = 784
    decoder_dim2: int = 512
    batch_size: int = 500
    n_epoch: int = 100
    beta_kl: float = 1
    activ: str = 'relu'
    adam_learning_rate: float = 0.0005

@dataclass
class CVAETrainResults:
    cvae: Any
    encoder: Any
    decoder: Any
    loss_fig: Any
    output_folder: Path

default_params = CVAETrainParameters()



def make_loss_df(cvae_hist):

    # write out the loss for later plotting
    # unpack the loss values
    val_recon_loss = cvae_hist.history['val_recon_loss']
    train_recon_loss = cvae_hist.history['recon_loss']



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



def plot_reconstruction_cvae(encoder, decoder,
        X_temp, Y_temp, label_temp, label_num, perturb_temp, bulk_temp,
        batch_size=500, use_cvae4=True):


    # now use the encoder to get the latent spaces
    if use_cvae4:
        # now use the encoder to get the latent spaces
        mu_slack, z_slack = encoder.predict([X_temp, label_temp, bulk_temp, perturb_temp], batch_size=batch_size)

        # now concatenate together
        z_concat = np.hstack([z_slack, label_temp, bulk_temp, perturb_temp])

    else:
        # now use the encoder to get the latent spaces
        mu_slack, z_slack = encoder.predict([X_temp, bulk_temp, perturb_temp], batch_size=batch_size)

        # now concatenate together
        z_concat = np.hstack([z_slack, bulk_temp, perturb_temp])



    # and decode
    decoded_outputs = decoder.predict(z_concat, batch_size=batch_size)

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

def calc_CVAE_perturbation(X_full, Y_full, meta_df, res1_enc, res1_dec, 
                            scaler, batch_size, 
                            label_1hot_full, 
                            bulk_1hot_full, 
                            drug_1hot_full,
                            genes_ordered, top_lim=100):

    from scipy.stats import rankdata

    label_1hot_temp = np.copy(label_1hot_full)
    bulk_1hot_temp = np.copy(bulk_1hot_full)
    perturb_1hot_temp = np.copy(drug_1hot_full)


    # get the single cell data 
    idx_sc_ref = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.samp_type == "sc_ref")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.cell_prop_type == "cell_type_specific")
    idx_sc_ref = np.where(idx_sc_ref)[0]


    ## this is to match up sample amounts across comparators
    idx_sc_ref = np.random.choice(idx_sc_ref, 2000, replace=True) 


    # make the metadata file
    ctrl_test_meta_df = meta_df.copy()
    ctrl_test_meta_df = ctrl_test_meta_df.iloc[idx_sc_ref]
    ctrl_test_meta_df.isTraining = "Test"
    ctrl_test_meta_df.stim = "CTRL"


    X_sc_ref = np.copy(X_full)
    X_sc_ref = X_sc_ref[idx_sc_ref,]

    # get the sample_ids we will perturb
    idx_sc_ref = np.where(np.logical_and(meta_df.stim == "CTRL", meta_df.samp_type == "bulk"))[0]
    idx_sc_ref = np.random.choice(idx_sc_ref, 2000, replace=True) 
    sample_code_unpert = label_1hot_temp[idx_sc_ref]

    idx_sc_ref = np.where(np.logical_and(meta_df.stim == "STIM", meta_df.samp_type == "bulk"))[0]
    idx_sc_ref = np.random.choice(idx_sc_ref, 2000, replace=True) 
    sample_code_pert = label_1hot_temp[idx_sc_ref]

    # get the bulk code
    idx_bulk = np.where(meta_df.samp_type == "bulk")[0]
    bulk_code = bulk_1hot_temp[idx_sc_ref]


    #####
    # get (un)perturbed latent codes
    #####
    idx_stim = np.where(meta_df.stim == "STIM")[0]
    idx_stim = np.random.choice(idx_stim, 2000, replace=True) 
    perturbed_code = perturb_1hot_temp[idx_stim]

    idx_ctrl = np.where(meta_df.stim == "CTRL")[0]
    idx_ctrl = np.random.choice(idx_stim, 2000, replace=True) 
    unperturbed_code = perturb_1hot_temp[idx_ctrl]


    ######
    # now put it all together
    ######

    mu_slack, z_slack = res1_enc.predict([X_sc_ref, sample_code_pert, bulk_code, perturbed_code], batch_size=batch_size)
    z_concat = np.hstack([z_slack, sample_code_pert, bulk_code, perturbed_code])
    decoded_0_1 = res1_dec.predict(z_concat, batch_size=batch_size)
    decoded_0_1 = scaler.inverse_transform(decoded_0_1)


    mu_slack, z_slack = res1_enc.predict([X_sc_ref, sample_code_unpert, bulk_code, unperturbed_code], batch_size=batch_size)
    z_concat = np.hstack([z_slack, sample_code_unpert, bulk_code, unperturbed_code])
    decoded_0_0 = res1_dec.predict(z_concat, batch_size=batch_size)
    decoded_0_0 = scaler.inverse_transform(decoded_0_0)

    ######
    # now get DE genes
    ######

    top_genes = {}
    de_genes_all = None
    for curr_cell_type in Y_full.columns:


        # this is for the "projected" expression
        curr_idx = np.where(ctrl_test_meta_df.Y_max == curr_cell_type)[0]
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
    return (ctrl_test_meta_df, decoded_0_0, decoded_0_1, top_genes)



def calc_CVAE_perturbation_sample_specific(X_full, Y_full,
                            meta_df, encoder, decoder, 
                            scaler, batch_size, 
                            label_1hot_full, index_label, Label_full,
                            bulk_1hot_full, drug_1hot_full,
                            genes_ordered, top_lim=100):


    label_1hot_temp = np.copy(label_1hot_full)
    bulk_1hot_temp = np.copy(bulk_1hot_full)
    perturb_1hot_temp = np.copy(drug_1hot_full)


    # get the single cell data 
    idx_sc_ref = np.logical_and(meta_df.stim == "CTRL", meta_df.isTraining == "Train")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.samp_type == "sc_ref")
    idx_sc_ref = np.logical_and(idx_sc_ref, meta_df.cell_prop_type == "cell_type_specific")
    idx_sc_ref = np.where(idx_sc_ref)[0]
    idx_sc_ref = idx_sc_ref[range(1000)] # dont use the other sc ref

    ## this is to match up sample amounts across samples
    idx_sc_ref = np.tile(idx_sc_ref, 12) 



    X_sc_ref = np.copy(X_full)
    X_sc_ref = X_sc_ref[idx_sc_ref,]

    # get the sample_ids we will perturb
    sample_interest = ['1488', '1244', '1016', '101', '1039', '107']
    sample_code_idx = np.logical_and(meta_df.cell_prop_type == "cell_type_specific", 
                                        np.isin(meta_df.sample_id, sample_interest))
    sample_code_idx = np.where(sample_code_idx)[0]
    sample_code = label_1hot_temp[sample_code_idx]

    # make the metadata file
    ctrl_test_meta_df = meta_df.copy()
    ctrl_test_meta_df = ctrl_test_meta_df.iloc[idx_sc_ref]
    ctrl_test_meta_df.isTraining = "Test"
    ctrl_test_meta_df.stim = "CTRL"

    ctrl_test_meta_df.sample_id = index_label[Label_full][sample_code_idx]


    # get the bulk code
    idx_bulk = np.where(meta_df.samp_type == "bulk")[0]
    idx_bulk = np.random.choice(idx_bulk, 6000, replace=True) 
    idx_bulk = np.tile(idx_bulk, 2)
    bulk_code = bulk_1hot_temp[idx_bulk]

    #####
    # get (un)perturbed latent codes
    #####
    idx_stim = np.where(meta_df.stim == "STIM")[0][range(6000)]
    idx_stim = np.tile(idx_stim, 2)
    perturbed_code = perturb_1hot_temp[idx_stim]

    idx_ctrl = np.where(meta_df.stim == "CTRL")[0][range(6000)]
    idx_ctrl = np.tile(idx_ctrl, 2)
    unperturbed_code = perturb_1hot_temp[idx_ctrl]


    ######
    # now put it all together
    ######

    mu_slack, z_slack = encoder.predict([X_sc_ref, sample_code, bulk_code, perturbed_code], batch_size=batch_size)
    z_concat = np.hstack([z_slack, sample_code, bulk_code, perturbed_code])
    decoded_0_1 = decoder.predict(z_concat, batch_size=batch_size)
    decoded_0_1 = scaler.inverse_transform(decoded_0_1)


    mu_slack, z_slack = encoder.predict([X_sc_ref, sample_code, bulk_code, unperturbed_code], batch_size=batch_size)
    z_concat = np.hstack([z_slack, sample_code, bulk_code, unperturbed_code])
    decoded_0_0 = decoder.predict(z_concat, batch_size=batch_size)
    decoded_0_0 = scaler.inverse_transform(decoded_0_0)

    ######
    # now get DE genes
    ######

    top_genes = {}
    de_genes_all = None
    for curr_cell_type in Y_full.columns:


        # this is for the "projected" expression
        curr_idx = np.where(ctrl_test_meta_df.Y_max == curr_cell_type)[0]
        proj_ctrl = decoded_0_0[curr_idx]
        proj_stim = decoded_0_1[curr_idx]

        # take the median for nomalization

        proj_ctrl = np.median(rankdata(proj_ctrl, axis=1), axis=0)
        proj_stim = np.median(rankdata(proj_stim, axis=1), axis=0)
        #proj_ctrl = np.median(proj_ctrl, axis=0)
        #proj_stim = np.median(proj_stim, axis=0)
        proj_log2FC = np.abs(proj_stim-proj_ctrl)

        # make into DF
        proj_log2FC_df = pd.DataFrame(proj_log2FC, index=genes_ordered)

        intersect_proj = proj_log2FC_df.loc[genes_ordered][0]
        #top_proj_genes = intersect_proj.index[np.argsort(np.abs(intersect_proj))].tolist()[::-1][0:top_lim]
        top_proj_genes = intersect_proj.index[np.argsort(np.abs(intersect_proj))].tolist()[0:top_lim]

        top_genes[curr_cell_type] = top_proj_genes

    return (ctrl_test_meta_df, decoded_0_0, decoded_0_1, top_genes)



def train_cvae(res_data_path, exp_id, use_cvae4,
                n_tot_samples, n_drugs, n_tech, 
                X_cvae, label_cvae, bulk_cvae, drug_cvae,
                params: CVAETrainParameters=default_params):
    
    # set seeds
    from numpy.random import seed
    seed(1)
    from tensorflow.random import set_seed
    set_seed(2)

    n_x = X_cvae.shape[1]
    n_label = n_tot_samples


    ##################################################
    #####. Train Model first pass
    ##################################################
    if use_cvae4:
        cvae, encoder, decoder = cvae4.instantiate_model(
            n_x=n_x,
            n_label=n_label,
            n_drug=n_drugs,
            n_tech=n_tech,
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
    else:
        cvae, encoder, decoder = cvae3.instantiate_model(
            n_x=n_x,
            n_drug=n_drugs,
            n_tech=n_tech,
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
    idx_train = np.random.choice(range(X_cvae.shape[0]), np.ceil(X_cvae.shape[0]*0.8).astype(int), replace=False)
    idx_test = np.setdiff1d(range(X_cvae.shape[0]), idx_train)

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

    idx_train = np.random.choice(idx_train, train_size_samp*5, replace=True)
    idx_test = np.random.choice(idx_test, test_size_samp*5, replace=True)

    if use_cvae4:
        cvae_hist = cvae.fit([X_cvae[idx_train], 
                        label_cvae[idx_train], 
                        bulk_cvae[idx_train], 
                        drug_cvae[idx_train]], 
                        X_cvae[idx_train], verbose = 0, batch_size=params.batch_size, epochs=params.n_epoch,
                        validation_data=([X_cvae[idx_test], 
                                            label_cvae[idx_test], 
                                            bulk_cvae[idx_test], 
                                            drug_cvae[idx_test]], 
                                            X_cvae[idx_test]))
    else:
        cvae_hist = cvae.fit([X_cvae[idx_train], 
                    bulk_cvae[idx_train], 
                    drug_cvae[idx_train]], 
                    X_cvae[idx_train], verbose = 0, batch_size=params.batch_size, epochs=params.n_epoch,
                    validation_data=([X_cvae[idx_test], 
                                        bulk_cvae[idx_test], 
                                        drug_cvae[idx_test]], 
                                        X_cvae[idx_test]))
                                               
                                        
    loss_df = make_loss_df(cvae_hist)

    # plot loss
    loss_fig = make_loss_fig(loss_df)


    cvae.save(f"{res_data_path}/{exp_id}_cvae")
    encoder.save(f"{res_data_path}/{exp_id}_encoder")
    decoder.save(f"{res_data_path}/{exp_id}_decoder")

    


    return CVAETrainResults(
        cvae=cvae,
        encoder=encoder,
        decoder=decoder,
        loss_fig=loss_fig,
        output_folder=res_data_path,
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
