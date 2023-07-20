"""
This is the interface to initializing, training, plotting 
and estimating perturbation responses for BuDDI.
BuDDI calls the scripts buddi3 or buddi4, which contain the training
and models creating code. buddi3 or buddi4 only differ by the number of 
dimensions (other than the slack latent space) BuDDI would like to have, 3 or 4.

"""

import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from buddi.models import buddi4, buddi3
from buddi.preprocessing import sc_preprocess
from buddi.plotting import validation_plotting as vp

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
class BuddiTrainParameters:
    """
    Parameters for constructing a Buddi model.

    n_label_z: dimension of latent code for each non-y latent space

    encoder_dim: dim of encoder hidden layer
    decoder_dim: dim of decoder hidden layer
    class_dim1: for auxillary classifier first hidden dim
    class_dim2: for auxillary classifier first hidden dim
    batch_size: batch size (500)
    n_epoch: number of epochs
    alpha_rot: weight for sample classifier in loss
    alpha_prop: weight for cell type proportion classifier in loss
    alpha_bulk: weight for bulk vs sc classifier in loss
    alpha_drug: weight for perturbation classifier in loss
    beta_kl_slack: weight for KL term on slack
    beta_kl_rot: weight for KL term on sample latent space
    beta_kl_drug: weight for KL term on drug latent space
    beta_kl_bulk: weight for KL term on bulk vs sc latent space
    activ: activation function used in most nodes
    adam_learning_rate: ADAM learning rate

    """
    n_label_z: int = 64
    encoder_dim: int = 512
    decoder_dim: int = 512
    class_dim1: int = 512
    class_dim2: int = 256
    batch_size: int = 500
    n_epoch: int = 100
    alpha_rot: int = 100
    alpha_prop: int = 100
    alpha_bulk: int = 100
    alpha_drug: int = 100
    beta_kl_slack: int = 0.1
    beta_kl_rot: int = 100
    beta_kl_drug: int = 100
    beta_kl_bulk: int = 100
    activ: str = 'relu'
    adam_learning_rate: float = 0.0005

@dataclass
class BuddiTrainResults:
    """
    Parameters for constructing a Buddi model results.

    known_prop_vae: supervised VAE
    unknown_prop_vae: unsupervised VAE
    encoder_unlab: unsupervised encoder, all latent dimensions
    encoder_lab: supervised encoder, all latent dimensions
    decoder: decoder result
    classifier: cell type proportion estimator
    loss_fig: figure of training and a training-validation set
    spr_fig: figure of the spearman correlation between the true and estimated cell-type proportion 
    output_folder: Path to where all results will be written

    """
    known_prop_vae: Any
    unknown_prop_vae: Any
    encoder_unlab: Any
    encoder_lab: Any
    decoder: Any
    classifier: Any
    loss_fig: Any
    spr_fig: Any
    output_folder: Path

default_params = BuddiTrainParameters()


def _make_loss_df(loss_history, use_buddi4):

    max_idx = 5
    if use_buddi4:
        max_idx = 6

    # unpack the loss values
    labeled_total_loss = [item[0] for item in loss_history]
    unlabeled_total_loss = [item[max_idx][0] for item in loss_history]

    labeled_recon_loss = [item[1] for item in loss_history]
    unlabeled_recon_loss = [item[max_idx][1] for item in loss_history]

    labeled_prop_loss = [item[2] for item in loss_history]

    labeled_samp_loss = [item[3] for item in loss_history]
    unlabeled_samp_loss = [item[max_idx][2] for item in loss_history]


    if use_buddi4:
        labeled_drug_loss = [item[4] for item in loss_history]
        unlabeled_drug_loss = [item[max_idx][2] for item in loss_history]

        labeled_bulk_loss = [item[5] for item in loss_history]
        unlabeled_bulk_loss = [item[max_idx][2] for item in loss_history]
    else:
        labeled_bulk_loss = [item[4] for item in loss_history]
        unlabeled_bulk_loss = [item[max_idx][2] for item in loss_history]
       

    # cross validation LOSS make into a dataframe
    total_loss = labeled_total_loss + unlabeled_total_loss + [a + b for a, b in zip(labeled_total_loss, unlabeled_total_loss)]
    loss_df = pd.DataFrame(data=total_loss, columns=['total_loss'])
    loss_df['type'] = ["labeled"]*len(loss_history) + ["unlabeled"]*len(loss_history) + ["sum"]*len(loss_history)
    loss_df['batch'] = [*range(len(loss_history))] + [*range(len(loss_history))] + [*range(len(loss_history))]

    recon_loss = labeled_recon_loss + unlabeled_recon_loss + [a + b for a, b in zip(labeled_recon_loss, unlabeled_recon_loss)]
    loss_df['recon_loss'] = recon_loss

    prop_loss = labeled_prop_loss + [0]*len(loss_history) + labeled_prop_loss
    loss_df['prop_loss'] = prop_loss

    samp_loss = labeled_samp_loss + unlabeled_samp_loss + [a + b for a, b in zip(labeled_samp_loss, unlabeled_samp_loss)]
    loss_df['samp_loss'] = samp_loss

    bulk_loss = labeled_bulk_loss + unlabeled_bulk_loss + [a + b for a, b in zip(labeled_bulk_loss, unlabeled_bulk_loss)]
    loss_df['bulk_loss'] = bulk_loss

    if use_buddi4:
        drug_loss = labeled_drug_loss + unlabeled_drug_loss + [a + b for a, b in zip(labeled_drug_loss, unlabeled_drug_loss)]
        loss_df['drug_loss'] = drug_loss


    
    # add the log to make it easier to plot
    loss_df["log_total_loss"] = np.log10(loss_df["total_loss"]+1)
    loss_df["log_recon_loss"] = np.log10(loss_df["recon_loss"]+1)
    loss_df["log_samp_loss"] = np.log10(loss_df["samp_loss"]+1)
    loss_df["log_prop_loss"] = np.log10(loss_df["prop_loss"]+1)
    loss_df["log_bulk_loss"] = np.log10(loss_df["bulk_loss"]+1)

    if use_buddi4:
        loss_df["log_drug_loss"] = np.log10(loss_df["drug_loss"]+1)



    return(loss_df)

def make_loss_df(full_loss_history, use_buddi4):
    """
    Create a pandas dataframe from the loss history returned by training BuDDI

    full_loss_history: returned history from BuDDI
    use_buddi4: True if BuDDI was trained using 4 latent spaces, otherwise false (3 latent spaces)

    """
    cv_loss_history = full_loss_history[0]
    meta_history = full_loss_history[1]
    val_loss_history = full_loss_history[2]

    cv_loss_df = _make_loss_df(cv_loss_history, use_buddi4)
    val_loss_df = _make_loss_df(val_loss_history, use_buddi4)

    # format the hold-out spearman correlation
    meta_median = [np.median(x) for x in meta_history]
    meta_mean = [np.mean(x) for x in meta_history]

    meta_median_df = pd.DataFrame(meta_median)
    meta_median_df.columns = ["median_spr"]

    meta_median_df["mean_spr"] = meta_mean
    meta_median_df['batch'] = [*range(len(meta_history))]
    meta_median_df['type'] = "spearman"



    return (cv_loss_df, val_loss_df, meta_median_df)


def _make_loss_fig(loss_df, ax, title, loss_to_plot):
    ## plot loss
    g = sns.lineplot(
        x="batch", y=loss_to_plot,
        data=loss_df,
        hue="type",
        legend="full",
        alpha=0.3, ax= ax
    )
    
    title = f"{title} Final Loss sum: {np.round(loss_df[loss_to_plot].iloc[-1], 3)}"
    ax.set_title(title)
    return g

def make_loss_fig(cv_loss_df, val_loss_df, use_buddi4=True):

    """
    Plot the loss history returned by training BuDDI after converting to a pandas dataframe using make_loss_df

    cv_loss_df: cross validation returned history from BuDDI
    val_loss_df: held-out training validation set loss
    use_buddi4: True if BuDDI was trained using 4 latent spaces, otherwise false (3 latent spaces)

    """

    if use_buddi4:
        fig, axs = plt.subplots(6, 2, figsize=(15,25))

        _make_loss_fig(cv_loss_df, ax=axs[0, 0], title=f"Total Loss", loss_to_plot="log_total_loss")
        _make_loss_fig(cv_loss_df, ax=axs[1, 0], title=f"Recon Loss", loss_to_plot="log_recon_loss")
        _make_loss_fig(cv_loss_df, ax=axs[2, 0], title=f"Samp Loss", loss_to_plot="log_samp_loss")
        _make_loss_fig(cv_loss_df, ax=axs[3, 0], title=f"Prop Loss", loss_to_plot="log_prop_loss")
        _make_loss_fig(cv_loss_df, ax=axs[4, 0], title=f"Drug Loss", loss_to_plot="log_drug_loss")
        _make_loss_fig(cv_loss_df, ax=axs[5, 0], title=f"Bulk Loss", loss_to_plot="log_bulk_loss")

        _make_loss_fig(val_loss_df, ax=axs[0, 1], title=f"Val Total Loss", loss_to_plot="log_total_loss")
        _make_loss_fig(val_loss_df, ax=axs[1, 1], title=f"Val Recon Loss", loss_to_plot="log_recon_loss")
        _make_loss_fig(val_loss_df, ax=axs[2, 1], title=f"Val Samp Loss", loss_to_plot="log_samp_loss")
        _make_loss_fig(val_loss_df, ax=axs[3, 1], title=f"Val Prop Loss", loss_to_plot="log_prop_loss")
        _make_loss_fig(val_loss_df, ax=axs[4, 1], title=f"Val Drug Loss", loss_to_plot="log_drug_loss")
        _make_loss_fig(val_loss_df, ax=axs[5, 1], title=f"Val Bulk Loss", loss_to_plot="log_bulk_loss")

        fig.suptitle("Cross validation and hold-out validation Loss curves", fontsize=14)
    else:
        fig, axs = plt.subplots(6, 2, figsize=(15,25))

        _make_loss_fig(cv_loss_df, ax=axs[0, 0], title=f"Total Loss", loss_to_plot="log_total_loss")
        _make_loss_fig(cv_loss_df, ax=axs[1, 0], title=f"Recon Loss", loss_to_plot="log_recon_loss")
        _make_loss_fig(cv_loss_df, ax=axs[2, 0], title=f"Samp Loss", loss_to_plot="log_samp_loss")
        _make_loss_fig(cv_loss_df, ax=axs[3, 0], title=f"Prop Loss", loss_to_plot="log_prop_loss")
        _make_loss_fig(cv_loss_df, ax=axs[4, 0], title=f"Bulk Loss", loss_to_plot="log_bulk_loss")

        _make_loss_fig(val_loss_df, ax=axs[0, 1], title=f"Val Total Loss", loss_to_plot="log_total_loss")
        _make_loss_fig(val_loss_df, ax=axs[1, 1], title=f"Val Recon Loss", loss_to_plot="log_recon_loss")
        _make_loss_fig(val_loss_df, ax=axs[2, 1], title=f"Val Samp Loss", loss_to_plot="log_samp_loss")
        _make_loss_fig(val_loss_df, ax=axs[3, 1], title=f"Val Prop Loss", loss_to_plot="log_prop_loss")
        _make_loss_fig(val_loss_df, ax=axs[4, 1], title=f"Val Bulk Loss", loss_to_plot="log_bulk_loss")

        fig.suptitle("Cross validation and hold-out validation Loss curves", fontsize=14)
    return fig


def make_spearman_val_fig(spr_loss_df):

    """
    Plot the Spearman correlation between the true and estimated cell-type proportions over trainig epochs

    spr_loss_df: loss dataframe after converting to a pandas dataframe using make_loss_df

    """

    fig, axs = plt.subplots(2, figsize=(10,5))

    _make_loss_fig(spr_loss_df, ax=axs[0], title=f"Median Spr", loss_to_plot="median_spr")
    _make_loss_fig(spr_loss_df, ax=axs[1], title=f"Mean Spr", loss_to_plot="mean_spr")

    fig.suptitle("Median Spearman Corr. on held out set", fontsize=14)

    return fig

def plot_latent_spaces_buddi4(encoder_unlab, classifier,
        X_temp, Y_temp, label_temp, perturb_temp, bulk_temp, 
        batch_size, hide_sample_ids=False, alpha=1):


    """
    Plot PCA of latent spaces, colored by sources of variation, BuDDI is assumed to have 4 latent spaces

    encoder_unlab: BuDDI unsupervised encoder
    classifier: BuDDI cell-type proportion estimator 
    X_temp: Normalized gene expression, assumed in the same order as expected by BuDDI
    Y_temp: Cell type proportions
    label_temp: smaple IDs
    perturb_temp: perturbation IDs
    bulk_temp: array indicating bulk or sc 
    batch_size: batch size
    hide_sample_ids: plot the sample ids or not
    alpha: alpha value for points in plot  

    """


    z_slack, mu_slack, _, z_rot, mu_rot, _, z_drug, mu_drug, _, z_bulk, mu_bulk, _ = encoder_unlab.predict(X_temp, batch_size=batch_size)
    prop_outputs = classifier.predict(X_temp, batch_size=batch_size)

    # now concatenate together
    z_concat = np.hstack([z_slack, prop_outputs, z_rot, z_drug, z_bulk])


    fig, axs = plt.subplots(4, 5, figsize=(30,20))

    plot_df = vp.get_pca_for_plotting(np.asarray(prop_outputs))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,0], title="Cell Type", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,0], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,0], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,0], title="", alpha=alpha, legend_title="Perturbed")

    plot_df = vp.get_pca_for_plotting(np.asarray(mu_rot))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,1], title="Sample ID", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,1], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,1], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,1], title="", alpha=alpha, legend_title="Perturbed")

    plot_df = vp.get_pca_for_plotting(np.asarray(mu_bulk))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,2], title="Bulk vs SC", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,2], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,2], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,2], title="", alpha=alpha, legend_title="Perturbed")

    plot_df = vp.get_pca_for_plotting(np.asarray(mu_drug))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,3], title="Drug", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,3], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,3], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,3], title="", alpha=alpha, legend_title="Perturbed")


    plot_df = vp.get_pca_for_plotting(np.asarray(mu_slack))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,4], title="Slack", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,4], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,4], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,4], title="", alpha=alpha, legend_title="Perturbed")


    count_idx = 0
    for ax1 in axs:
        inner_count_idx = 0
        for ax2 in ax1:
            if (count_idx == 1 and hide_sample_ids) or inner_count_idx < 4:
                ax2.get_legend().remove()
            inner_count_idx = inner_count_idx + 1

        count_idx = count_idx + 1

    return fig


def plot_latent_spaces_buddi4_umap(encoder_unlab, classifier,
        X_temp, Y_temp, label_temp, perturb_temp, bulk_temp, 
        batch_size, hide_sample_ids=False, alpha=1):

    """
    Plot UMAP of latent spaces, colored by sources of variation, BuDDI is assumed to use 4 latent spaces

    encoder_unlab: BuDDI unsupervised encoder
    classifier: BuDDI cell-type proportion estimator 
    X_temp: Normalized gene expression, assumed in the same order as expected by BuDDI
    Y_temp: Cell type proportions
    label_temp: smaple IDs
    perturb_temp: perturbation IDs
    bulk_temp: array indicating bulk or sc 
    batch_size: batch size
    hide_sample_ids: plot the sample ids or not
    alpha: alpha value for points in plot  

    """


    z_slack, mu_slack, _, z_rot, mu_rot, _, z_drug, mu_drug, _, z_bulk, mu_bulk, _ = encoder_unlab.predict(X_temp, batch_size=batch_size)
    prop_outputs = classifier.predict(X_temp, batch_size=batch_size)

    # now concatenate together
    z_concat = np.hstack([z_slack, prop_outputs, z_rot, z_drug, z_bulk])


    fig, axs = plt.subplots(4, 5, figsize=(30,20))

    plot_df = vp.get_umap_for_plotting(np.asarray(prop_outputs))
    vp.plot_umap(plot_df, color_vec=Y_temp, ax=axs[0,0], title="Cell Type", alpha=alpha, legend_title="Cell Type")
    vp.plot_umap(plot_df, color_vec=label_temp, ax=axs[1,0], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_umap(plot_df, color_vec=bulk_temp, ax=axs[2,0], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_umap(plot_df, color_vec=perturb_temp, ax=axs[3,0], title="", alpha=alpha, legend_title="Perturbed")

    plot_df = vp.get_umap_for_plotting(np.asarray(mu_rot))
    vp.plot_umap(plot_df, color_vec=Y_temp, ax=axs[0,1], title="Sample ID", alpha=alpha, legend_title="Cell Type")
    vp.plot_umap(plot_df, color_vec=label_temp, ax=axs[1,1], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_umap(plot_df, color_vec=bulk_temp, ax=axs[2,1], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_umap(plot_df, color_vec=perturb_temp, ax=axs[3,1], title="", alpha=alpha, legend_title="Perturbed")

    plot_df = vp.get_umap_for_plotting(np.asarray(mu_bulk))
    vp.plot_umap(plot_df, color_vec=Y_temp, ax=axs[0,2], title="Bulk vs SC", alpha=alpha, legend_title="Cell Type")
    vp.plot_umap(plot_df, color_vec=label_temp, ax=axs[1,2], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_umap(plot_df, color_vec=bulk_temp, ax=axs[2,2], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_umap(plot_df, color_vec=perturb_temp, ax=axs[3,2], title="", alpha=alpha, legend_title="Perturbed")

    plot_df = vp.get_umap_for_plotting(np.asarray(mu_drug))
    vp.plot_umap(plot_df, color_vec=Y_temp, ax=axs[0,3], title="Drug", alpha=alpha, legend_title="Cell Type")
    vp.plot_umap(plot_df, color_vec=label_temp, ax=axs[1,3], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_umap(plot_df, color_vec=bulk_temp, ax=axs[2,3], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_umap(plot_df, color_vec=perturb_temp, ax=axs[3,3], title="", alpha=alpha, legend_title="Perturbed")


    plot_df = vp.get_umap_for_plotting(np.asarray(mu_slack))
    vp.plot_umap(plot_df, color_vec=Y_temp, ax=axs[0,4], title="Slack", alpha=alpha, legend_title="Cell Type")
    vp.plot_umap(plot_df, color_vec=label_temp, ax=axs[1,4], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_umap(plot_df, color_vec=bulk_temp, ax=axs[2,4], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_umap(plot_df, color_vec=perturb_temp, ax=axs[3,4], title="", alpha=alpha, legend_title="Perturbed")


    count_idx = 0
    for ax1 in axs:
        inner_count_idx = 0
        for ax2 in ax1:
            if (count_idx == 1 and hide_sample_ids) or inner_count_idx < 4:
                ax2.get_legend().remove()
            inner_count_idx = inner_count_idx + 1

        count_idx = count_idx + 1

    return fig


def plot_latent_spaces_buddi3(encoder_unlab, classifier,
        X_temp, Y_temp, label_temp, perturb_temp, bulk_temp, 
        batch_size, alpha=1):


    """
    Plot PCA of latent spaces, colored by sources of variation, BuDDI is assumed to have 3 latent spaces

    encoder_unlab: BuDDI unsupervised encoder
    classifier: BuDDI cell-type proportion estimator 
    X_temp: Normalized gene expression, assumed in the same order as expected by BuDDI
    Y_temp: Cell type proportions
    label_temp: smaple IDs
    perturb_temp: perturbation IDs
    bulk_temp: array indicating bulk or sc 
    batch_size: batch size
    alpha: alpha value for points in plot  

    """

    z_slack, mu_slack, _, z_rot, mu_rot, _, z_bulk, mu_bulk, _ = encoder_unlab.predict(X_temp, batch_size=batch_size)
    prop_outputs = classifier.predict(X_temp, batch_size=batch_size)

    # now concatenate together
    z_concat = np.hstack([z_slack, prop_outputs, z_rot, z_bulk])


    fig, axs = plt.subplots(4, 4, figsize=(30,20))

    plot_df = vp.get_pca_for_plotting(np.asarray(prop_outputs))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,0], title="Cell Type", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,0], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,0], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,0], title="", alpha=alpha, legend_title="Perturbed")

    plot_df = vp.get_pca_for_plotting(np.asarray(mu_rot))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,1], title="Sample ID", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,1], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,1], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,1], title="", alpha=alpha, legend_title="Perturbed")


    plot_df = vp.get_pca_for_plotting(np.asarray(mu_bulk))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,2], title="Bulk vs SC", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,2], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,2], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,2], title="", alpha=alpha, legend_title="Perturbed")


    plot_df = vp.get_pca_for_plotting(np.asarray(mu_slack))
    vp.plot_pca(plot_df, color_vec=Y_temp, ax=axs[0,3], title="Slack", alpha=alpha, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_temp, ax=axs[1,3], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=bulk_temp, ax=axs[2,3], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_pca(plot_df, color_vec=perturb_temp, ax=axs[3,3], title="", alpha=alpha, legend_title="Perturbed")



    count_idx = 0
    for ax1 in axs:
        inner_count_idx = 0
        for ax2 in ax1:
            if inner_count_idx < 3:
                ax2.get_legend().remove()
            inner_count_idx = inner_count_idx + 1

        count_idx = count_idx + 1

    return fig


def plot_latent_spaces_buddi3_umap(encoder_unlab, classifier,
        X_temp, Y_temp, label_temp, perturb_temp, bulk_temp, 
        batch_size, alpha=1):


    """
    Plot TSNE of latent spaces, colored by sources of variation, BuDDI is assumed to have 3 latent spaces

    encoder_unlab: BuDDI unsupervised encoder
    classifier: BuDDI cell-type proportion estimator 
    X_temp: Normalized gene expression, assumed in the same order as expected by BuDDI
    Y_temp: Cell type proportions
    label_temp: smaple IDs
    perturb_temp: perturbation IDs
    bulk_temp: array indicating bulk or sc 
    batch_size: batch size
    alpha: alpha value for points in plot  

    """

    z_slack, mu_slack, _, z_rot, mu_rot, _, z_bulk, mu_bulk, _ = encoder_unlab.predict(X_temp, batch_size=batch_size)
    prop_outputs = classifier.predict(X_temp, batch_size=batch_size)

    # now concatenate together
    z_concat = np.hstack([z_slack, prop_outputs, z_rot, z_bulk])

    fig, axs = plt.subplots(4, 4, figsize=(30,20))

    plot_df = vp.get_tsne_for_plotting(np.asarray(prop_outputs))
    vp.plot_tsne(plot_df, color_vec=Y_temp, ax=axs[0,0], title="Cell Type", alpha=alpha, legend_title="Cell Type")
    vp.plot_tsne(plot_df, color_vec=label_temp, ax=axs[1,0], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_tsne(plot_df, color_vec=bulk_temp, ax=axs[2,0], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_tsne(plot_df, color_vec=perturb_temp, ax=axs[3,0], title="", alpha=alpha, legend_title="Perturbed")

    plot_df = vp.get_tsne_for_plotting(np.asarray(mu_rot))
    vp.plot_tsne(plot_df, color_vec=Y_temp, ax=axs[0,1], title="Sample ID", alpha=alpha, legend_title="Cell Type")
    vp.plot_tsne(plot_df, color_vec=label_temp, ax=axs[1,1], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_tsne(plot_df, color_vec=bulk_temp, ax=axs[2,1], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_tsne(plot_df, color_vec=perturb_temp, ax=axs[3,1], title="", alpha=alpha, legend_title="Perturbed")


    plot_df = vp.get_tsne_for_plotting(np.asarray(mu_bulk))
    vp.plot_tsne(plot_df, color_vec=Y_temp, ax=axs[0,2], title="Bulk vs SC", alpha=alpha, legend_title="Cell Type")
    vp.plot_tsne(plot_df, color_vec=label_temp, ax=axs[1,2], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_tsne(plot_df, color_vec=bulk_temp, ax=axs[2,2], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_tsne(plot_df, color_vec=perturb_temp, ax=axs[3,2], title="", alpha=alpha, legend_title="Perturbed")


    plot_df = vp.get_tsne_for_plotting(np.asarray(mu_slack))
    vp.plot_tsne(plot_df, color_vec=Y_temp, ax=axs[0,3], title="Slack", alpha=alpha, legend_title="Cell Type")
    vp.plot_tsne(plot_df, color_vec=label_temp, ax=axs[1,3], title="", alpha=alpha, legend_title="Sample ID")
    vp.plot_tsne(plot_df, color_vec=bulk_temp, ax=axs[2,3], title="", alpha=alpha, legend_title="Bulk vs SC")
    vp.plot_tsne(plot_df, color_vec=perturb_temp, ax=axs[3,3], title="", alpha=alpha, legend_title="Perturbed")


    count_idx = 0
    for ax1 in axs:
        inner_count_idx = 0
        for ax2 in ax1:
            if inner_count_idx < 3:
                ax2.get_legend().remove()
            inner_count_idx = inner_count_idx + 1

        count_idx = count_idx + 1

    return fig


def plot_latent_spaces(encoder_unlab, classifier, 
        X_temp, Y_temp, label_temp, perturb_temp, bulk_temp=None, 
        batch_size=500, use_buddi4=True, use_pca=True, hide_sample_ids=False,
        alpha=1):
    

    """
    Plot PCA or TSNE of latent spaces, colored by sources of variation

    encoder_unlab: BuDDI unsupervised encoder
    classifier: BuDDI cell-type proportion estimator 
    X_temp: Normalized gene expression, assumed in the same order as expected by BuDDI
    Y_temp: Cell type proportions
    label_temp: smaple IDs
    perturb_temp: perturbation IDs
    bulk_temp: array indicating bulk or sc 
    batch_size: batch size
    use_buddi4: Whether BuDDI has 3 or 4 latent spaces
    use_pca: whether to use PCA or TSNE
    hide_sample_ids: hide sample IDs during plotting
    alpha: alpha value for points in plot  

    """

    if use_pca:
        if use_buddi4:
            res = plot_latent_spaces_buddi4(encoder_unlab, classifier, X_temp, Y_temp, label_temp, perturb_temp, bulk_temp, batch_size, hide_sample_ids, alpha=alpha)
        else:
            res = plot_latent_spaces_buddi3(encoder_unlab, classifier, X_temp, Y_temp, label_temp, perturb_temp, bulk_temp, batch_size, alpha=alpha)
    else:
        if use_buddi4:
            res = plot_latent_spaces_buddi4_umap(encoder_unlab, classifier, X_temp, Y_temp, label_temp, perturb_temp, bulk_temp, batch_size, hide_sample_ids, alpha=alpha)
        else:
            res = plot_latent_spaces_buddi3_umap(encoder_unlab, classifier, X_temp, Y_temp, label_temp, perturb_temp, bulk_temp, batch_size, alpha=alpha)


    return res

def plot_reconstruction_buddi(encoder_unlab, classifier, decoder,
        X_temp, Y_temp, label_temp, perturb_temp, 
        batch_size=500, use_buddi4=True):


    """
    Plot reconstruction results of BuDDI

    encoder_unlab: BuDDI unsupervised encoder
    classifier: BuDDI cell-type proportion estimator 
    decoder: BuDDI decoder
    X_temp: Normalized gene expression, assumed in the same order as expected by BuDDI
    Y_temp: Cell type proportions
    label_temp: smaple IDs
    perturb_temp: perturbation IDs
    bulk_temp: array indicating bulk or sc 
    batch_size: batch size
    use_buddi4: Whether BuDDI has 3 or 4 latent spaces

    """


    prop_outputs = classifier.predict(X_temp, batch_size=batch_size)

    # now use the encoder to get the latent spaces
    if use_buddi4:
        z_slack, mu_slack, l_sigma_slack, z_rot, mu_rot, l_sigma_rot, z_drug, mu_drug, l_sigma_drug, z_bulk, mu_bulk, l_sigma_bulk = encoder_unlab.predict(X_temp, batch_size=batch_size)
    
        # now concatenate together
        z_concat = np.hstack([z_slack, prop_outputs, z_rot, z_drug, z_bulk])
    else:
        z_slack, mu_slack, l_sigma_slack, z_rot, mu_rot, l_sigma_rot, z_drug, mu_drug, l_sigma_drug = encoder_unlab.predict(X_temp, batch_size=batch_size)

        # now concatenate together
        z_concat = np.hstack([z_slack, prop_outputs, z_rot, z_drug])



    # and decode
    decoded_outputs = decoder.predict(z_concat, batch_size=batch_size)

    # combine the true output and the reconstruction
    X_dup = np.vstack([X_temp, decoded_outputs])


    Y_dup = np.append(Y_temp, Y_temp)

    label_dup = np.append(label_temp, label_temp)
    perturb_dup = np.append(perturb_temp, perturb_temp)
    source_dup = np.asarray(np.append([0]*label_temp.shape[0], [1]*label_temp.shape[0]))
    source_dup_str = ["Recon" if x  else "Orig" for x in source_dup ]

    fig, axs = plt.subplots(1, 3, figsize=(30,5))

    plot_df = vp.get_pca_for_plotting(np.asarray(X_dup))
    vp.plot_pca(plot_df, color_vec=Y_dup, ax=axs[0], title="", alpha=1, legend_title="Cell Type")
    vp.plot_pca(plot_df, color_vec=label_dup, ax=axs[1], title="", alpha=1, legend_title="Sample ID")
    vp.plot_pca(plot_df, color_vec=source_dup_str, ax=axs[2], title="", alpha=1, legend_title="Data Source")


    fig.suptitle("Reconstructed and Original Training Data", fontsize=14)
    axs[1].legend([],[], frameon=False)

    return fig


def calc_buddi_perturbation_sample_specific(meta_df, X_full, Y_full, sample_interest, scaler, 
                            encoder_unlab, decoder, batch_size, 
                            genes_ordered, top_lim=100, use_buddi4=True):

    """
    Estimate sample-specific perturbation response for BuDDI

    meta_df: pandas dataframe of the metadata associated with X_full and Y_full
    X_full: Normalized gene expression, assumed in the same order as expected by BuDDI
    Y_full: Cell type proportions
    sample_interest: that sample ID to perturb
    encoder_unlab: BuDDI unsupervised encoder
    decoder: BuDDI decoder
    batch_size: batch size
    genes_ordered: the gene IDs matching the genes in X_full, which should also be in the same order as BuDDI
    top_lim: how many genes to report as top perturbed
    use_buddi4: Whether BuDDI has 3 or 4 latent spaces

    """


    tot_simulated_sample = Y_full.shape[1]*len(sample_interest)*100

    X_temp = np.copy(X_full)

    #####
    # get cell type latent codes
    #####
    # the codes for cell type proportion
    # and tile to repeat for each sample
    sc_props = sc_preprocess.get_single_celltype_prop_matrix(num_samp=100, cell_order=Y_full.columns)
    sc_props = np.tile(sc_props, (len(sample_interest),1))


    #####
    # get perturbation latent codes
    #####

    # get the index to get the perturbation latent codes
    pert_code_idx = np.logical_and(meta_df.isTraining == "Train", meta_df.stim == "STIM")
    pert_code_idx = np.where(pert_code_idx)[0] 
    pert_code_idx = np.random.choice(pert_code_idx, tot_simulated_sample, replace=True)

    if use_buddi4:
        z_slack, _, _, z_rot, _, _, z_pert, _, _, z_bulk, _, _ = encoder_unlab.predict(X_temp[pert_code_idx,], batch_size=batch_size)
    else:
        z_slack, _, _, z_rot, _, _, z_pert, _, _ = encoder_unlab.predict(X_temp[pert_code_idx,], batch_size=batch_size)

    # get the index to get the UNperturbed latent codes
    unpert_code_idx = np.logical_and(meta_df.isTraining == "Train", meta_df.stim == "CTRL")
    unpert_code_idx = np.where(unpert_code_idx)[0] 
    unpert_code_idx = np.random.choice(unpert_code_idx, tot_simulated_sample, replace=True)
    if use_buddi4:
        _, _, _, _, _, _, z_unpert, _, _, _, _, _ = encoder_unlab.predict(X_temp[unpert_code_idx,], batch_size=batch_size)
    else:
        _, _, _, _, _, _, z_unpert, _, _ = encoder_unlab.predict(X_temp[unpert_code_idx,], batch_size=batch_size)

    #####
    # get sample latent codes
    #####
    # get the index to get the sample latent codes
    sample_code_idx = np.logical_and(meta_df.isTraining == "Train", 
                                        np.isin(meta_df.sample_id, sample_interest))
    sample_code_idx = np.logical_and(sample_code_idx, meta_df.stim == "CTRL")
    sample_code_idx = np.where(sample_code_idx)[0] 
    sample_code_idx = np.repeat(sample_code_idx, 100)

    if use_buddi4:
        _, _, _, z_samples, _, _, _, _, _, _, _, _ = encoder_unlab.predict(X_temp[sample_code_idx,], batch_size=batch_size)
    else:
        _, _, _, z_samples, _, _, _, _, _ = encoder_unlab.predict(X_temp[sample_code_idx,], batch_size=batch_size)

    # make the metadata table 
    temp_meta_df = meta_df.iloc[sample_code_idx]
    temp_meta_df.isTraining = "Test"
    temp_meta_df.cell_prop_type = "cell_type_specific"

    prop_max = np.copy(sc_props)
    prop_max = np.argmax(prop_max, axis=1)
    prop_max = Y_full.columns[prop_max]
    temp_meta_df.Y_max = prop_max


    ######
    # now put it all together
    ######

    # now concatenate together and add the stim codes to the latent
    if use_buddi4:
        z_concat_perturb = np.hstack([z_slack, sc_props, z_samples, z_pert, z_bulk])
    else:
        z_concat_perturb = np.hstack([z_slack, sc_props, z_samples, z_bulk])
    decoded_0_1 = decoder.predict(z_concat_perturb, batch_size=batch_size)
    decoded_0_1 = scaler.inverse_transform(decoded_0_1)

    # now concatenate together and add the stim codes to the latent
    if use_buddi4:
        z_concat_unperturb = np.hstack([z_slack, sc_props, z_samples, z_unpert, z_bulk])
    else:
        z_concat_unperturb = np.hstack([z_slack, sc_props, z_samples, z_bulk])

    decoded_0_0 = decoder.predict(z_concat_unperturb, batch_size=batch_size)
    decoded_0_0 = scaler.inverse_transform(decoded_0_0)

    ######
    # now get the differential genes
    ######


    top_genes = {}
    de_genes_all = None
    for curr_cell_type in Y_full.columns:


        # this is for the "projected" expression
        curr_idx = np.where(temp_meta_df.Y_max == curr_cell_type)[0]
        proj_ctrl = decoded_0_0[curr_idx]
        proj_stim = decoded_0_1[curr_idx]

        # take the median for normalization

        proj_ctrl = np.median(rankdata(proj_ctrl, axis=1), axis=0)
        proj_stim = np.median(rankdata(proj_stim, axis=1), axis=0)
        proj_log2FC = np.abs(proj_stim-proj_ctrl)

        # make into DF
        proj_log2FC_df = pd.DataFrame(proj_log2FC, index=genes_ordered)

        intersect_proj = proj_log2FC_df.loc[genes_ordered][0]
        top_proj_genes = intersect_proj.index[np.argsort(np.abs(intersect_proj))].tolist()[::-1][0:top_lim]

        top_genes[curr_cell_type] = top_proj_genes



    return (temp_meta_df, decoded_0_0, decoded_0_1, top_genes)



def calc_buddi_perturbation(meta_df, X_full, Y_full, scaler, 
                            encoder_unlab, decoder, batch_size, 
                            genes_ordered, top_lim=100, use_buddi4=True):
 
    """
    Plot general perturbation response for BuDDI. 
    We couple all latents spaces except for the perturbation effect during sampling.

    meta_df: pandas dataframe of the metadata associated with X_full and Y_full
    X_full: Normalized gene expression, assumed in the same order as expected by BuDDI
    Y_full: Cell type proportions
    scaler: the scaler used during normalization
    encoder_unlab: BuDDI unsupervised encoder
    decoder: BuDDI decoder
    batch_size: batch size
    genes_ordered: the gene IDs matching the genes in X_full, which should also be in the same order as BuDDI
    top_lim: how many genes to report as top perturbed
    use_buddi4: Whether BuDDI has 3 or 4 latent spaces

    """

    # get the training data
    # so we can use it to get the latent codes

    # get the points we are interested in
    # we will use their codes and only change the perturbation codes
    subset_idx = np.logical_and(meta_df.isTraining == "Train", meta_df.cell_prop_type == "cell_type_specific")
    subset_idx = np.logical_and(subset_idx, meta_df.stim == "CTRL")
    subset_idx = np.where(subset_idx)[0] 
    subset_idx = np.random.choice(subset_idx, 8000, replace=True)

    temp_meta_df = meta_df.iloc[subset_idx]

    X_temp = np.copy(X_full)
    X_unpert = X_temp[subset_idx,]

    Y_temp = np.copy(Y_full)
    Y_temp = Y_temp[subset_idx]


    #####
    # get unperturbed latent codes
    #####
    if use_buddi4:
        _, _, _, _, _, _, z_unpert, _, _, _, _, _  = encoder_unlab.predict(X_unpert, batch_size=batch_size)
    else:
        _, _, _, _, _, _, z_unpert, _, _  = encoder_unlab.predict(X_unpert, batch_size=batch_size)

    #####
    # get perturbation latent codes
    #####

    # get the index to get the perturbation latent codes
    pert_code_idx = np.logical_and(meta_df.isTraining == "Train", meta_df.stim == "STIM")
    pert_code_idx = np.where(pert_code_idx)[0] 
    pert_code_idx = np.random.choice(pert_code_idx, len(subset_idx), replace=True)
    if use_buddi4:
        z_slack, _, _, z_samples, _, _, z_pert, _, _, z_bulk, _, _  = encoder_unlab.predict(X_temp[pert_code_idx,], batch_size=batch_size)
    else:
        z_slack, _, _, z_samples, _, _, z_bulk, _, _  = encoder_unlab.predict(X_temp[pert_code_idx,], batch_size=batch_size)


    # make the metadata table 
    temp_meta_df = meta_df.iloc[subset_idx]
    temp_meta_df.isTraining = "Test"
    temp_meta_df.cell_prop_type = "cell_type_specific"


    ######
    # now put it all together
    ######

    # now concatenate together and add the stim codes to the latent
    if use_buddi4:
        z_concat_perturb = np.hstack([z_slack, Y_temp, z_samples, z_pert, z_bulk])
    else:
        z_concat_perturb = np.hstack([z_slack, Y_temp, z_samples, z_bulk])
    decoded_0_1 = decoder.predict(z_concat_perturb, batch_size=batch_size)
    decoded_0_1 = scaler.inverse_transform(decoded_0_1)

    # now concatenate together and add the stim codes to the latent
    if use_buddi4:
        z_concat_unperturb = np.hstack([z_slack, Y_temp, z_samples, z_unpert, z_bulk])
    else:
        z_concat_unperturb = np.hstack([z_slack, Y_temp, z_samples, z_bulk])

    decoded_0_0 = decoder.predict(z_concat_unperturb, batch_size=batch_size)
    decoded_0_0 = scaler.inverse_transform(decoded_0_0)

    ######
    # now get the differential genes
    ######


    top_genes = {}
    de_genes_all = None
    for curr_cell_type in Y_full.columns:


        # this is for the "projected" expression
        curr_idx = np.where(temp_meta_df.Y_max == curr_cell_type)[0]
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



    return (temp_meta_df, decoded_0_0, decoded_0_1, top_genes)


def train_buddi(res_data_path, exp_id, use_buddi4, 
                n_tot_samples, n_drugs, n_tech, 
                X_unkp, label_unkp, drug_unkp, bulk_unkp,
                X_kp, y_kp, label_kp, drug_kp, bulk_kp,
                params: BuddiTrainParameters=default_params):
    

    """
    Train BuDDI

    res_data_path: path to write out the trained models
    exp_id: experiment ID used in generating pseudobulks
    use_buddi4: Whether BuDDI has 3 or 4 latent spaces
    n_tot_samples: number of samples
    n_drugs: number of drug states (length of 1-hot encoded vector of drug IDs)
    n_tech: number of technologies, sc+bulk = 2
    X_unkp: gene expression of bulks (unknown proportions)
    label_unkp: bulk sample IDs
    drug_unkp: bulk drug status 
    bulk_unkp: bulk IDs ( should always be 1)
    X_kp: gene expression of pseudobulks (known proportions)
    y_kp: cell type proportions of pseudobulks
    label_kp: sample IDs of pseudobulks
    drug_kp: drug status of pseudobulks
    bulk_kp: bulk IDs (should always be 0)
    params: remaining BuddiTrainParameters

    """


    # set seeds
    from numpy.random import seed
    seed(1)
    from tensorflow.random import set_seed
    set_seed(2)

    n_x = X_unkp.shape[1]
    n_y = y_kp.shape[1]
    n_label = n_tot_samples

    ##################################################
    #####. Train Model first pass
    ##################################################
    if use_buddi4:
        known_prop_vae, unknown_prop_vae, encoder_unlab, encoder_lab, decoder, classifier = buddi4.instantiate_model(
            n_x,
            n_y,
            n_label,
            n_drugs,
            n_tech,
            n_label_z = params.n_label_z, 
            encoder_dim = params.encoder_dim, 
            decoder_dim = params.decoder_dim, 
            class_dim1 = params.class_dim1, 
            class_dim2 = params.class_dim2, 
            batch_size = params.batch_size, 
            n_epoch = params.n_epoch, 
            alpha_rot = params.alpha_rot,  
            alpha_drug = params.alpha_drug,  
            alpha_bulk = params.alpha_bulk,  
            alpha_prop = params.alpha_prop,  
            beta_kl_slack = params.beta_kl_slack, 
            beta_kl_rot = params.beta_kl_rot,
            beta_kl_drug = params.beta_kl_drug,
            beta_kl_bulk = params.beta_kl_bulk,
            activ = params.activ, 
            optim = tf.keras.optimizers.legacy.Adam(learning_rate=params.adam_learning_rate)
        )


        full_loss_history = buddi4.fit_model(known_prop_vae, unknown_prop_vae,
            encoder_unlab, encoder_lab, decoder, classifier,
            X_unkp, label_unkp, drug_unkp, bulk_unkp,
            X_kp, y_kp,label_kp, drug_kp, bulk_kp,
            epochs=params.n_epoch, batch_size=params.batch_size)
    else:
        known_prop_vae, unknown_prop_vae, encoder_unlab, encoder_lab, decoder, classifier = buddi3.instantiate_model(
            n_x,
            n_y,
            n_label,
            n_tech,
            n_label_z = params.n_label_z, 
            encoder_dim = params.encoder_dim, 
            decoder_dim = params.decoder_dim, 
            class_dim1 = params.class_dim1, 
            class_dim2 = params.class_dim2, 
            batch_size = params.batch_size, 
            n_epoch = params.n_epoch, 
            alpha_rot = params.alpha_rot,  
            alpha_bulk = params.alpha_bulk,  
            alpha_prop = params.alpha_prop,  
            beta_kl_slack = params.beta_kl_slack, 
            beta_kl_rot = params.beta_kl_rot,
            beta_kl_bulk = params.beta_kl_bulk,
            activ = params.activ, 
            optim = tf.keras.optimizers.legacy.Adam(learning_rate=params.adam_learning_rate)
        )


        full_loss_history = buddi3.fit_model(known_prop_vae, unknown_prop_vae,
            encoder_unlab, encoder_lab, decoder, classifier,
            X_unkp, label_unkp, bulk_unkp,
            X_kp, y_kp,label_kp, bulk_kp,
            epochs=params.n_epoch, batch_size=params.batch_size)

    cv_loss, val_loss, spr_loss = make_loss_df(full_loss_history, use_buddi4)

    # write loss out
    loss_file = os.path.join(res_data_path, f"{exp_id}-cv_loss.pkl")
    cv_loss.to_pickle(loss_file)
    loss_file = os.path.join(res_data_path, f"{exp_id}-val_loss.pkl")
    val_loss.to_pickle(loss_file)

    # plot loss
    loss_fig = make_loss_fig(cv_loss, val_loss, use_buddi4)
    spr_fig = make_spearman_val_fig(spr_loss)


    known_prop_vae.save(f"{res_data_path}/{exp_id}_known_prop_vae")
    unknown_prop_vae.save(f"{res_data_path}/{exp_id}_unknown_prop_vae")
    encoder_unlab.save(f"{res_data_path}/{exp_id}_encoder_unlab")
    encoder_lab.save(f"{res_data_path}/{exp_id}_encoder_lab")
    decoder.save(f"{res_data_path}/{exp_id}_decoder")
    classifier.save(f"{res_data_path}/{exp_id}_classifier")

    


    return BuddiTrainResults(
        known_prop_vae=known_prop_vae,
        unknown_prop_vae=unknown_prop_vae,
        encoder_unlab=encoder_unlab,
        encoder_lab=encoder_lab,
        decoder=decoder,
        classifier=classifier,
        loss_fig=loss_fig,
        spr_fig=spr_fig,
        output_folder=res_data_path,
    )


# ===============================================================
# === simulating
# ===============================================================

@dataclass
class BuddiSimulateParameters:
    pass

def simulate_perturbations(params: BuddiSimulateParameters):
    pass

