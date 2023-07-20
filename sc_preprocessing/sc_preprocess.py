#!/usr/bin/env python

# general imports
import warnings
import numpy as np
import os
import pandas as pd
import sklearn as sk
import scipy as sp
from scipy.sparse import coo_matrix

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

# file processing
import pickle
import gzip
from pathlib import Path


# cell type specific pseudobulk
def get_cell_type_sum(in_adata, cell_type_id, num_samples):

  # get the expression of the cells of interest
  cell_df = in_adata[in_adata.obs["scpred_CellType"].isin([cell_type_id])]

  # if there is none of the cell type, 
  # just get the first elements and we will return zero
  mult_by_zero = False
  if len(cell_df) == 0:
    cell_df = in_adata[0:2]
    mult_by_zero = True

  # now to the sampling
  cell_sample = sk.utils.resample(cell_df, n_samples = num_samples, replace=True)

  # add  poisson noise
  #dense_X = cell_sample.X.todense()
  #noise_mask = np.random.poisson(dense_X+1)
  #dense_X = dense_X + noise_mask

  sum_per_gene = cell_sample.X.sum(axis=0)

  # set to zero
  if mult_by_zero:
    sum_per_gene = sum_per_gene*0


  return sum_per_gene

# method to generate a proportion vector
def gen_prop_vec_lognormal(len_vector, num_cells):

  rand_vec = np.random.lognormal(5, np.random.uniform(1,3), len_vector) # 1

  rand_vec = np.round((rand_vec/np.sum(rand_vec))*num_cells)
  if(np.sum(rand_vec) != num_cells):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_cells - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)
  
  return rand_vec

# method to generate true proportion vector
def true_prop_vec(in_adata, num_cells):

  rand_vec = in_adata.obs["scpred_CellType"].value_counts() / in_adata.obs["scpred_CellType"].shape[0]
  rand_vec = np.array(rand_vec)

  rand_vec = np.round(rand_vec*num_cells)
  if(np.sum(rand_vec) != num_cells):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_cells - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)
  
  return rand_vec

# total pseudobulk
def make_prop_and_sum(in_adata, num_samples, num_cells, use_true_prop, cell_noise, useSampleNoise=True):
  len_vector = in_adata.obs["scpred_CellType"].unique().shape[0]

  # instantiate the expression and proportion vectors
  total_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  total_prop = pd.DataFrame(columns = in_adata.obs["scpred_CellType"].unique())

  test_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  test_prop = pd.DataFrame(columns = in_adata.obs["scpred_CellType"].unique())


  # cell specific noise, new noise for each sample
  if cell_noise == None:
    cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for i in range(len_vector)]

  # iterate over all the samples we would like to make
  for samp_idx in range(num_samples+100):
    if samp_idx % 100 == 0:
      print(samp_idx)

    n_cells = num_cells
    if num_cells is None:
      n_cells = np.random.uniform(200, 5000)

    if use_true_prop:
      props_vec = true_prop_vec(in_adata, n_cells)
    else:
      props_vec = gen_prop_vec_lognormal(len_vector, n_cells)
    props = pd.DataFrame(props_vec)
    props = props.transpose()
    props.columns = in_adata.obs["scpred_CellType"].unique()

    sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])

    #iterate over all the cell types
    for cell_idx in range(len_vector):
      cell_type_id = in_adata.obs["scpred_CellType"].unique()[cell_idx]
      num_cell = props_vec[cell_idx]
      ct_sum = get_cell_type_sum(in_adata, cell_type_id, num_cell)

      # add noise if we don't want the true proportions
      #if not use_true_prop:
      ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])

      sum_over_cells = sum_over_cells + ct_sum


    #sum_over_cells = pd.DataFrame(sum_over_cells)
    #sum_over_cells.columns = in_adata.var['gene_ids']

    # add sample noise
    if useSampleNoise:
      sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])  # 0.1
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      sum_over_cells = np.random.poisson(sum_over_cells)[0]
    else:
      sum_over_cells = sum_over_cells.T



    sum_over_cells = pd.DataFrame(sum_over_cells)
    sum_over_cells = sum_over_cells.T
    sum_over_cells.columns = in_adata.var['gene_ids']

    if samp_idx < num_samples:
      total_prop = pd.concat([total_prop, props])
      total_expr = pd.concat([total_expr, sum_over_cells])
    else:
      test_prop = pd.concat([test_prop, props])
      test_expr = pd.concat([test_expr, sum_over_cells])


  return (total_prop, total_expr, test_prop, test_expr)

def get_only1_celltype_prop_matrix(num_samp, cell_order):
  num_celltypes = len(cell_order)  

  total_prop = pd.DataFrame(columns = cell_order)


  for curr_cell_idx in range(num_celltypes):
    curr_prop = [0]*num_celltypes
    curr_prop[curr_cell_idx] = 1

    curr_cell_prop_df = get_corr_prop_matrix(num_samp, curr_prop, cell_order, min_corr=0.95)
    total_prop = pd.concat([total_prop, curr_cell_prop_df])

  return total_prop

def get_single_celltype_prop_matrix(num_samp, cell_order):
  num_celltypes = len(cell_order)  

  total_prop = pd.DataFrame(columns = cell_order)


  for curr_cell_idx in range(num_celltypes):
    curr_prop = [0.01]*num_celltypes
    curr_prop[curr_cell_idx] = 1

    curr_cell_prop_df = get_corr_prop_matrix(num_samp, curr_prop, cell_order, min_corr=0.95)
    total_prop = pd.concat([total_prop, curr_cell_prop_df])

  return total_prop

def get_corr_prop_matrix(num_samp, real_prop, cell_order, min_corr=0.8):

  # now generate all the proportions
  total_prop = pd.DataFrame(columns = cell_order)

  while total_prop.shape[0] < num_samp:
    ## generate the proportions matrix
    curr_prop_vec_noise = real_prop*np.random.lognormal(0, 1, len(real_prop))
    curr_prop_vec_noise = np.asarray(curr_prop_vec_noise/np.sum(curr_prop_vec_noise))
    curr_coef = np.corrcoef(curr_prop_vec_noise, real_prop)[0,1]

    if curr_coef > min_corr:
      props = pd.DataFrame(curr_prop_vec_noise)
      props = props.transpose()
      props.columns = cell_order 
      total_prop = pd.concat([total_prop, props])

  return total_prop


def calc_prop(in_adata, cell_order):

  tab = in_adata.obs.groupby(['scpred_CellType']).size()
  tab = tab[cell_order]
  real_prop = np.asarray(tab/np.sum(tab))
  prop_cols = np.asarray(cell_order)

  props = pd.DataFrame(real_prop)
  props = props.transpose()
  props.columns = prop_cols      

  return props

# total pseudobulk
def use_prop_make_sum(in_adata, num_cells, props_vec, cell_noise, sample_noise=None, useSampleNoise=True):

  len_vector = props_vec.shape[1]
  cell_order = props_vec.columns.values.to_list()

  # instantiate the expression and proportion vectors
  total_expr = pd.DataFrame(columns = in_adata.var['gene_ids'])
  total_prop = pd.DataFrame(columns = cell_order)


  # cell specific noise, new noise for each sample
  if cell_noise == None:
      cell_noise = [np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0]) for i in range(len_vector)]

  # iterate over all the samples we would like to make
  for samp_idx in range(props_vec.shape[0]):
    if samp_idx % 100 == 0:
        print(samp_idx)

    n_cells = num_cells
    if num_cells is None:
      n_cells = np.random.uniform(200, 5000)


    props = pd.DataFrame(props_vec.iloc[samp_idx])
    props = props.transpose()
    props.columns = cell_order

    sum_over_cells = np.zeros(in_adata.var['gene_ids'].shape[0])


    #iterate over all the cell types
    for cell_idx in range(len_vector):
      cell_type_id = cell_order[cell_idx]
      num_cell = props_vec.iloc[samp_idx, cell_idx]*n_cells
      num_cell = num_cell.astype(int)
      ct_sum = get_cell_type_sum(in_adata, cell_type_id, num_cell)

      # add noise if we don't want the true proportions
      #if not use_true_prop:
      ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])

      sum_over_cells = sum_over_cells + ct_sum


    #sum_over_cells = pd.DataFrame(sum_over_cells)
    #sum_over_cells.columns = in_adata.var['gene_ids']

    #sum_over_cells = pd.DataFrame(sum_over_cells)
    #sum_over_cells.columns = in_adata.var['gene_ids']

    # add sample noise
    if useSampleNoise:
      sample_noise = np.random.lognormal(0, 1, in_adata.var['gene_ids'].shape[0])  # 0.1
      sum_over_cells = np.multiply(sum_over_cells, sample_noise)
      # library size
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, 1)[0]
      # random variability
      sum_over_cells = sum_over_cells*np.random.lognormal(0, 0.1, in_adata.var['gene_ids'].shape[0])
      # add poisson noise
      sum_over_cells = np.random.poisson(sum_over_cells)[0]
    else:
      sum_over_cells = sum_over_cells.T


    sum_over_cells = pd.DataFrame(sum_over_cells)
    sum_over_cells = sum_over_cells.T
    sum_over_cells.columns = in_adata.var['gene_ids']

    total_expr = pd.concat([total_expr, sum_over_cells])
    total_prop = pd.concat([total_prop, props])



  return (total_prop, total_expr, sample_noise)

def read_single_kang_pseudobulk_file(data_path, sample_id, stim_status, isTraining, file_name):

  pseudobulk_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_pseudo_splits.pkl")
  prop_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_prop_splits.pkl")

  gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")
  sig_file = os.path.join(data_path, f"{file_name}_sig.pkl")

  pseudobulk_path = Path(pseudobulk_file)
  prop_path = Path(prop_file)
  gene_path = Path(gene_file)
  sig_path = Path(sig_file)

  prop_df = pickle.load( open( prop_path, "rb" ) )
  pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
  gene_df = pickle.load( open( gene_path, "rb" ) )
  sig_df = pickle.load( open( sig_path, "rb" ) )

  num_samps = pseudobulks_df.shape[0] 
  samp_type = ["bulk"]*num_samps
  if sample_id == "1015" or sample_id == "1256":
    cell_prop_type = ["random"]*1000+["cell_type_specific"]*1000
    samp_type = ["sc_ref"]*2000
  elif isTraining == "Train":
    cell_prop_type = ["realistic"]*num_samps
  else:
    cell_prop_type = ["realistic"]*100+["cell_type_specific"]*1000

  metadata_df = pd.DataFrame(data = {"sample_id":[sample_id]*num_samps, 
                                    "stim":[stim_status]*num_samps,
                                    "isTraining":[isTraining]*num_samps,
                                    "cell_prop_type":cell_prop_type,
                                    "samp_type":samp_type,})

  return (pseudobulks_df, prop_df, gene_df, sig_df, metadata_df)


def read_single_pseudobulk_file(data_path, sample_id, stim_status, isTraining, file_name, num_rand_pseudo, num_ct_pseudo):

  pseudobulk_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_pseudo_splits.pkl")
  prop_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_prop_splits.pkl")

  gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")
  sig_file = os.path.join(data_path, f"{file_name}_sig.pkl")

  pseudobulk_path = Path(pseudobulk_file)
  prop_path = Path(prop_file)
  gene_path = Path(gene_file)
  sig_path = Path(sig_file)

  prop_df = pickle.load( open( prop_path, "rb" ) )
  pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
  gene_df = pickle.load( open( gene_path, "rb" ) )
  sig_df = pickle.load( open( sig_path, "rb" ) )

  num_samps = pseudobulks_df.shape[0] 
  samp_type = ["bulk"]*num_samps
  cell_prop_type = ["random"]*num_rand_pseudo+["cell_type_specific"]*num_ct_pseudo 
  samp_type = ["sc_ref"]*(num_rand_pseudo+num_ct_pseudo)
  
  
  metadata_df = pd.DataFrame(data = {"sample_id":[sample_id]*num_samps, 
                                    "stim":[stim_status]*num_samps,
                                    "isTraining":[isTraining]*num_samps,
                                    "cell_prop_type":cell_prop_type,
                                    "samp_type":samp_type,})

  return (pseudobulks_df, prop_df, gene_df, sig_df, metadata_df)



def read_single_liver_pseudobulk_file(data_path, sample_id, stim_status, isTraining, file_name):

  pseudobulk_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_pseudo_splits.pkl")
  prop_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_prop_splits.pkl")

  gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")
  sig_file = os.path.join(data_path, f"{file_name}_sig.pkl")

  pseudobulk_path = Path(pseudobulk_file)
  prop_path = Path(prop_file)
  gene_path = Path(gene_file)
  sig_path = Path(sig_file)

  prop_df = pickle.load( open( prop_path, "rb" ) )
  pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
  gene_df = pickle.load( open( gene_path, "rb" ) )
  sig_df = pickle.load( open( sig_path, "rb" ) )

  num_samps = pseudobulks_df.shape[0] 
  samp_type = ["bulk"]*num_samps
  #if sample_id == "samp1":
  # 800 because 8 cell types
  cell_prop_type = ["random"]*1000+["cell_type_specific"]*800 
  samp_type = ["sc_ref"]*1800
  #elif isTraining == "Train":
  #  cell_prop_type = ["realistic"]*num_samps
  #else:
  #  cell_prop_type = ["realistic"]*100+["cell_type_specific"]*900

  metadata_df = pd.DataFrame(data = {"sample_id":[sample_id]*num_samps, 
                                    "stim":[stim_status]*num_samps,
                                    "isTraining":[isTraining]*num_samps,
                                    "cell_prop_type":cell_prop_type,
                                    "samp_type":samp_type,})

  return (pseudobulks_df, prop_df, gene_df, sig_df, metadata_df)



def read_single_covid_pseudobulk_file(data_path, sample_id, stim_status, isTraining, file_name):

  pseudobulk_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_pseudo_splits.pkl")
  prop_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_prop_splits.pkl")

  gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")
  sig_file = os.path.join(data_path, f"{file_name}_sig.pkl")

  pseudobulk_path = Path(pseudobulk_file)
  prop_path = Path(prop_file)
  gene_path = Path(gene_file)
  sig_path = Path(sig_file)

  prop_df = pickle.load( open( prop_path, "rb" ) )
  pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
  gene_df = pickle.load( open( gene_path, "rb" ) )
  sig_df = pickle.load( open( sig_path, "rb" ) )

  num_samps = pseudobulks_df.shape[0] 
  samp_type = ["bulk"]*num_samps
  #if sample_id == "samp1":
  # 1500 because 15 cell types
  cell_prop_type = ["random"]*1000+["cell_type_specific"]*1300 
  samp_type = ["sc_ref"]*2300
  #elif isTraining == "Train":
  #  cell_prop_type = ["realistic"]*num_samps
  #else:
  #  cell_prop_type = ["realistic"]*100+["cell_type_specific"]*900

  metadata_df = pd.DataFrame(data = {"sample_id":[sample_id]*num_samps, 
                                    "stim":[stim_status]*num_samps,
                                    "isTraining":[isTraining]*num_samps,
                                    "cell_prop_type":cell_prop_type,
                                    "samp_type":samp_type,})

  return (pseudobulks_df, prop_df, gene_df, sig_df, metadata_df)


def read_all_covid_pseudobulk_files(data_path, file_name):

  sample_order = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
  stim_order = ['CTRL']
  train_order = ['Train']

  X_concat = None
  Y_concat = None
  meta_concat = None


  for curr_samp in sample_order:
        
      pseudobulks_df, prop_df, gene_df, sig_df, metadata_df = read_single_covid_pseudobulk_file(data_path, curr_samp, "CTRL", 'Train', file_name)

      if X_concat is None:
        X_concat, Y_concat, meta_concat = pseudobulks_df, prop_df, metadata_df
      else:
        X_concat = pd.concat([X_concat, pseudobulks_df])
        Y_concat = pd.concat([Y_concat, prop_df])
        meta_concat = pd.concat([meta_concat, metadata_df])


  return (X_concat, Y_concat, gene_df, meta_concat)



def read_single_kidney_pseudobulk_file(data_path, sample_id, stim_status, isTraining, file_name):

  pseudobulk_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_pseudo_splits.pkl")
  prop_file = os.path.join(data_path, f"{file_name}_{sample_id}_{stim_status}_{isTraining}_prop_splits.pkl")

  gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")
  sig_file = os.path.join(data_path, f"{file_name}_sig.pkl")

  pseudobulk_path = Path(pseudobulk_file)
  prop_path = Path(prop_file)
  gene_path = Path(gene_file)
  sig_path = Path(sig_file)

  prop_df = pickle.load( open( prop_path, "rb" ) )
  pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
  gene_df = pickle.load( open( gene_path, "rb" ) )
  sig_df = pickle.load( open( sig_path, "rb" ) )

  num_samps = pseudobulks_df.shape[0] 
  samp_type = ["bulk"]*num_samps
  #if sample_id == "samp1":
  # 1500 because 15 cell types
  cell_prop_type = ["random"]*1000+["cell_type_specific"]*1500 
  samp_type = ["sc_ref"]*2500
  #elif isTraining == "Train":
  #  cell_prop_type = ["realistic"]*num_samps
  #else:
  #  cell_prop_type = ["realistic"]*100+["cell_type_specific"]*900

  metadata_df = pd.DataFrame(data = {"sample_id":[sample_id]*num_samps, 
                                    "stim":[stim_status]*num_samps,
                                    "isTraining":[isTraining]*num_samps,
                                    "cell_prop_type":cell_prop_type,
                                    "samp_type":samp_type,})

  return (pseudobulks_df, prop_df, gene_df, sig_df, metadata_df)


def read_all_kang_pseudobulk_files(data_path, file_name, num_bulks_training=10, seed=10):

  sample_order = ['1015', '1256', '1488', '1244', '1016', '101', '1039', '107']
  stim_order = ['STIM', 'CTRL']
  train_order = ['Train', 'Test']

  X_concat = None
  Y_concat = None
  meta_concat = None



  for curr_samp in sample_order:
    if curr_samp == '1015' or curr_samp == '1256':
      pseudobulks_df, prop_df, gene_df, sig_df, metadata_df = read_single_kang_pseudobulk_file(data_path, curr_samp, "CTRL", "Train", file_name)

      if X_concat is None:
        X_concat, Y_concat, meta_concat = pseudobulks_df, prop_df, metadata_df
      else:
        X_concat = pd.concat([X_concat, pseudobulks_df])
        Y_concat = pd.concat([Y_concat, prop_df])
        meta_concat = pd.concat([meta_concat, metadata_df])

      continue
    print(curr_samp)
    for curr_stim in stim_order:
      print(curr_stim)

      for curr_train in train_order:
        print(curr_train)

        pseudobulks_df, prop_df, gene_df, sig_df, metadata_df = read_single_kang_pseudobulk_file(data_path, curr_samp, curr_stim, curr_train, file_name)

        # subsample the number of bulks used in training
        if curr_train == "Train":
          np.random.seed(seed)
          subsamp_idx = np.random.choice(range(pseudobulks_df.shape[0]), num_bulks_training)
          pseudobulks_df = pseudobulks_df.iloc[subsamp_idx]
          prop_df = prop_df.iloc[subsamp_idx]
          metadata_df = metadata_df.iloc[subsamp_idx]

        X_concat = pd.concat([X_concat, pseudobulks_df])
        Y_concat = pd.concat([Y_concat, prop_df])
        meta_concat = pd.concat([meta_concat, metadata_df])

  return (X_concat, Y_concat, gene_df, meta_concat)


def read_all_kidney_pseudobulk_files(data_path, file_name, num_bulks_training=10):

  sample_order = ['samp1', 'samp2', 'samp3']
  stim_order = ['STIM', 'CTRL']
  train_order = ['Train', 'Test']

  X_concat = None
  Y_concat = None
  meta_concat = None


  for curr_samp in sample_order:
    if curr_samp == 'samp1':
      pseudobulks_df, prop_df, gene_df, sig_df, metadata_df = read_single_kidney_pseudobulk_file(data_path, curr_samp, "CTRL", "Train", file_name)

      if X_concat is None:
        X_concat, Y_concat, meta_concat = pseudobulks_df, prop_df, metadata_df
      else:
        X_concat = pd.concat([X_concat, pseudobulks_df])
        Y_concat = pd.concat([Y_concat, prop_df])
        meta_concat = pd.concat([meta_concat, metadata_df])

      continue
    print(curr_samp)
    for curr_stim in stim_order:
      print(curr_stim)

      for curr_train in train_order:
        print(curr_train)

        pseudobulks_df, prop_df, gene_df, sig_df, metadata_df = read_single_kidney_pseudobulk_file(data_path, curr_samp, curr_stim, curr_train, file_name)

        # subsample the number of bulks used in training
        if curr_train == "Train":
          subsamp_idx = np.random.choice(range(pseudobulks_df.shape[0]), num_bulks_training)
          pseudobulks_df = pseudobulks_df.iloc[subsamp_idx]
          prop_df = prop_df.iloc[subsamp_idx]
          metadata_df = metadata_df.iloc[subsamp_idx]

        X_concat = pd.concat([X_concat, pseudobulks_df])
        Y_concat = pd.concat([Y_concat, prop_df])
        meta_concat = pd.concat([meta_concat, metadata_df])

  return (X_concat, Y_concat, gene_df, meta_concat)


def read_diva_files(data_path, file_idx, file_name, use_test=False):

    pseudo_str = "pseudo"
    prop_str = "prop"
    if use_test:
      pseudo_str = "testpseudo"
      prop_str = "testprop"
     

    if file_idx is not None:
      pbmc_rep1_pseudobulk_file = os.path.join(data_path, f"{file_name}_{pseudo_str}_{file_idx}.pkl")
      pbmc_rep1_prop_file = os.path.join(data_path, f"{file_name}_{prop_str}_{file_idx}.pkl")
    else:
      pbmc_rep1_pseudobulk_file = os.path.join(data_path, f"{file_name}_pseudo.pkl")
      pbmc_rep1_prop_file = os.path.join(data_path, f"{file_name}_prop.pkl")

    pbmc_rep1_gene_file = os.path.join(data_path, f"{file_name}_genes.pkl")
    pbmc_rep1_sig_file = os.path.join(data_path, f"{file_name}_sig.pkl")

    pseudobulk_path = Path(pbmc_rep1_pseudobulk_file)
    prop_path = Path(pbmc_rep1_prop_file)
    gene_path = Path(pbmc_rep1_gene_file)
    sig_path = Path(pbmc_rep1_sig_file)

    prop_df = pickle.load( open( prop_path, "rb" ) )
    pseudobulks_df = pickle.load( open( pseudobulk_path, "rb" ) )
    gene_df = pickle.load( open( gene_path, "rb" ) )
    sig_df = pickle.load( open( sig_path, "rb" ) )

    return (pseudobulks_df, prop_df, gene_df, sig_df)

def read_all_diva_files(data_path, idx_range, file_name, use_test=False):

    X_concat = None

    for idx in idx_range:
        X_train, Y_train, gene_df, _ = read_diva_files(data_path, idx, file_name, use_test)
        X_train.columns = gene_df

        if X_concat is None:
            X_concat, Y_concat = X_train, Y_train
        else:
            X_concat = pd.concat([X_concat, X_train])
            Y_concat = pd.concat([Y_concat, Y_train])

    return (X_concat, Y_concat, gene_df)


def write_cs_bp_files(cybersort_path, out_file_id, pbmc1_a_df, X_train, patient_idx=0):
    # write out the scRNA-seq signature matrix
    sig_out_file = os.path.join(cybersort_path, f"{out_file_id}_{patient_idx}_cybersort_sig.tsv.gz")
    sig_out_path = Path(sig_out_file)
    pbmc1_a_df = pbmc1_a_df.transpose()

    # cast from matrix to pd
    pbmc1_a_df = pd.DataFrame(pbmc1_a_df)

    pbmc1_a_df.to_csv(sig_out_path, sep='\t',header=False)

    # write out the bulk RNA-seq mixture matrix
    sig_out_file = os.path.join(cybersort_path, f"{out_file_id}_{patient_idx}_cybersort_mix.tsv.gz")
    sig_out_path = Path(sig_out_file)

    X_train.to_csv(sig_out_path, sep='\t',header=True)
