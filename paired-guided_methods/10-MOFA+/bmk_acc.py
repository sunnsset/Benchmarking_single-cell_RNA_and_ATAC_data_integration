# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')


import numpy as np
import pandas as pd
import networkx as nx

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

from sklearn.decomposition import PCA
from umap import UMAP
from scipy.stats import pearsonr, spearmanr

import TI as ti
import benchmark as bmk
import utils as utils
import dataset
import model as model
import post_align as palign

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20


def calc_score(z_rna, z_atac, pt_rna, pt_atac, label_rna, label_atac):
    # calculate the diffusion pseudotime
    dpt_mtx = ti.dpt(np.concatenate((z_rna, z_atac), axis = 0), n_neigh = 10)
    pt_infer = dpt_mtx[np.argmin(pt_rna), :]
    pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
    pt_infer = pt_infer/np.max(pt_infer)

    pt_true = np.concatenate((pt_rna, pt_atac))
    pt_true[pt_true.argsort()] = np.arange(len(pt_true))
    pt_true = pt_true/np.max(pt_true)
    
    # backbone
    z = np.concatenate((z_rna, z_atac), axis = 0)
    cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
    
    groups, mean_cluster, conn = ti.backbone_inf(z, resolution = 0.01)
    mean_cluster = np.array(mean_cluster)
    root = groups[np.argmin(pt_infer)]
    G = nx.from_numpy_matrix(conn)
    T = nx.dfs_tree(G, source = root)
    
    # find trajectory backbone
    branching_nodes = [x for x,d in T.out_degree() if (d >= 2)]
    paths = [nx.shortest_path(G, source = root, target = x) for x,d in T.out_degree() if (d == 0)]
    branches = []
    for path in paths:
        last_idx = 0
        for idx, node in enumerate(path):
            if node in branching_nodes:
                if len(path[last_idx:idx]) > 0:
                    branches.append(path[last_idx:idx])
                    last_idx = idx
        if len(path[last_idx:]) > 0:
            branches.append(path[last_idx:])         
    branches = sorted(list(set(map(tuple,branches))))

    # find cells for all branches
    cell_labels_predict = np.zeros(groups.shape)
    for idx, branch in enumerate(branches):
        for x in branch:
            cell_labels_predict[groups == x] = idx
            
    F1 = bmk.F1_branches(branches = cell_labels, branches_gt = cell_labels_predict)
    kt = bmk.kendalltau(pt_infer, pt_true)
    return F1, kt

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


# In[] calculate the score for latent embedding
scores = pd.DataFrame(columns = ["dataset", "kendall-tau", "F1-score"])

latent_dim = 8
reg_d = 1
reg_g = 1
reg_mmd = 1
norm = "l1"
seeds = [0, 1, 2]
for data_name in ["lin1", "lin2", "lin3", "lin4", "lin5", "lin6",
                 "bifur1", "bifur2", "bifur3","bifur4", "bifur5", "bifur6",
                 "trifur1", "trifur2", "trifur3","trifur4","trifur5","trifur6"]: 
    print(data_name)

    # load data
    counts_rna = pd.read_csv("../data/simulated/" + data_name + "/GxC1.txt", sep = "\t", header = None).values.T
    counts_atac = pd.read_csv("../data/simulated/" + data_name + "/RxC2.txt", sep = "\t", header = None).values.T
    label_rna = pd.read_csv("../data/simulated/" + data_name + "/cell_label1.txt", sep = "\t")["pop"].values.squeeze()
    label_atac = pd.read_csv("../data/simulated/" + data_name + "/cell_label2.txt", sep = "\t")["pop"].values.squeeze()
    pt_rna = pd.read_csv("../data/simulated/" + data_name + "/pseudotime1.txt", header = None).values.squeeze()
    pt_atac = pd.read_csv("../data/simulated/" + data_name + "/pseudotime2.txt", header = None).values.squeeze()

    # preprocessing
    libsize = 100
    counts_rna = counts_rna/np.sum(counts_rna, axis = 1)[:, None] * libsize
    counts_rna = np.log1p(counts_rna)

    # load liger
    z_rna_liger = pd.read_csv("results_acc/" + data_name + "/Liger_H1.csv", index_col = 0)
    z_atac_liger = pd.read_csv("results_acc/" + data_name + "/Liger_H2.csv", index_col = 0)
    F1, kt = calc_score(z_rna = z_rna_liger, z_atac = z_atac_liger, pt_rna = pt_rna, pt_atac = pt_atac, label_rna = label_rna, label_atac = label_atac)
    scores = scores.append({
        "dataset": data_name,
        "model": "LIGER", 
        "latent_dim": 0,
        "reg_d": 0,
        "reg_g": 0,
        "reg_mmd": 0,
        "norm": 0,
        "seed": 0,            
        "kendall-tau": kt,
        "F1-score": F1,
    }, ignore_index = True)

    # load seurat
    coembed = pd.read_csv("results_acc/" + data_name + "/Seurat_pca.txt", sep = "\t").values
    z_rna_seurat = coembed[:z_rna_liger.shape[0],:]
    z_atac_seurat = coembed[z_rna_liger.shape[0]:,:]
    F1, kt = calc_score(z_rna = z_rna_seurat, z_atac = z_atac_seurat, pt_rna = pt_rna, pt_atac = pt_atac, label_rna = label_rna, label_atac = label_atac)
    scores = scores.append({
        "dataset": data_name,
        "model": "Seurat", 
        "latent_dim": 0,
        "reg_d": 0,
        "reg_g": 0,
        "reg_mmd": 0,
        "norm": 0,
        "seed": 0,            
        "kendall-tau": kt,
        "F1-score": F1,
    }, ignore_index = True)

    # load unioncom
    z_rna_unioncom = np.load("results_acc/" + data_name + "/unioncom_rna.npy")
    z_atac_unioncom = np.load("results_acc/" + data_name + "/unioncom_atac.npy")
    F1, kt = calc_score(z_rna = z_rna_unioncom, z_atac = z_atac_unioncom, pt_rna = pt_rna, pt_atac = pt_atac, label_rna = label_rna, label_atac = label_atac)
    scores = scores.append({
        "dataset": data_name,
        "model": "UnionCom", 
        "latent_dim": 0,
        "reg_d": 0,
        "reg_g": 0,
        "reg_mmd": 0,
        "norm": 0,
        "seed": 0,           
        "kendall-tau": kt,
        "F1-score": F1,
    }, ignore_index = True)

        
    for seed in seeds: 
        results_dir = "results_acc/" 
        # load scDART               
        z_rna_scDART = torch.FloatTensor(np.load(results_dir + data_name + "/z_rna_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy"))
        z_atac_scDART = torch.FloatTensor(np.load(results_dir + data_name + "/z_atac_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy"))
        # post-maching
        z_rna_scDART, z_atac_scDART = palign.match_alignment(z_rna = z_rna_scDART, z_atac = z_atac_scDART, k = 10)
        z_atac_scDART, z_rna_scDART = palign.match_alignment(z_rna = z_atac_scDART, z_atac = z_rna_scDART, k = 10)
        z_rna_scDART = z_rna_scDART.numpy()
        z_atac_scDART = z_atac_scDART.numpy()
        F1, kt = calc_score(z_rna = z_rna_scDART, z_atac = z_atac_scDART, pt_rna = pt_rna, pt_atac = pt_atac, label_rna = label_rna, label_atac = label_atac)
        scores = scores.append({
            "dataset": data_name,
            "model": "scDART", 
            "latent_dim": latent_dim,
            "reg_d": reg_d,
            "reg_g": reg_g,
            "reg_mmd": reg_mmd,
            "norm": norm,
            "seed": seed,            
            "kendall-tau": kt,
            "F1-score": F1,
        }, ignore_index = True)
        
        results_dir = "results_acc_anchor/"
        # load scDART with anchor
        z_rna_scDART_anchor = torch.FloatTensor(np.load(results_dir + data_name + "/z_rna_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy"))
        z_atac_scDART_anchor = torch.FloatTensor(np.load(results_dir + data_name + "/z_atac_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy"))
        # post-maching
        z_rna_scDART_anchor, z_atac_scDART_anchor = palign.match_alignment(z_rna = z_rna_scDART_anchor, z_atac = z_atac_scDART_anchor, k = 10)
        z_atac_scDART_anchor, z_rna_scDART_anchor = palign.match_alignment(z_rna = z_atac_scDART_anchor, z_atac = z_rna_scDART_anchor, k = 10)
        z_rna_scDART_anchor = z_rna_scDART_anchor.numpy()
        z_atac_scDART_anchor = z_atac_scDART_anchor.numpy()        
        F1, kt = calc_score(z_rna = z_rna_scDART_anchor, z_atac = z_atac_scDART_anchor, pt_rna = pt_rna, pt_atac = pt_atac, label_rna = label_rna, label_atac = label_atac)
        scores = scores.append({
            "dataset": data_name,
            "model": "scDART-anchor", 
            "latent_dim": latent_dim,
            "reg_d": reg_d,
            "reg_g": reg_g,
            "reg_mmd": reg_mmd,
            "norm": norm,
            "seed": seed,            
            "kendall-tau": kt,
            "F1-score": F1,
        }, ignore_index = True)

scores.to_csv("results_acc/scores_full.csv")

# In[]
import seaborn as sns
scores = pd.read_csv("results_acc/scores_full.csv", index_col = 0)
scores["backbone"] = [x[:-1] for x in scores["dataset"]]
# make sure to select the dataset accordingly
scores = scores[scores["dataset"].isin(["lin2", "lin3", "lin5", "bifur1", "bifur2", "bifur5", "trifur1", "trifur2", "trifur5"])]
# drop linear
scores = scores[scores["backbone"] != "lin"]

# fig = plt.figure(figsize = (10, 7))
# ax = fig.subplots(nrows = 1, ncols = 1)
# sns.boxplot(data = scores, x = "dataset", y = "kendall-tau", hue = "model", ax = ax)

# fig = plt.figure(figsize = (10, 7))
# ax = fig.subplots(nrows = 1, ncols = 1)
# sns.boxplot(data = scores, x = "dataset", y = "F1-score", hue = "model", ax = ax)

boxprops = dict(linestyle='-', linewidth=2,alpha=.7)
medianprops = dict(linestyle='-', linewidth=2, color='firebrick')

fig = plt.figure(figsize = (7, 5))
ax = fig.subplots(nrows = 1, ncols = 1)
sns.boxplot(data = scores, x = "backbone", y = "kendall-tau", hue = "model", ax = ax, palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)
ax.legend(loc='upper left', frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
adjust_box_widths(fig, 0.9)
fig.savefig("results_acc/kt.png", bbox_inches = "tight")

fig = plt.figure(figsize = (7, 5))
ax = fig.subplots(nrows = 1, ncols = 1)
sns.boxplot(data = scores, x = "backbone", y = "F1-score", hue = "model", ax = ax, palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)
ax.legend(loc='upper left', frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
adjust_box_widths(fig, 0.9)
fig.savefig("results_acc/F1.png", bbox_inches = "tight")



# In[] The gene activity module


# %%
