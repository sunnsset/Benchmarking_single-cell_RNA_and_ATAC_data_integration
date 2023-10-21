import sys, os
sys.path.append('../')
sys.path.append('../src/')

# from unioncom import UnionCom
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# import utils as utils


def lsi_ATAC(X, k = 100, use_first = False):
    """\
    Description:
    ------------
        Compute LSI with TF-IDF transform, i.e. SVD on document matrix, can do tsne on the reduced dimension

    Parameters:
    ------------
        X: cell by feature(region) count matrix
        k: number of latent dimensions
        use_first: since we know that the first LSI dimension is related to sequencing depth, we just ignore the first dimension since, and only pass the 2nd dimension and onwards for t-SNE
    
    Returns:
    -----------
        latent: cell latent matrix
    """    
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    # binarize the scATAC-Seq count matrix
    bin_X = np.where(X < 1, 0, 1)
    
    # perform Latent Semantic Indexing Analysis
    # get TF-IDF matrix
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    normed_count = tfidf.fit_transform(bin_X)
    
    # # perform SVD on the sparse matrix
    # lsi = TruncatedSVD(n_components = k, random_state=42)
    # lsi_r = lsi.fit_transform(normed_count)

    # print(normed_count.toarray())
    # print(normed_count.shape)
    
    # use the first component or not
    # if use_first:
    #     return lsi_r
    # else:
    #     return lsi_r[:, 1:]
    
    return normed_count.toarray()

    
if __name__ == '__main__':
    results_dir = "results_acc/"

    for data_name in ["lin1", "lin2", "lin3", "lin4", "lin5", "lin6",
                     "bifur1", "bifur2", "bifur3","bifur4", "bifur5", "bifur6",
                     "trifur1", "trifur2", "trifur3","trifur4","trifur5","trifur6"]: 
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
        counts_atac = np.where(counts_atac < 1, 0, 1)

        # dimension reduction?
        # counts_atac = lsi_ATAC(counts_atac, k = 100)

        # run unioncom
        uc = UnionCom.UnionCom(epoch_pd = 10000)
        integrated_data = uc.fit_transform([counts_rna, counts_atac])
        z_rna = integrated_data[0]
        z_atac = integrated_data[1]

        pca_op = PCA(n_components = 2)
        z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
        z_rna_pca = z[:z_rna.shape[0],:]
        z_atac_pca = z[z_rna.shape[0]:,:]

        utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, anno2 = label_atac, mode = "separate", save = results_dir + data_name + "/unioncom_pca.png", figsize = (20,10), axis_label = "PCA")
        np.save(file = results_dir + data_name + "/unioncom_rna.npy", arr = z_rna)
        np.save(file = results_dir + data_name + "/unioncom_atac.npy", arr = z_atac)
