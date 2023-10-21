import numpy as np 
import matplotlib.pyplot as plt
import scipy

# 做轨迹预测的函数
def dpt(data, n_neigh = 5):
    '''\
    Description:
    -----------
        Calculates DPT between all points in the data, directly ouput similarity matrix, which is the diffusion pseudotime matrix, a little better than diffusion map
        
    Parameters:
    -----------
        data: 
            Feature matrix, numpy.array of the size [n_samples, n_features]
        n_neigh: 
            Larger correspond to slower decay
        use_potential:
            Expand shorter cell and compress distant cell
    
    Returns:
    -----------
        DPT: 
            Similarity matrix calculated from diffusion pseudo-time
    '''
    import graphtools as gt
    from scipy.spatial.distance import pdist, squareform
    # Calculate from raw data would be too noisy, dimension reduction is necessary, construct graph adjacency matrix with n_pca 100
    G = gt.Graph(data, n_pca=None, knn = n_neigh, use_pygsp=True)
    
    # Calculate eigenvectors of the diffusion operator
    # G.diff_op is a diffusion operator, return similarity matrix calculated from diffusion operation
    W, V = scipy.sparse.linalg.eigs(G.diff_op, k = 1)
    
    # Remove first eigenspace
    T_tilde = G.diff_op.toarray() - (V[:,0] @ V[:,0].T)
    
    # Calculate M
    I = np.eye(T_tilde.shape[1])
    M = np.linalg.inv(I - T_tilde) - I
    M = np.real(M)  
        
    DPT = squareform(pdist(M))

    
    return DPT


def pt_inference(latent_rna, latent_atac, root_idx = 0, use_dpt = True):
    latent_joint = np.concatenate((latent_rna, latent_atac), axis = 0)
    dist = dpt(latent_joint, n_neigh = int(0.1 * latent_joint.shape[0]))
    
    pt = dist[root_idx, :]
    return pt

# 做MST轨迹的函数
def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print( 'Your adjacency matrix contained redundant nodes.' )
    return g

def leiden(conn, resolution = 0.05, random_state = 0, n_iterations = -1):
    try:
        import leidenalg as la
    except ImportError:
        raise ImportError(
            'Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip3 install leidenalg`.'
        )
        
    start = print('running Leiden clustering')

    partition_kwargs = {}
    # # convert adjacency matrix into igraph
    g = get_igraph_from_adjacency(conn)
    
    # Parameter setting
    partition_type = la.RBConfigurationVertexPartition
    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)     
    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    partition_kwargs['resolution_parameter'] = resolution
        
    # Leiden algorithm
    # part = la.find_partition(g, la.CPMVertexPartition, **partition_kwargs)
    part = la.find_partition(g, partition_type, **partition_kwargs)
    # groups store the length |V| array, the integer in each element(node) denote the cluster it belong
    groups = np.array(part.membership)

    n_clusters = int(np.max(groups) + 1)
    
    print('finished')
    return groups, n_clusters

def nearest_neighbor(features, k = 15, sigma = 3):
    from sklearn.neighbors import kneighbors_graph, NearestNeighbors
    from sklearn.metrics import pairwise_distances
    import random
    import networkx as nx
    from umap.umap_ import fuzzy_simplicial_set
    from scipy.sparse import coo_matrix

    nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(features)
    knn_dists, knn_indices = nbrs.kneighbors(features)

    X = coo_matrix(([], ([], [])), shape=(features.shape[0], 1))

    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors = k,
        metric = None,
        random_state = None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]
    connectivities = connectivities.toarray()
    G = nx.from_numpy_matrix(connectivities, create_using=nx.Graph)
    return connectivities, G


def backbone_inf(X, resolution = 0.5):
    import networkx as nx
    conn, G = nearest_neighbor(X, k = 15)
    groups, n_clust = leiden(conn, resolution=resolution)

    mean_cluster = [[] for x in range(n_clust)]

    for i, cat in enumerate(np.unique(groups)):
        idx = np.where(groups == cat)[0]
        mean_cluster[int(cat)] = np.mean(X[idx,:], axis = 0)

    mst = np.zeros((n_clust,n_clust))

    for i in range(n_clust):
        for j in range(n_clust):
            mst[i,j] = np.linalg.norm(np.array(mean_cluster[i]) - np.array(mean_cluster[j]), ord = 2)

    G = nx.from_numpy_matrix(-mst)
    T = nx.maximum_spanning_tree(G, weight = 'weight', algorithm = 'kruskal')
    T = nx.to_numpy_matrix(T)
    # conn is the adj of the MST.

    return groups, mean_cluster, T



# 计算肯德尔系数的函数
def kt(pt_pred, pt_true):
    from scipy.stats import kendalltau
    """\
    Description
        kendall tau correlationship
    
    Parameters
    ----------
    pt_pred
        inferred pseudo-time
    pt_true
        ground truth pseudo-time
    Returns
    -------
    tau
        returned score
    """
    pt_true = pt_true.squeeze()
    pt_pred = pt_pred.squeeze()
    tau, p_val = kendalltau(pt_pred, pt_true)
    return tau, p_val


def plot_traj(z_rna, z_atac, pt, save = None, figsize = (10,7), axis_label = "Latent", **kwargs):
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
    }
    _kwargs.update(kwargs)
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot()
    z = np.concatenate((z_rna, z_atac), axis = 0)
         
    ax.scatter(z[:,0], z[:,1], color = 'gray', alpha = 0.1)

    order = np.argsort(pt)

    pseudo_visual = ax.scatter(z[order,0],z[order,1], c = np.arange(order.shape[0])/order.shape[0], cmap=plt.get_cmap('gnuplot'), **_kwargs)

    ax.tick_params(axis = "both", which = "major", labelsize = 15)

    ax.set_xlabel(axis_label + " 1", fontsize = 19)
    ax.set_ylabel(axis_label + " 2", fontsize = 19)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)   

    cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = ax)
    cbar.ax.tick_params(labelsize = 15)
    cbar.ax.set_ylabel('Pseudotime', rotation=-270, fontsize = 20, labelpad = 20)

    if save:
        fig.savefig(save, bbox_inches = "tight")

        
def calc_score(z, pt, cell_labels, plot = None):
    # calculate the diffusion pseudotime
    dpt_mtx = dpt(z, n_neigh = 10)
    pt_infer = dpt_mtx[np.argmin(pt), :]
    pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
    pt_infer = pt_infer/np.max(pt_infer)

    pt_true = pt
    # pt_true[pt_true.argsort()] = np.arange(len(pt_true))
    # pt_true = pt_true/np.max(pt_true)
        
    groups, mean_cluster, conn = backbone_inf(z, resolution = 0.5)
    mean_cluster = np.array(mean_cluster)
    root = groups[np.argmin(pt_infer)]
    import networkx as nx
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
    
    import benchmark as bmk
    F1 = bmk.F1_branches(branches = cell_labels, branches_gt = cell_labels_predict)

    # print(np.unique(cell_labels))
    # print(cell_labels.shape)
    # print(np.unique(cell_labels_predict))
    # print(cell_labels_predict.shape)
    
    kt = bmk.kendalltau(pt_infer, pt_true)
    if plot is not None:
        plot_backbone(z, None, groups = groups, anno = cell_labels, 
                  T = conn, mean_cluster = mean_cluster, mode = "joint", figsize=(10,7), 
                  save = plot, axis_label = "LSI")
    return F1, kt