import torch
import os

class Config(object):
    def __init__(self, DB):
        self.DB=DB
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        
        if self.DB == '10x':
            # DB info
            self.number_of_class = 11
            self.input_size = 15463
            self.rna_paths = ['/home/xcx/codes/scJoint/data_10x/exprs_10xPBMC_rna.npz']
            self.rna_labels = ['/home/xcx/codes/scJoint/data_10x/cellType_10xPBMC_rna.txt']
            self.atac_paths = ['/home/xcx/codes/scJoint/data_10x/exprs_10xPBMC_atac.npz']
            self.atac_labels = [] #Optional. If atac_labels are provided, accuracy after knn would be provided.
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''
        
        elif self.DB == "MOp":
            self.number_of_class = 21
            self.input_size = 18603
            self.rna_paths = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
            self.rna_labels = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
            self.atac_paths = ['data_MOp/YaoEtAl_ATAC_exprs.npz',\
                                'data_MOp/YaoEtAl_snmC_exprs.npz']
            self.atac_labels = ['data_MOp/YaoEtAl_ATAC_cellTypes.txt',\
                                'data_MOp/YaoEtAl_snmC_cellTypes.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.001
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 20
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
            
        elif self.DB == "db4_control":
            self.number_of_class = 7 # Number of cell types in CITE-seq data
            self.input_size = 17668 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            self.rna_paths = ['/home/xcx/codes/scJoint/data/citeseq_control_rna.npz'] # RNA gene expression from CITE-seq data
            self.rna_labels = ['/home/xcx/codes/scJoint/data/citeseq_control_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.atac_paths = ['/home/xcx/codes/scJoint/data/asapseq_control_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/xcx/codes/scJoint/data/asapseq_control_cellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = ['/home/xcx/codes/scJoint/data/citeseq_control_adt.npz'] # Protein expression from CITE-seq data
            self.atac_protein_paths = ['/home/xcx/codes/scJoint/data/asapseq_control_adt.npz'] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 


        elif self.DB == "P0":
            self.number_of_class = 19
            self.input_size = 19322
            self.rna_paths = ['/home/xcx/MYBenchmark-codes/4-scJoint/data/P0/binary_counts_rna.csv']
            self.rna_labels = ['/home/xcx/MYBenchmark-codes/4-scJoint/anno-num-P0.txt']
            self.atac_paths = ['/home/xcx/MYBenchmark-codes/4-scJoint/data/P0/binary_gene_activity_atac.csv']
            self.atac_labels = ['/home/xcx/MYBenchmark-codes/4-scJoint/anno-num-P0.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 50
            self.epochs_stage3 = 50
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 0.1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
            

        elif self.DB == "1469":
            self.number_of_class = 5
            self.input_size = 933
            self.rna_paths = ['/home/xcx/MYBenchmark-codes/4-scJoint/data/1469/binary_counts_rna.csv']
            self.rna_labels = ['/home/xcx/MYBenchmark-codes/4-scJoint/anno-num-1469.txt']
            self.atac_paths = ['/home/xcx/MYBenchmark-codes/4-scJoint/data/1469/binary_gene_activity_atac.csv']
            self.atac_labels = ['/home/xcx/MYBenchmark-codes/4-scJoint/anno-num-1469.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 25
            self.epochs_stage3 = 25
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 50
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''      


        elif self.DB == "uterus":
            self.number_of_class = 4
            self.input_size = 1000
            self.rna_paths = ["/home/xcx/MYBenchmark-codes/4-scJoint/data/uterus/binary_counts_rna.mtx"]
            self.rna_labels = ["/home/xcx/MYBenchmark-codes/4-scJoint/anno-num-uterus-rna.txt"]
            self.atac_paths = ["/home/xcx/MYBenchmark-codes/4-scJoint/data/uterus/binary_counts_atac.mtx"]
            self.atac_labels = ["/home/xcx/MYBenchmark-codes/4-scJoint/anno-num-uterus-atac.txt"]
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 50
            self.epochs_stage3 = 50
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 10
            self.checkpoint = ''  
