import torch
import os
from datetime import datetime

from config import Config
from util.trainingprocess_stage1_P0 import TrainingProcessStage1
from util.trainingprocess_stage3_P0 import TrainingProcessStage3
from util.knn_P0 import KNN
 
import pandas as pd

def main():    
    # hardware constraint for speed test
    torch.set_num_threads(1)

    os.environ['OMP_NUM_THREADS'] = '1'
    
    # initialization 
    config = Config(DB="P0")    
    torch.manual_seed(config.seed)
    starttime = datetime.now()
    print('Start time: ', datetime.now().strftime('%H:%M:%S'))

    
    # stage1 training
    print('Training start [Stage1]')
    model_stage1= TrainingProcessStage1(config)    
    for epoch in range(config.epochs_stage1):
        print('Epoch:', epoch)
        model_stage1.train(epoch)
    
    print('Write embeddings')
    model_stage1.write_embeddings()
    print('Stage 1 finished: ', (datetime.now()-starttime).seconds)
    
    # KNN
    print('KNN')
    KNN(config, neighbors = 20, knn_rna_samples=10000)
    print('KNN finished: ', (datetime.now()-starttime).seconds)
    
    
    # stage3 training
    print('Training start [Stage3]')
    model_stage3 = TrainingProcessStage3(config)    
    for epoch in range(config.epochs_stage3):
       print('Epoch:', epoch)
       model_stage3.train(epoch)
        
    print('Write embeddings [Stage3]')
    model_stage3.write_embeddings()
    print('Stage 3 finished: ', (datetime.now()-starttime).seconds)
    
    # KNN
    print('KNN stage3')
    KNN(config, neighbors = 20, knn_rna_samples=10000) # 30&20000
    print('KNN finished: ', (datetime.now()-starttime).seconds)
    
if __name__ == "__main__":
    main()
