import ot 
import os
import json
import warnings
import torch
import argparse
import numpy as np
import pickle
from pathlib import Path
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from src.classifiers import MultiLayerPerceptron
from src.wasserstein_barycenter_plans import WassersteinBarycenterPlan
from src.wasserstein_barycenter_plans import wasserstein_barycenter
from src.barycenter_utils import set_seed
from src.reflecto_loaders import get_reflecto_loaders
from src.spectral_embedding import SPEMB, adjacency_matrix___
from ot.da import SinkhornTransport

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--numItermax', default=500, type=int)
parser.add_argument('--reg_e_bar', default=1e-2, type=float)
parser.add_argument('--reg_e', default=1e-4, type=float)
parser.add_argument('--n_component', default=4)
parser.add_argument('--limit_max', default=1e+3)
parser.add_argument('--StopThr', default=1e-2)
parser.add_argument('--classifier', default='MLP')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--momentum_sgd', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=[0], nargs='+')
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR/"reflectometry"
SAVE_DIR = BASE_DIR / "prepared_reflecto_datasets"
SAVE_DIR.mkdir(exist_ok=True)
SAVE_FILE = SAVE_DIR / "source_datasets.npz"

source_paths_list = [
    {
        'data': DATA_DIR/"DataBase_facteurs_compression/Facteur_compression_16/files_tous_samples/files_tous_samples/train/matrice_sample_train.csv",
        'labels': DATA_DIR / "DataBase_facteurs_compression/Facteur_compression_16/files_tous_samples/files_tous_samples/train/matrice_target_classes_train.csv"
    },
    {
        'data': DATA_DIR / "DataBase_facteurs_compression/Facteur_compression_2/files_tous_samples/train/matrice_sample_train.csv",
        'labels': DATA_DIR / "DataBase_facteurs_compression/Facteur_compression_2/files_tous_samples/train/matrice_target_classes_train.csv"
    },
    {
        'data': DATA_DIR / "DataBase_longueurs_cable/Cable_coaxial_5m/files_tous_samples/train/matrice_sample_train.csv",
        'labels': DATA_DIR / "DataBase_longueurs_cable/Cable_coaxial_5m/files_tous_samples/train/matrice_target_classes_train.csv"
    },
    {
        'data': DATA_DIR / "DataBase_facteurs_compression/Facteur_compression_4/files_tous_samples/train/matrice_sample_train.csv",
        'labels': DATA_DIR / "DataBase_facteurs_compression/Facteur_compression_4/files_tous_samples/train/matrice_target_classes_train.csv"
    }
]


if SAVE_FILE.exists():
    with open(SAVE_FILE, "rb") as f:
        source_datasets = pickle.load(f)
#else:
#    source_datasets = get_reflecto_loaders(
#        source_paths_list,
#        num_samples_src=200
#    )
#    with open(SAVE_FILE, "wb") as f:
#        pickle.dump(source_datasets, f)
    

domain_names = ['CF_16', 'CF_2', 'CL_5m', 'CF_4']
domains = list(range(len(domain_names)))

print('\n')
print('-' * 52)
print('|{:^51}|'.format('SeOT - Reflectometry Dataset'))
print('-' * 52)
print('|{:^25}|{:^25}|'.format('Domain', 'Accuracy'))

for target in domains:
    sources = [d for d in domains if d != target]
    acc_results = []
    
    Xs = []
    ys = []
    for source_idx in sources:
        X_source, y_source = source_datasets[source_idx]
        Xs.append(X_source)
        ys.append(y_source)
    
    Xt, yt = source_datasets[target]

    Xs = [X_source - X_source.mean(axis=0, keepdims=True) for X_source in Xs]
    Xt = Xt - Xt.mean(axis=0, keepdims=True)
    
    for seed in args.seed:
        set_seed(seed)
        
        yb = np.concatenate(ys, axis=0)

        transport_solver = partial(
            SinkhornTransport,
            reg_e=args.reg_e,
            norm='max'
        )
        
        barycenter_solver = partial(
            wasserstein_barycenter,
            numItermax=args.numItermax,
            reg=args.reg_e_bar, 
            limit_max=args.limit_max,
            stopThr=args.StopThr, 
            ys=ys, 
            ybar=yb, 
            verbose=False
        )

        baryP = WassersteinBarycenterPlan(
            barycenter_solver=barycenter_solver,
            transport_solver=transport_solver,
            barycenter_initialization="random_cls"
        )

        Xbar, G_bar = baryP.fit(Xs=Xs, Xt=Xt, ys=ys, yt=None)
        A_bar = adjacency_matrix___(G_bar=G_bar)

        embedding, _ = SPEMB(
            adjacency=A_bar, 
            n_components=args.n_component, 
            eigen_solver='arpack', 
            random_state=0, 
            norm_laplacian=True, 
            drop_first=False
        )
        
        bar_embeddings = embedding[:Xbar.shape[0]]
        Xt_embeddings = embedding[-Xt.shape[0]:]

        if args.classifier == 'RF':

            bar_embeddings = bar_embeddings.numpy() if hasattr(bar_embeddings, 'numpy') else bar_embeddings
            Xt_embeddings = Xt_embeddings.numpy() if hasattr(Xt_embeddings, 'numpy') else Xt_embeddings

            clf = RandomForestClassifier(
                n_estimators=1000,
                max_depth=13,
                n_jobs=-1,
                random_state=0
            )
            
            clf.fit(bar_embeddings, yb)
            
            pred_t = clf.predict(Xt_embeddings)
            acc = accuracy_score(yt, pred_t) * 100

        else:
            yb_fold = torch.from_numpy(yb).long()
            yt_fold = torch.from_numpy(yt).long()
            
            clf = MultiLayerPerceptron(
                input_dim=bar_embeddings.shape[1],   
                hidden_dim=512,
                output_dim=len(np.unique(yb))
                ).to(args.device)

            args.criterion = torch.nn.CrossEntropyLoss().to(args.device)

            args.optimizer = torch.optim.Adam(
                clf.parameters(), lr=args.lr, 
                #momentum=args.momentum_sgd, 
                #weight_decay=args.weight_decay
                )
            
            args.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                args.optimizer, 
                milestones=[100],  
                gamma=0.1
                )

            history = clf._trainer_fit(
                args,
                bar_embeddings,
                yb_fold,
                Xt_embeddings,
                yt_fold,
                test=True
                )
            
            acc = history['t_test_acc'][-1] * 100
        
        acc_results.append(acc)
    
    acc_mean = np.mean(acc_results)
    acc_std = np.std(acc_results)

    print('|{:^25}|{:^25}|'.format(domain_names[target], f'{acc_mean:.2f} ± {acc_std:.2f}'))