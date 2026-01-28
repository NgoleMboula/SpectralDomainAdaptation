import os
import json
import warnings
import torch
import argparse
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.classifiers import MultiLayerPerceptron
from src.wasserstein_barycenter_plans import WassersteinBarycenterPlan, wasserstein_barycenter
from src.barycenter_utils import set_seed
from src.spectral_embedding import SPEMB, adjacency_matrix___
from ot.da import SinkhornTransport, SinkhornL1l2Transport
from sklearn.manifold import TSNE
from tabulate import tabulate

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', default="MSD", type=str)
parser.add_argument('--algorithm', default="SeOT", type=str)
parser.add_argument('--data_path', default="./data/", type=str)
parser.add_argument('--out_path', default='./logs', type=str)
parser.add_argument('--numItermax', default=500, type=int)
parser.add_argument('--reg_e_bar', default=1e-2, type=float)
parser.add_argument('--reg_e', default=1e-4, type=float)
parser.add_argument('--n_component', default=10, type=int)
parser.add_argument('--limit_max', default=1e+3, type=float)
parser.add_argument('--StopThr', default=1e-2, type=float)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--momentum_sgd', type=float, default=0.9)
parser.add_argument('--seeds', default=[0], nargs='+')
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()
os.makedirs(args.out_path, exist_ok=True)

benchmarks_info = {
    "MSD": {"file": "MSD.npy", "fold": "MSD_crossval_index.json",
            "domains": ['Noiseless', "buccaneer2", "destroyerengine", "f16", "factory2"], "remove_col": 17},
    "MGR": {"file": "MGR.npy", "fold": "MGR_crossval_index.json",
            "domains": ['Noiseless', "buccaneer2", "destroyerengine", "f16", "factory2"], "remove_col": 17},
    "CMU-PIE": {"file": "FR.npy", "fold": "Faces_crossval_index.json",
              "domains": ['PIE07', 'PIE29', 'PIE05', 'PIE09'], "remove_col": None},
    "Office31": {"file": "office31_amazon_resnet_50.pkl",
                 "domains": ['amazon', 'dslr', 'webcam'], "is_pickle": True}
}

info = benchmarks_info.get(args.benchmark)
if info is None:
    raise ValueError("Unknown benchmark")

domain_names = info["domains"]

if info.get("is_pickle", False):
    with open(os.path.join(args.data_path, info["file"]), 'rb') as f:
        data = pickle.load(f)
    Xs, ys = data[domain_names[args.source]]
    Xt, yt = data[domain_names[args.target]]
    Xs = Xs.numpy() if isinstance(Xs, torch.Tensor) else Xs
    ys = ys.numpy() if isinstance(ys, torch.Tensor) else ys
    Xt = Xt.numpy() if isinstance(Xt, torch.Tensor) else Xt
    yt = yt.numpy() if isinstance(yt, torch.Tensor) else yt
else:
    dataset = np.load(os.path.join(args.data_path, info["file"]))
    with open(os.path.join(args.data_path, info["fold"]), 'r') as f:
        fold_dict = json.load(f)

    X = dataset[:, :-2]
    y = dataset[:, -2]
    if info["remove_col"] is not None:
        X = np.delete(X, info["remove_col"], axis=1)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
y = dataset[:, -2]
m = dataset[:, -1]
domains = np.unique(m).astype(int)
targets = [i for i in range(1, 5)] if args.benchmark=="MGR" else domains

results = {
    'benchmark': args.benchmark,
    'algorithm': args.algorithm,
    'n_component': args.n_component,
    'reg_e': args.reg_e,
    'reg_e_bar': args.reg_e_bar,
    'numItermax': args.numItermax,
    'limit_max': args.limit_max,
    'StopThr': args.StopThr,
    'seeds': args.seeds,
    'domains': domain_names,
    'targets': {},
    'overall': {}
}

domain_accs = []
op_tbl = []

for target in targets:
    target_name = domain_names[target]
    sources = [d for d in domains if d != target]
    acc_results = []

    results['targets'][target_name] = {
        'sources': [domain_names[s] for s in sources],
        'target': target_name,
        'per_seed': []
    }

    inds = [
        np.concatenate([fold_dict['Domain {}'.format(s + 1)]['Fold {}'.format(f)] for f in range(1, 6)])
        for s in sources
    ]

    
    Xs = [X[ind] for ind in inds]
    ys = [y[ind] for ind in inds]

    indt = np.concatenate([fold_dict['Domain {}'.format(target + 1)]['Fold {}'.format(f)] for f in range(1, 6)])
    Xt = X[indt]
    yt = y[indt]

    Xs = [X_source - X_source.mean(axis=0, keepdims=True) for X_source in Xs]
    Xt = Xt - Xt.mean(axis=0, keepdims=True)

    for seed in args.seeds:
        set_seed(seed)
        seed_result = {'seed': seed}

        if args.algorithm.lower() == 'seot':
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

            embedding, eigenvals = SPEMB(
                adjacency=A_bar,
                n_components=args.n_component,
                eigen_solver='arpack',
                random_state=0,
                norm_laplacian=True,
                drop_first=False
            )
            
            seed_result['eigenvalues'] = eigenvals.tolist() if hasattr(eigenvals, 'tolist') else eigenvals
            eigenvals_sorted = np.sort(eigenvals)
            gaps = np.diff(eigenvals_sorted)

            max_gap_idx = np.argmax(gaps)  
            max_gap_value = gaps[max_gap_idx]

            bar_embeddings = embedding[:Xbar.shape[0]]
            Xt_embeddings = embedding[-Xt.shape[0]:]

            if args.benchmark in ['MGR', 'CMU-PIE', 'Office31']:
                bar_embeddings = bar_embeddings.numpy() if hasattr(bar_embeddings, 'numpy') else bar_embeddings
                Xt_embeddings = Xt_embeddings.numpy() if hasattr(Xt_embeddings, 'numpy') else Xt_embeddings

                if args.benchmark == 'MGR':
                    clf = RandomForestClassifier(n_estimators=1000, max_depth=13, n_jobs=-1, random_state=0)
                elif args.benchmark == 'CMU-PIE':
                    clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
                elif args.benchmark == 'Office31':
                    clf = RandomForestClassifier(n_estimators=1000, max_depth=13, n_jobs=-1, random_state=0)

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
                args.optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                args.scheduler = torch.optim.lr_scheduler.MultiStepLR(args.optimizer, milestones=[100], gamma=0.1)
                seed_result['optimizer'] = 'Adam'

                history = clf._trainer_fit(args, bar_embeddings, yb_fold, Xt_embeddings, yt_fold, test=True)
                acc = history['t_test_acc'][-1] * 100
        else:
            raise ValueError(f"Expected '--algorithm' to be SPOT, got {args.algorithm}")
        
        acc_results.append(acc)
        seed_result['accuracy'] = acc
        results['targets'][target_name]['per_seed'].append(seed_result)

    acc_mean = np.mean(acc_results)
    acc_std = np.std(acc_results)
    results['targets'][target_name]['mean_accuracy'] = acc_mean
    results['targets'][target_name]['std_accuracy'] = acc_std
    domain_accs.append(acc_mean)

    op_tbl.append([target_name, f"{acc_mean:.2f} ± {acc_std:.2f}"])
    os.system('cls' if os.name == 'nt' else 'clear')
    print(tabulate(op_tbl, headers=["Domain", "Accuracy"], tablefmt="fancy_grid", stralign="center"))

overall_avg = np.mean(domain_accs)
results['overall']['avg_accuracy'] = overall_avg
results['overall']['domain_accuracies'] = dict(zip(domain_names, domain_accs))

output_file_pkl = os.path.join(args.out_path, f'results_{args.benchmark}_{args.algorithm}.pkl')
with open(output_file_pkl, 'wb') as f_pkl:
    pickle.dump(results, f_pkl)

output_file_txt = os.path.join(args.out_path, f'results_{args.benchmark}_{args.algorithm}.txt')
with open(output_file_txt, 'w') as f_txt:
    f_txt.write(json.dumps({
        'benchmark': args.benchmark,
        'algorithm': args.algorithm,
        'n_component': args.n_component,
        'reg_e': args.reg_e,
        'reg_e_bar': args.reg_e_bar,
        'numItermax': args.numItermax,
        'limit_max': args.limit_max,
        'StopThr': args.StopThr,
        'seeds': args.seeds,
        'domains': domain_names,
    }, indent=4))
    f_txt.write("\n\n")

    for target_name in results["targets"]:
        for seed_entry in results['targets'][target_name]['per_seed']:
            seed_entry.pop("optimizer", None) 

        f_txt.write(f'=================== {target_name} ===================\n')
        f_txt.write(json.dumps(results["targets"][target_name], indent=4))
        f_txt.write("\n\n")

    f_txt.write('=================== Overall ===================\n')
    f_txt.write(json.dumps(results["overall"], indent=4))

print(f"\nResults saved to: {output_file_pkl} and {output_file_txt}")