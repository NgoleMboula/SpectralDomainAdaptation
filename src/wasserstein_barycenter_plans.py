import numpy as np
import ot 
import time
import torch
from ot.bregman import sinkhorn
from ot.da import SinkhornTransport
#from ot.utils import unif
from src.ot_solvers import sinkhorn_
from src.barycenter_utils import ( barycentric_projection,
                            penalize_y,
                            unif,
                            init_XB_)


def wasserstein_barycenter(mu_s, Xs, Xbar, ys=None, ybar=None, reg=1e-3, b=None, weights=None,
                        method='sinkhorn', norm='max', metric='sqeuclidean', numitermax=100,
                        numInneritermax=1000, stopThr=1e-4, verbose=False, innerVerbose=False,
                        log=False, line_search=False, limit_max=np.inf, callbacks=None, implementation='torch',
                        device='cuda:0' if torch.cuda.is_available() else 'cpu', **kwargs):
    
    """Computes the Wasserstein barycenter of multiple distributions through WBT algorithm by 
    iteratively computing Sinkhorn transport plans between each source and the current barycenter 
    and updating the barycenter via weighted barycentric mapping of sources."""
    
    N = len(mu_s)
    k = Xbar.shape[0]
    d = Xbar.shape[1]

    if b is None:
        b = np.ones([k, ]) / k
    if weights is None : 
        weights = [1/N] * N

    displacement_norm = stopThr + 1
    iter_count = 0
    comp_start = time.time()
    log_dict = {'displacement_square_norms': [],
                'barycenter_coordinates': [Xbar]}
    old_Xbar = np.zeros([k, d])
    while (displacement_norm > stopThr and iter_count < numitermax):
        tstart = time.time()
        T_sum = np.zeros([k, d])
        transport_plans = []

        for i in range(N):
            Mi = ot.dist(Xs[i], Xbar, metric = metric)
            Mi = ot.utils.cost_normalization(Mi, norm = norm)
            if ys is not None and ybar is not None:
                assert Xs[i].shape[0] == len(ys[i]), f"Source {i} data/label mismatch"
                assert Xbar.shape[0] == len(ybar), "Barycenter data/label mismatch"
                Mi = penalize_y(ys=ys[i], yt=ybar, M=Mi, limit_max=limit_max)
            if implementation == 'torch':
                T_i = sinkhorn_(mu_s[i], b, Mi, reg, numItermax=numInneritermax, device=device)
            else:
                T_i = sinkhorn(mu_s[i], b, Mi, reg, numItermax=numInneritermax, verbose=innerVerbose, **kwargs)
            transport_plans.append(T_i.T)
        T_sum = sum([
            wi * barycentric_projection(Xt=Xsi, coupling=Ti) for wi, Ti, Xsi in zip(weights, transport_plans, Xs)
        ])
        
        alpha = 1.0
        
        Xbar = (1 - alpha) * Xbar + alpha * T_sum
        displacement_norm = np.sum(np.square(Xbar - old_Xbar))
        old_Xbar = Xbar.copy()
        tfinish = time.time()

        if callbacks is not None:
            for callback in callbacks:
                callback(Xbar, ybar, transport_plans)
        if log:
            log_dict["displacement_square_norms"].append(displacement_norm)
            log_dict["barycenter_coordinates"].append(Xbar)

        iter_count += 1
    if log:
        return Xbar, transport_plans, log_dict
    else:
        return Xbar, transport_plans


class WassersteinBarycenterPlan:

    "Computes the Wasserstein barycenter of multiple sources,"
    "computes the couplings between the barycenter and the target, "
    "and returns all the transport plans between the barycenter, sources, and target."
    
    def __init__(self,
                 barycenter_initialization='zeros',
                 weights=None,
                 verbose=False,
                 barycenter_solver=wasserstein_barycenter,
                 transport_solver=SinkhornTransport):
        self.barycenter_initialization = barycenter_initialization
        self.weights = weights
        self.verbose = verbose
        self.barycenter_solver = barycenter_solver
        self.transport_solver = transport_solver

    def _log(self, msg):
        if self.verbose:
            print(f"[WassersteinBarycenterPlan] {msg}")

    def fit(self, Xs=None, Xt=None, ys=None, yt=None):
        self._log("Fitting Wasserstein barycenter plan...")
        self.xs_ = Xs
        self.ys_ = ys
        self.xt_ = Xt

        self._log("Initializing source distributions...")
        mu_s = [unif(X.shape[0]) for X in Xs]

        self._log(f"Initializing barycenter with method '{self.barycenter_initialization}'")
        
        if self.barycenter_initialization == 'random_cls':
            self.Xbar, self.ybar = init_XB_(np.concatenate(self.xs_, axis=0),
                                            np.concatenate(self.ys_, axis=0))
        else:
            raise ValueError(f"Invalid barycenter_initialization='{self.barycenter_initialization}'. ")

        if self.weights is None:
            self.weights = unif(len(self.xs_))
            self._log("No weights provided, using uniform distribution.")

        self._log("Solving barycenter...")
        bary, couplings = self.barycenter_solver(mu_s=mu_s, Xs=self.xs_, Xbar=self.Xbar)
        couplings = [c.T for c in couplings]

        source_to_bary_coupling = {"Barycenter Coupling {}".format(i): c for i, c in enumerate(couplings)}
        self._log(f"Source-to-barycenter coupling computed with {len(couplings)} distributions.")

        self.Xbar = bary

        self._log("Fitting barycenter-to-target transport...")
        self.BaryT = self.transport_solver()
        self.BaryT.fit(Xs=self.Xbar, ys=self.ybar, Xt=Xt)

        bary_to_target_coupling = self.BaryT.coupling_
        self._log("Barycenter-to-target coupling computed.")

        self.coupling_ = {
            "source_to_barycenter": source_to_bary_coupling,
            "barycenter_to_target": bary_to_target_coupling
        }

        self._log("Transport completed.")
        return self.Xbar, self.coupling_

    
    
