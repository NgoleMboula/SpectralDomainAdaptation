import torch
import ot
from typing import Literal, Optional, Union

def sinkhorn_(a, b, M, reg_e, numItermax=1000, stopThr=1e-9,
              device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    
    """
    Compute the entropic-regularized optimal transport plan between two 
    discrete distributions a and b using the Sinkhorn-Knopp algorithm,and 
    returns the resulting transport Plan.
    """
    
    a = torch.Tensor(a).double() if type(a) != torch.Tensor else a
    b = torch.Tensor(b).double() if type(b) != torch.Tensor else b
    M = torch.Tensor(M).double() if type(M) != torch.Tensor else M

    with torch.no_grad():
        a = a.to(device)
        b = b.to(device)

        ns = len(a)
        nt = len(b)

        u = (torch.ones(ns) / ns).to(device).double()
        v = (torch.ones(nt) / nt).to(device).double()
        K = torch.exp(- M / reg_e).to(device)
        Kp = (1 / a).view(-1, 1) * K
        cpt = 0
        err = 1
        
        while (err > stopThr and cpt < numItermax):
            uprev = u
            vprev = v
            KtU = K.t().matmul(u)
            v = torch.div(b.view(-1, 1), KtU.view(-1 ,1))
            #v = b / KtU
            u = 1. / Kp.matmul(v)
            #u = a / (K @ v)

            if (torch.any(KtU == 0) or
                torch.any(torch.isnan(u)) or
                torch.any(torch.isnan(v))):
                u = uprev
                v = vprev

            if cpt % 100 == 0:
                transp = u.view(-1, 1) * K * v.view(1, -1)
                #err = (torch.sum(transp) - b).norm(1).pow(2).cpu().numpy()
                err = (transp.sum(dim=0) - b).norm(1)

            cpt += 1

        G = torch.matmul(torch.diag(u.view(-1)), torch.matmul(K, torch.diag(v.view(-1))))        
        return G.detach().cpu().numpy()


class MaskedSinkhorn:
    def __init__(self,
                 reg: float = 0.1,
                 p: float = 2.0,
                 n_sink: int = 200,
                 tau_factor: float = 0.9,
                 verbose: bool = False,
                 device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.reg = reg
        self.p = p
        self.n_sink = n_sink
        self.tau_factor = tau_factor
        self.verbose = verbose
        self.device = device

    def _sinkhorn_kernel(self, C, reg_e):
        """Return Gibbs kernel K = exp(-C/eps) with numerical stability."""
        return torch.exp(-C / reg_e) + 1e-20

    def _sinkhorn_plan(self, K, a, b, n_iter=200):
        """Balanced plan T = diag(u)* K * diag(v) with marginals a, b."""
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        for _ in range(n_iter):
            
            u = a / (K @ v + 1e-20)
            v = b / (K.t() @ u + 1e-20)
            
            if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
                break
        
        return u.view(-1, 1) * K * v.view(1, -1)

    def fit(self, Xs, ys, Xt):
        device = self.device
        
        Xs = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        Xt = torch.as_tensor(Xt, dtype=torch.float32, device=device)
        ys = torch.as_tensor(ys, dtype=torch.long, device=device).view(-1)

        ns, nt = Xs.shape[0], Xt.shape[0]
        K = int(ys.max().item()) + 1
        
        tau = self.tau_factor * torch.log(torch.tensor(K, dtype=torch.float32, device=device) + 1)

        a = torch.full((ns,), 1.0 / ns, device=device, dtype=torch.float32)
        b = torch.full((nt,), 1.0 / nt, device=device, dtype=torch.float32)

        
        C = torch.cdist(Xs, Xt, p=self.p) ** 2
        Kmat = self._sinkhorn_kernel(C, reg_e=self.reg)
        T0 = self._sinkhorn_plan(Kmat, a, b, self.n_sink)

        
        if torch.any(torch.isnan(T0)):
            if self.verbose:
                print("Warning: Initial transport plan contains NaN values")
            return T0

        
        onehot_s = torch.nn.functional.one_hot(ys.long(), num_classes=K).float()
        U = T0.t() @ onehot_s
        P = U / (b.unsqueeze(1) + 1e-20)  
        entropy = -(P * torch.log(P + 1e-20)).sum(dim=1)
        J_conf = (entropy < tau)
        conf_idx = torch.nonzero(J_conf).squeeze()
        
        if conf_idx.numel() > 0:
            labels_conf = P[conf_idx].argmax(dim=1)

            if self.verbose:
                print(f"Confident targets below τ={tau:.2f} : {conf_idx.numel()} / {nt}")

            
            mask = torch.zeros_like(C, dtype=torch.bool, device=device)
            for j, k_conf in zip(conf_idx.tolist(), labels_conf.tolist()):
                for k in range(K):
                    if k != k_conf:
                        rows_bad = (ys == k)
                        mask[rows_bad, j] = True

            
            K_masked = Kmat.clone()
            K_masked[mask] = 0.0

           
            T1 = self._sinkhorn_plan(K_masked, a, b, self.n_sink)
        else:
            T1 = T0  

        return T1




