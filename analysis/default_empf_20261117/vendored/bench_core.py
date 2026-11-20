"""Donor-disjoint splits, training-only transforms, and shared metrics."""
from __future__ import annotations
from dataclasses import dataclass
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.utils.extmath import randomized_svd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
TYPES = ["Micro-PVM_1", "Micro-PVM_2", "Micro-PVM_2_1-SEAAD", "Micro-PVM_2_3-SEAAD", "Micro-PVM_3-SEAAD", "Micro-PVM_4-SEAAD"]
STAGES = ["Not AD", "Low", "Intermediate", "High"]
CONDITION_NAMES = ["Braak/6", "Thal/5", "CERAD/3", "ADNC/3", "Male", "APOE4_dose", "age_training_standardized"]


def filehash(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def encode_condition(row):
    braak = {f"Braak {r}":i for i,r in enumerate(["0","I","II","III","IV","V","VI"])}
    cerad = {"Absent":0,"Sparse":1,"Moderate":2,"Frequent":3}
    genotype = row["APOE Genotype"].split("/")
    if len(genotype)!=2 or any(x not in ["2","3","4"] for x in genotype):
        raise ValueError(f"Unrecognized APOE genotype {genotype}")
    v = np.array([braak[row["Braak"]]/6, int(row["Thal"].removeprefix("Thal "))/5,
                  cerad[row["CERAD score"]]/3, STAGES.index(row["Overall AD neuropathological Change"])/3,
                  {"Female":0,"Male":1}[row["Sex"]], genotype.count("4"), float(row["Age at Death"])],dtype=np.float64)
    if not np.isfinite(v).all() or not (0<=v[1]<=1):
        raise ValueError("Invalid donor condition")
    return v


@dataclass
class Dataset:
    counts: sp.csr_matrix
    donor: np.ndarray
    typ: np.ndarray
    names: np.ndarray
    donors: np.ndarray
    raw_c: np.ndarray


def load_dataset():
    root = ROOT/"rebuilt"
    plan = json.loads((root/"candidate_cohort.json").read_text())
    meta = json.loads((root/"mg_meta.json").read_text())
    cols,rows = meta["cols"],meta["rows"]
    donors = np.array(sorted(plan["donors"]),dtype=str)
    matrices,names,labels,types,conditions = [],[],[],[],[]
    for d in donors:
        path = root/"counts"/f"{d}.npz"
        rec = json.loads(path.with_suffix(".receipt.json").read_text())
        namespath = path.with_name(d+"_names.npy")
        if filehash(path)!=rec["counts_sha256"] or filehash(namespath)!=rec["names_sha256"]:
            raise ValueError("Count/name hash mismatch")
        mat = sp.load_npz(path).tocsr()
        ns = np.load(namespath,allow_pickle=False)
        if len(ns)!=mat.shape[0] or len(ns)!=plan["donors"][d] or len(ns)<100:
            raise ValueError("Donor count mismatch")
        records = [dict(zip(cols,rows[str(n)])) for n in ns]
        cc = np.stack([encode_condition(r) for r in records])
        if not np.equal(cc,cc[0]).all() or any(r["Donor ID"]!=d for r in records):
            raise ValueError("Conflicting donor covariates/identity")
        if any(r["Used in analysis"]!="True" or r["Supertype"] not in TYPES for r in records):
            raise ValueError("Unexpected cohort cell")
        matrices.append(mat); names.extend(ns); labels.extend([d]*len(ns))
        types.extend(TYPES.index(r["Supertype"]) for r in records); conditions.append(cc[0])
    C = sp.vstack(matrices,format="csr")
    if C.shape!=(35765,36601) or len(donors)!=82 or len(set(names))!=len(names):
        raise ValueError("Frozen 82 donor / 35765 cell cohort mismatch")
    return Dataset(C,np.array(labels),np.array(types),np.array(names),donors,np.stack(conditions))


def make_splits(data):
    raw,ds = data.raw_c,data.donors
    masks = {"S1":np.isclose(raw[:,0],4/6), "S2":(raw[:,4]==1)&np.isclose(raw[:,3],1),
             "S3":(raw[:,4]==0)&np.isclose(raw[:,3],2/3), "S4":np.isclose(raw[:,0],1)}
    if [masks[x].sum() for x in ["S1","S2","S3","S4"]]!=[23,12,10,15]:
        raise ValueError("Frozen split counts do not match actual metadata")
    rng = np.random.default_rng(20261026)
    folds = np.empty(len(ds),int)
    stage = np.rint(raw[:,3]*3).astype(int)
    for s in range(4):
        idx = rng.permutation(np.where(stage==s)[0]); folds[idx]=np.arange(len(idx))%5
    for f in range(5):
        masks[f"S0f{f}"] = folds==f
    splits = {}
    for name,mask in masks.items():
        outer = np.where(~mask)[0]
        rng = np.random.default_rng(20261027)
        val = []
        for s in range(4):
            ids = outer[stage[outer]==s]
            if len(ids)>2:
                val.extend(rng.permutation(ids)[:max(1,int(round(.1*len(ids))))])
        val = np.array(sorted(val),int)
        fit = np.setdiff1d(outer,val)
        splits[name] = {"name":name,"fit_donors":ds[fit].tolist(),"validation_donors":ds[val].tolist(),
                        "test_donors":ds[mask].tolist(),"outer_training_donors":ds[outer].tolist(),
                        "transform_fit_donors":ds[fit].tolist(),
                        "validation_rule":"ADNC-stratified approximately 10% within outer train; seed20261027; all models/transforms exclude validation donors from fitting"}
    return splits


def log_hvg(H):
    X = H.toarray() if sp.issparse(H) else np.asarray(H)
    X = X.astype(np.float32)
    return np.log1p(X*(1e4/np.maximum(X.sum(-1,keepdims=True),1)))


class TypeClassifier(nn.Module):
    def __init__(self,n_pc=50):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_pc,64),nn.ReLU(),nn.Linear(64,len(TYPES)))
    def forward(self,x):
        return self.net(x)
    def fit(self,X,y):
        torch.manual_seed(0)
        opt = torch.optim.Adam(self.parameters(),lr=.003)
        xt,yt = torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long)
        for _ in range(600):
            idx = torch.randint(len(xt),(1024,))
            loss = nn.functional.cross_entropy(self(xt[idx]),yt[idx])
            opt.zero_grad();loss.backward();opt.step()
        self.eval()
    @torch.no_grad()
    def predict(self,X):
        return torch.cat([self(torch.as_tensor(x,dtype=torch.float32)).argmax(-1)
                          for x in np.array_split(X,max(1,int(np.ceil(len(X)/4096))))]).numpy()


@dataclass
class Space:
    hvg: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    W: np.ndarray
    pc_sd: np.ndarray
    age_mean: float
    age_sd: float
    clf: TypeClassifier
    def standardized(self,H):
        return (log_hvg(H)-self.mean)/self.sd
    def project(self,H):
        return ((self.standardized(H)@self.W)/self.pc_sd).astype(np.float32)
    def condition(self,raw):
        c = np.array(raw,dtype=np.float32,copy=True)
        c[:,-1]=(c[:,-1]-self.age_mean)/self.age_sd
        return c
    def summary(self,pcs):
        comp = np.bincount(self.clf.predict(pcs),minlength=len(TYPES))/len(pcs)
        return np.concatenate([np.sqrt(comp),pcs.mean(0)[:20]])


def fit_space(data,split,*,n_hvg=2000,n_pc=50):
    fit_mask = np.isin(data.donor,split["fit_donors"])
    C = data.counts[fit_mask].astype(np.float32)
    scale = 1e4/np.maximum(np.asarray(C.sum(1)).ravel(),1)
    L = sp.diags(scale)@C; L.data=np.log1p(L.data)
    mean = np.asarray(L.mean(0)).ravel()
    var = np.asarray(L.multiply(L).mean(0)).ravel()-mean**2
    hvg = np.argsort(-(var/np.maximum(mean,.001))*(mean>.01),kind="stable")[:n_hvg]
    X = log_hvg(C[:,hvg]); m=X.mean(0); sd=X.std(0)+1e-6
    Z = (X-m)/sd
    subset = np.random.default_rng(0).choice(len(Z),min(20000,len(Z)),replace=False)
    _,_,Vt = randomized_svd(Z[subset],n_components=n_pc,n_iter=5,random_state=0)
    W=Vt.T.copy(); ps=np.maximum((Z@W).std(0),1e-6)
    pc = ((Z@W)/ps).astype(np.float32)
    torch.manual_seed(0)
    clf = TypeClassifier(n_pc);clf.fit(pc,data.typ[fit_mask])
    age=data.raw_c[np.isin(data.donors,split["fit_donors"]),-1]
    if age.std()==0:
        raise ValueError("Training age has zero variance")
    return Space(hvg,m,sd,W,ps,float(age.mean()),float(age.std()),clf)


def save_space(space,folder,split):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"space.npz",**{k:getattr(space,k) for k in ["hvg","mean","sd","W","pc_sd","age_mean","age_sd"]})
    torch.save(space.clf.state_dict(),folder/"type_classifier.pt")
    write_json(folder/"split.json",split)


def load_space(folder):
    folder=Path(folder)
    with np.load(folder/"space.npz",allow_pickle=False) as f:
        kwargs={k:f[k] for k in f.files}
    clf=TypeClassifier(kwargs["W"].shape[1]);clf.load_state_dict(torch.load(folder/"type_classifier.pt",map_location="cpu",weights_only=True));clf.eval()
    return Space(**kwargs,clf=clf)


def energy_score(Y,y):
    cross=np.linalg.norm(Y-y,axis=1).mean()
    return float(cross-.5*cdist(Y,Y).sum()/(len(Y)*(len(Y)-1))) if len(Y)>1 else float(cross)


def cell_energy(A,B):
    return float(2*cdist(A,B).mean()-cdist(A,A).mean()-cdist(B,B).mean())


def paired_ci(a,b,seed=20261026):
    diff=np.asarray(a)-np.asarray(b)
    draws=np.random.default_rng(seed).choice(diff,(2000,len(diff)),replace=True).mean(1)
    return {"mean":float(diff.mean()),"lower":float(np.percentile(draws,2.5)),"upper":float(np.percentile(draws,97.5))}
