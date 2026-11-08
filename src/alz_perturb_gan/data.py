"""Swappable perturbation datasets.

Every trainer and baseline sees the same `PerturbationSet`: a control state,
named perturbations with embeddings and held-out profiles, and training cells.
The synthetic builder needs no downloads. The h5ad builder reads an Arc
Virtual Cell Challenge file when you have one, and otherwise is unused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Perturbation:
    """One named perturbation, single or combinatorial."""

    name: str
    embedding: torch.Tensor  # [E]
    profile: torch.Tensor  # [G] mean observed outcome; eval profiles are for scoring only
    members: tuple[str, ...]  # single-perturbation names combined into this one
    combination: bool
    split: str  # "train" or "eval"

    def __post_init__(self) -> None:
        if self.split not in ("train", "eval"):
            raise ValueError("split must be 'train' or 'eval'")
        self.embedding = self.embedding.detach().float().cpu()
        self.profile = self.profile.detach().float().cpu()


@dataclass
class PerturbationSet:
    """The only object models are allowed to train and score against.

    `cell_*` tensors are training cells. For unpaired assays (the Virtual Cell
    files), each cell's control is the non-targeting mean repeated. Synthetic
    data is paired: each cell has its own control draw.
    """

    name: str
    source: str
    gene_names: list[str]
    control_mean: torch.Tensor  # [G]
    perturbations: list[Perturbation]
    cell_control: torch.Tensor  # [N, G]
    cell_outcome: torch.Tensor  # [N, G]
    cell_embedding: torch.Tensor  # [N, E]
    cell_perturbation: list[str]
    paired_cells: bool
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.control_mean = self.control_mean.detach().float().cpu()
        self.cell_control = self.cell_control.detach().float().cpu()
        self.cell_outcome = self.cell_outcome.detach().float().cpu()
        self.cell_embedding = self.cell_embedding.detach().float().cpu()
        names = [item.name for item in self.perturbations]
        if len(names) != len(set(names)):
            raise ValueError("perturbation names must be unique")
        if len(self.cell_perturbation) != self.cell_control.shape[0]:
            raise ValueError("cell_perturbation length must match the cell batch")
        genes = self.control_mean.shape[0]
        if self.cell_control.shape[-1] != genes or self.cell_outcome.shape[-1] != genes:
            raise ValueError("cell expression width must match control_mean")
        if list(self.gene_names) and len(self.gene_names) != genes:
            raise ValueError("gene_names length must match expression width")

    @property
    def n_genes(self) -> int:
        return int(self.control_mean.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.perturbations[0].embedding.shape[0])

    def by_name(self) -> dict[str, Perturbation]:
        return {item.name: item for item in self.perturbations}

    def named(self, split: str | None = None) -> list[Perturbation]:
        if split is None:
            return list(self.perturbations)
        return [item for item in self.perturbations if item.split == split]

    def train_profiles(self) -> tuple[list[str], torch.Tensor]:
        rows = self.named("train")
        if not rows:
            raise ValueError("dataset has no training perturbations")
        return [item.name for item in rows], torch.stack([item.profile for item in rows])


def make_synthetic(
    *,
    n_genes: int = 16,
    emb_dim: int = 4,
    n_single: int = 6,
    n_combo: int = 4,
    cells_per: int = 4,
    noise: float = 0.02,
    seed: int = 0,
    holdout_singles: int = 1,
    holdout_combos: int = 2,
    control_scale: float = 1.0,
) -> PerturbationSet:
    """Tiny linear-additive simulator.

    Single-perturbation effects are a linear map of the embedding. A combination
    embedding is the sum of its members, so its effect is the sum of theirs.
    Held-out combinations keep their member singles in the training split, which
    is the setting where an additive baseline is well defined.
    """
    if n_single < 2 or n_combo < 1:
        raise ValueError("need at least two singles and one combination")
    if not 0 <= holdout_singles < n_single or not 0 <= holdout_combos <= n_combo:
        raise ValueError("holdout counts must leave at least one training single")
    if holdout_combos == n_combo and holdout_singles == n_single:
        raise ValueError("at least one perturbation must stay in training")
    generator = torch.Generator().manual_seed(seed)
    control_mean = torch.full((n_genes,), control_scale)
    weight = torch.randn(n_genes, emb_dim, generator=generator) * 0.5
    single_emb = torch.randn(n_single, emb_dim, generator=generator)
    single_emb = single_emb / single_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    single_names = [f"s{index}" for index in range(n_single)]
    combo_specs: list[tuple[str, tuple[int, int]]] = []
    for index in range(n_combo):
        left = index % n_single
        right = (index + 1) % n_single
        combo_specs.append((f"s{left}+s{right}", (left, right)))

    def effect(embedding: torch.Tensor) -> torch.Tensor:
        return embedding @ weight.T

    perturbations: list[Perturbation] = []
    n_train_single = n_single - holdout_singles
    for index, name in enumerate(single_names):
        split = "train" if index < n_train_single else "eval"
        perturbations.append(
            Perturbation(
                name=name,
                embedding=single_emb[index],
                profile=control_mean + effect(single_emb[index]),
                members=(),
                combination=False,
                split=split,
            )
        )
    n_train_combo = n_combo - holdout_combos
    for index, (name, pair) in enumerate(combo_specs):
        embedding = single_emb[pair[0]] + single_emb[pair[1]]
        split = "train" if index < n_train_combo else "eval"
        perturbations.append(
            Perturbation(
                name=name,
                embedding=embedding,
                profile=control_mean + effect(embedding),
                members=(single_names[pair[0]], single_names[pair[1]]),
                combination=True,
                split=split,
            )
        )

    # Profiles above are the noiseless means. Cells add observation noise, and
    # the stored profile is the mean of the cells actually drawn so baselines
    # fit what the training split contains. Eval profiles stay the population
    # mean: scoring against the cells' own average would leak the test draw.
    controls = []
    outcomes = []
    embeddings = []
    owners = []
    realized: dict[str, list[torch.Tensor]] = {item.name: [] for item in perturbations}
    for item in perturbations:
        draws = []
        for _ in range(cells_per):
            control = control_mean + noise * torch.randn(n_genes, generator=generator)
            outcome = control + (item.profile - control_mean) + noise * torch.randn(n_genes, generator=generator)
            if item.split == "train":
                controls.append(control)
                outcomes.append(outcome)
                embeddings.append(item.embedding)
                owners.append(item.name)
            draws.append(outcome)
        realized[item.name] = draws
    for item in perturbations:
        if item.split == "train":
            item.profile = torch.stack(realized[item.name]).mean(dim=0)

    if not controls:
        raise ValueError("synthetic split produced no training cells")
    return PerturbationSet(
        name="synthetic",
        source="synthetic",
        gene_names=[f"g{index}" for index in range(n_genes)],
        control_mean=control_mean,
        perturbations=perturbations,
        cell_control=torch.stack(controls),
        cell_outcome=torch.stack(outcomes),
        cell_embedding=torch.stack(embeddings),
        cell_perturbation=owners,
        paired_cells=True,
        metadata={
            "seed": seed,
            "n_genes": n_genes,
            "emb_dim": emb_dim,
            "n_single": n_single,
            "n_combo": n_combo,
            "cells_per": cells_per,
            "noise": noise,
            "holdout_singles": holdout_singles,
            "holdout_combos": holdout_combos,
            "effect": "linear in the embedding; combinations add",
        },
    )


def _as_dense_mean(matrix) -> torch.Tensor:
    if hasattr(matrix, "mean"):
        mean = matrix.mean(axis=0)
    else:
        mean = matrix
    array = getattr(mean, "A1", None)
    if array is None:
        array = getattr(mean, "toarray", lambda: mean)()
    return torch.tensor(array, dtype=torch.float32).reshape(-1)


def perturbation_set_from_groups(
    *,
    name: str,
    source: str,
    gene_names: list[str],
    control_mean: torch.Tensor,
    profiles: dict[str, torch.Tensor],
    train_names: list[str],
    eval_names: list[str],
    embeddings: torch.Tensor | None = None,
    cells: dict[str, torch.Tensor] | None = None,
    combination_members: dict[str, tuple[str, ...]] | None = None,
    notes: str = "",
) -> PerturbationSet:
    """Build a `PerturbationSet` from pseudobulk profiles.

    This is the swap point for a new atlas: group cells, pass the means in,
    and the baselines, GAN, and HyperGAN bridge do not change.
    """
    if not train_names:
        raise ValueError("need at least one training perturbation")
    order = list(train_names) + [name for name in eval_names if name not in train_names]
    missing = [item for item in order if item not in profiles]
    if missing:
        raise ValueError(f"missing profiles for {missing}")
    width = int(control_mean.shape[0])
    if embeddings is None:
        embeddings = torch.eye(len(order), dtype=torch.float32)
        embedding_note = (
            "Embeddings are one-hot over perturbation names in this file. "
            "A linear model fit on them cannot transfer to an identity that was absent from training."
        )
    else:
        embeddings = embeddings.detach().float().cpu()
        if embeddings.shape[0] != len(order):
            raise ValueError("embeddings must have one row per perturbation")
        embedding_note = "Caller-supplied embeddings."
    members_of = combination_members or {}
    perturbations = []
    for index, pert_name in enumerate(order):
        members = tuple(members_of.get(pert_name, ()))
        perturbations.append(
            Perturbation(
                name=pert_name,
                embedding=embeddings[index],
                profile=profiles[pert_name],
                members=members,
                combination=bool(members),
                split="train" if pert_name in train_names else "eval",
            )
        )
    controls = []
    outcomes = []
    cell_emb = []
    owners = []
    by_name = {item.name: item for item in perturbations}
    if cells:
        for pert_name, matrix in cells.items():
            if pert_name not in by_name or by_name[pert_name].split != "train":
                continue
            block = matrix.detach().float().cpu()
            if block.ndim == 1:
                block = block.unsqueeze(0)
            controls.append(control_mean.expand(block.shape[0], -1).clone())
            outcomes.append(block)
            cell_emb.append(by_name[pert_name].embedding.expand(block.shape[0], -1).clone())
            owners.extend([pert_name] * block.shape[0])
    if not owners:
        for item in perturbations:
            if item.split != "train":
                continue
            controls.append(control_mean.unsqueeze(0))
            outcomes.append(item.profile.unsqueeze(0))
            cell_emb.append(item.embedding.unsqueeze(0))
            owners.append(item.name)
    return PerturbationSet(
        name=name,
        source=source,
        gene_names=list(gene_names),
        control_mean=control_mean.detach().float().cpu().reshape(width),
        perturbations=perturbations,
        cell_control=torch.cat(controls, dim=0),
        cell_outcome=torch.cat(outcomes, dim=0),
        cell_embedding=torch.cat(cell_emb, dim=0),
        cell_perturbation=owners,
        paired_cells=False,
        metadata={"notes": " ".join(part for part in (notes, embedding_note) if part)},
    )


def load_virtual_cell_h5ad(
    path: str,
    *,
    perturbation_key: str = "target_gene",
    control_labels: tuple[str, ...] = ("non-targeting", "non-targeting control", "control"),
    eval_fraction: float = 0.1,
    max_cells_per_perturbation: int = 32,
    seed: int = 0,
    embeddings: torch.Tensor | None = None,
):
    """Aggregate an Arc Virtual Cell Challenge h5ad into a `PerturbationSet`.

    Does not download anything. Non-targeting cells become `control_mean`.
    Other `target_gene` values become single perturbations. The last
    `eval_fraction` of names, in lexical order, are the eval split. This is a
    local stand-in for the official challenge split, which lives in a separate
    file. Expression is pseudobulk means; at most `max_cells_per_perturbation`
    training cells are kept for the GAN.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "Reading h5ad requires anndata. Install with pip install 'alz-perturb-gan[vcc]'."
        ) from exc

    adata = ad.read_h5ad(path)
    if perturbation_key not in adata.obs.columns:
        raise ValueError(
            f"obs has no {perturbation_key!r} column. Arc's 2025 files use 'target_gene'."
        )
    labels = adata.obs[perturbation_key].astype(str)
    control_set = {item.lower() for item in control_labels}
    is_control = labels.str.lower().isin(control_set)
    if int(is_control.sum()) == 0:
        raise ValueError(
            f"no control cells found under {perturbation_key} with labels {control_labels}"
        )
    gene_names = [str(name) for name in adata.var_names]
    control_mean = _as_dense_mean(adata.X[is_control.to_numpy()])
    names = sorted(set(labels[~is_control]))
    if not names:
        raise ValueError("no perturbed cells found")
    n_eval = 0 if len(names) == 1 else max(1, int(round(len(names) * eval_fraction)))
    n_eval = min(n_eval, len(names) - 1)
    eval_names = names[-n_eval:] if n_eval else []
    train_names = names[:-n_eval] if n_eval else names
    generator = torch.Generator().manual_seed(seed)
    profiles: dict[str, torch.Tensor] = {}
    cells: dict[str, torch.Tensor] = {}
    index = labels.to_numpy()
    for name in names:
        positions = (index == name).nonzero()[0]
        profiles[name] = _as_dense_mean(adata.X[positions])
        if name not in train_names:
            continue
        keep = positions
        if len(positions) > max_cells_per_perturbation:
            choice = torch.randperm(len(positions), generator=generator)[:max_cells_per_perturbation]
            keep = positions[choice.numpy()]
        block = adata.X[keep]
        if hasattr(block, "toarray"):
            block = block.toarray()
        cells[name] = torch.tensor(block, dtype=torch.float32)
    return perturbation_set_from_groups(
        name="virtual-cell-challenge",
        source="virtual_cell_h5ad",
        gene_names=gene_names,
        control_mean=control_mean,
        profiles=profiles,
        train_names=train_names,
        eval_names=eval_names,
        embeddings=embeddings,
        cells=cells,
        notes=(
            f"Loaded {path}. Control labels {control_labels} were excluded from the "
            "perturbation library. Eval names are a lexical holdout, not the official "
            "Virtual Cell Challenge split. Cells are unpaired: the GAN conditions on "
            "the non-targeting mean."
        ),
    )
