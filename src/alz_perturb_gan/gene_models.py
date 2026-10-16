"""Component-separated feature arms and direct, optimizer-free model construction."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import torch
from torch import nn
from .particle_integration import make_recipe
from .models import ConditionalGenerator, ConditionalCritic, DecisionModel
from .gene_features import FeatureProjector

ARMS = ('B0', 'G', 'H', 'GH', 'G_frozen', 'shuffle')


@dataclass(frozen=True)
class ModelConfig:
    n_genes: int
    library_size: int
    hidden: int = 128
    z_dim: int = 16
    particles: int = 256
    seed: int = 42
    steps: int = 20000
    batch_size: int = 256
    lr: float = .0002
    arm: str = 'B0'
    shuffle_seed: int = 142
    frozen_generator_arm: str = "G"
    trainer: str = "ka2"  # legacy checkpoints omitted this field


    def __post_init__(self):
        if self.frozen_generator_arm not in ('B0','G'): raise ValueError('invalid frozen-generator source arm')
        if self.arm not in ARMS or min(self.n_genes, self.library_size, self.hidden, self.z_dim, self.particles, self.steps, self.batch_size) < 1 or self.lr <= 0:
            raise ValueError('invalid model/arm configuration')


def _seeded(seed, constructor):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return constructor()


class GeneGenerator(nn.Module):
    def __init__(self, config, table):
        super().__init__()
        enabled = config.arm in ('G', 'GH', 'shuffle') or (config.arm == 'G_frozen' and config.frozen_generator_arm == 'G')
        self.encoder = _seeded(config.seed + 101, lambda: FeatureProjector(table, config.library_size)) if enabled else None
        self.register_buffer('onehot', torch.eye(config.library_size))
        self.backbone = _seeded(config.seed, lambda: ConditionalGenerator(config.n_genes, config.library_size, config.z_dim, config.hidden))

    def forward(self, z, control, candidate_ids):
        embedding = self.encoder(candidate_ids) if self.encoder is not None else self.onehot[candidate_ids]
        return self.backbone(z, control, embedding)


class GeneHead(nn.Module):
    def __init__(self, config, table):
        super().__init__()
        self.encoder = _seeded(config.seed + 102, lambda: FeatureProjector(table, config.library_size)) if config.arm in ('H', 'GH', 'shuffle') else None
        self.backbone = _seeded(config.seed + 2, lambda: DecisionModel(config.n_genes, config.hidden))
        # Allocate the same adapter shape in every arm. It is inactive/frozen in profile-only arms.
        self.adapter = _seeded(config.seed + 103, lambda: nn.Linear(config.library_size, config.hidden, bias=False))
        self.adapter.requires_grad_(self.encoder is not None)

    def logits(self, control, profile, target, candidate_ids):
        p, t = profile - control, target - control
        hidden = self.backbone.net[0](torch.cat([p, t, p * t], -1))
        if self.encoder is not None:
            hidden = hidden + self.adapter(self.encoder(candidate_ids))
        # Target identity/features are never an input.
        return self.backbone.net[1:](hidden).squeeze(-1)

    def forward(self, control, profile, target, candidate_ids):
        return self.logits(control, profile, target, candidate_ids).sigmoid()


@dataclass
class GeneModels:
    config: ModelConfig
    generator: GeneGenerator
    critic: ConditionalCritic
    decision: GeneHead
    prior: nn.Module
    recipe: object
    feature_provenance: dict
    checkpoint_provenance: dict = field(default_factory=dict)
    training_protocol_sha256: str | None = None

    def modules(self):
        return {k: getattr(self, k) for k in ('generator', 'critic', 'decision', 'prior')}

    def to(self, device):
        for m in self.modules().values(): m.to(device)
        return self

    @property
    def device(self): return next(self.generator.parameters()).device

    def validate_study(self, study):
        protocol = study.protocol
        if self.config.library_size != len(protocol.library) or self.config.n_genes != len(study.control):
            raise ValueError('common library/expression dimensions differ')
        for encoder in (self.generator.encoder, self.decision.encoder):
            if encoder is not None and (encoder.names != protocol.library or encoder.gene_ids != protocol.gene_ids or encoder.fit_names != protocol.fit):
                raise ValueError('feature identity order or normalization FIT split differs')

    def counts(self):
        return {k: {'allocated': sum(p.numel() for p in m.parameters()),
                    'trainable': sum(p.numel() for p in m.parameters() if p.requires_grad)} for k, m in self.modules().items()}


def build_models(config, table=None, *, data_kind='real', resolved_recipe=None):
    needs = config.arm in ('G','H','GH','shuffle') or (config.arm == 'G_frozen' and config.frozen_generator_arm == 'G')
    if needs and (table is None or (not table.verified and data_kind != 'synthetic')):
        raise ValueError('real feature arms require a frozen verified table')
    if table is not None and len(table.names) != config.library_size:
        raise ValueError('feature library size mismatch')
    if config.arm == 'shuffle': table = table.shuffled(config.shuffle_seed)
    if resolved_recipe is None:
        recipe = make_recipe(config.trainer, total_steps=config.steps, batch_size=config.batch_size, z_dim=config.z_dim,
                            num_particles=config.particles, lr=config.lr, d_lr_mult=1., prior_lr_mult=1.)
    else:
        from particlegan import Recipe
        from dataclasses import fields
        from .particle_integration import verify_source
        verify_source()
        if set(resolved_recipe) != {f.name for f in fields(Recipe)}:
            raise ValueError('checkpoint must store every resolved recipe field')
        recipe = Recipe(**resolved_recipe)
        if (recipe.z_dim, recipe.num_particles, recipe.batch_size, recipe.total_steps, recipe.lr,
            recipe.critic_formulation) != (config.z_dim, config.particles, config.batch_size, config.steps,
                                          config.lr, config.trainer):
            raise ValueError('resolved checkpoint recipe differs from model config')
    generator = GeneGenerator(config, table)
    critic = _seeded(config.seed + 1, lambda: ConditionalCritic(config.n_genes, config.n_genes + config.library_size, config.hidden))
    decision = GeneHead(config, table)
    prior = _seeded(config.seed + 3, recipe.make_prior)
    if config.arm == 'G_frozen':
        generator.requires_grad_(False); critic.requires_grad_(False); prior.requires_grad_(False)
    return GeneModels(config, generator, critic, decision, prior, recipe, table.provenance if table else {'source': 'onehot'})
