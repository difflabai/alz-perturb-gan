"""Round-3 arms for improve_20261109: round-1 flags, round-2 plug-ins, round-3 option flags.

This module is a COPY of ``../r2/arms.py`` (itself a copy of round-1
``../arms.py``): the round-1 flags ``offset``, ``genemean``, ``donorpc`` and
``covars`` keep their exact round-1 code paths and semantics ("Round-1 flags"
at the end of this docstring) and every round-2 plug-in keeps its round-2
semantics.  Round 3 adds (PROTOCOL_R3.md, work items R1-R5):

* training/recipe option flags ``t.<name>@<value>`` (core; table below);
* generator-architecture flags ``g.<name>[@v]`` and critic-architecture flags
  ``d.<name>[@v]``, registered by plug-in files like any arm, with two new
  hooks (``generator_kwargs``, ``critic_module``) and per-cell critic blocks
  (``FeatureBlock.cell_dim``);
* read-out generation modes ``raw``, ``ema@D``, ``snapens`` (last-25% rule),
  ``snapens@K``, ``snap@U``, ``snapema@U:D``.

Arm strings and flags
=====================

Arm string = comma-joined flags sorted by Python string order (= ``LC_ALL=C
sort``), ``base`` when there are none.  A flag is ``name`` or ``name@value``.
Names match ``([tgd]\\.)?[a-z][a-z0-9_]*``.  The value text must already be in
canonical form (for float values ``'%g' % v``: ``0.5`` not ``0.50``, ``2`` not
``2.0``; integers without sign or leading zeros); ``parse_arm`` refuses
non-canonical text so the fit folder ``fits_r3/{arm}/{panel}/fold{k}/seed{s}/``
is unique per arm.  Example: ``genemean@0.5,t.anneal@0.1,t.steps@40000``.

* Round-1 names without a value mean exactly what they meant in round 1.
* ``genemean@s`` (core, not a plug-in): the round-1 genemean block with its
  scale multiplied by s (``s > 0``, ``s != 1``; use ``genemean`` for s = 1).
  Same reference bags, same s_gm; block = ``s * s_gm * mean_cells(F)``.
  ``covars@``, ``offset@`` and ``donorpc@`` are refused.
* ``t.*``: training/recipe options, defined HERE (``TRAIN_OPTIONS``); plug-ins
  cannot register a ``t.`` name.
* Any other name (plain, ``g.*`` or ``d.*``) is a plug-in arm: a class
  registered with ``@register`` in a file ``r3/arm_<something>.py``.
  ``arm_<name with '.' replaced by '_'>.py`` (e.g. ``arm_g_width.py`` for
  ``g.width``) is imported first; if the name is still unknown every
  ``r3/arm_*.py`` is imported (a file may register several arms and modes).
  Import errors of unrelated plug-in files are only reported when the
  requested name is not found.  Convention (not enforced): ``g.`` arms only
  use the generator hooks (c), ``d.`` arms only the critic hooks (d, d2).

Round-3 training/recipe option flags (``t.``)
==============================================

Every ``t.`` flag changes the run through ParticleGAN 0.9.7's OWN recipe
fields (``get_recipe('gan', trainer='bispf_optimized', ...)`` with the
released adapter's arguments plus the overrides below; the native ``Recipe``
validation applies) or through the caller-owned batch shape, never by
patching optimizer internals.  A flag whose value equals the t005 default is
refused (omit the flag), so each run has one name.  ``config.json`` records
every flag under ``departures`` with the resolved recipe fields it changed.
With no ``t.`` flag the recipe is exactly ``released_update.resolved_recipe(T005)``.

=================  ==========================  ===========================================================
flag               recipe field(s) / effect    semantics (t005 default in brackets)
=================  ==========================  ===========================================================
t.steps@N          total_steps = N; run        declared update horizon AND run length [20000].  N in
                   length N                    1..200000.  The native cosine schedule uses this horizon.
                                               ``trainer.py --updates U`` may only shorten the run
                                               (U <= N, smoke tests); the horizon stays N.
t.anneal@F         lr_floor = F,               native cosine annealing of the G and D learning rates
                   network_lr_floor = F        (network multiplier) and of the particle learning rate
                                               (prior multiplier) to F x base rate.  F in [0, 1)
                                               [1 = constant].  Native law
                                               (recipes.learning_rate_scale): the update that starts
                                               after k completed updates uses
                                               s(k) = 1 for k <= S*N, then
                                               s(k) = F + (1-F)/2 (1 + cos(pi (k - S N)/((1-S) N))),
                                               and s = F for k >= N.  S = lr_anneal_start.
t.annealstart@S    lr_anneal_start = S         start fraction S in [0, 1) [native default 0.6];
                                               requires t.anneal.
t.netfloor@F       network_lr_floor = F        G/D floor different from the prior floor (prior keeps
                                               t.anneal's F); requires t.anneal; F in [0, 1].
t.nethorizon@N     network_lr_horizon_cap = N  G/D anneal over min(total_steps, N) updates (start S*N,
                                               floor reached at N, then hold); the prior keeps the
                                               full horizon; requires t.anneal; N >= 1.
t.ema@D            ema_decay = D               native parameter EMA of generator and particle prior
                                               (UpdatePolicy.finish_step: avg = D avg + (1-D) live, from
                                               the initial weights, after every update; serve_average
                                               stays 0 so training is unchanged).  D in (0, 1).  The
                                               fit's DEFAULT generation (generated.npz, state
                                               ``final``) then uses the native EMA generator and prior
                                               (policy ``averages``); mode ``raw`` uses the live
                                               weights.
t.glr@M            lr = .012 M                 generator learning rate x M (M > 0, <= 100).
t.dlr@M            d_lr_mult = .018 M / lr     critic learning rate x M (critic rate = .018 M).
t.plr@M            prior_lr_mult = .06 M / lr  particle (prior row) learning rate x M (= .06 M).
                                               The three rates are composed in Decimal like
                                               released_update.resolved_recipe.
t.particles@N      num_particles = N           particle-table rows [512].  N in 65..8192.  For N <=
                                               1024 the released adapter regularises the whole table
                                               (released semantics).  For N > 1024 the released
                                               adapter's own rule regularises only the unique rows
                                               sampled in the update (<= batch rows), which puts the
                                               SBNP regulariser in its <=64-row regime (small repulsion
                                               margin, no active isotropy term): a different prior
                                               regulariser, recorded in ``departures``.  N <= 64 is
                                               refused (SBNP small-latent regime changes the weight).
t.zdim@N           z_dim = N                   particle / donor latent dimension [8]; generator donor
                                               input width follows.  N in 1..64.
t.batch@B          batch_size = B; caller      donors per batch, both half-updates [32]; B in 16..256.
                   batch shape B               The recipe's batch_size field is validated and recorded
                                               but read by no 0.9.7 runtime component; the BISPF
                                               penalty's batch-dependent gates do not fire for the
                                               150+-wide critic input (cooler queried at its default
                                               batch 128 for dim > 4; small-batch gauge cap only below
                                               16, hence B >= 16), so the update law is unchanged; only
                                               the number of donors per gradient changes.
t.bag@K            caller bag shape K          cells per bag (real and fake) [128]; K in 16..1024.
                                               Not a recipe field (the bag is the adapter's
                                               observation).  Reference-bag normalisations of
                                               genemean / pbulk / donorpc2 stay computed with
                                               128-cell bags (fixed constants).
t.reg@M            reg_coeff = 1 M             BISPF critic-penalty coefficient x M (M in (0,100]);
                                               added by integration (PROTOCOL_R3 addition 1).
t.kappa@M          reg_kappa = 1 M             BISPF gradient cap: the penalty acts on
                                               |dD/dx| / sqrt(critic input width) above kappa [1].
                                               M = sqrt(150/width) restores base's absolute cap
                                               (0.26 at 2,150 wide).  Integration addition.
t.priorreg@M       prior_reg = .01 M           SBNP particle-table regulariser weight x M.
                                               Integration addition.
=================  ==========================  ===========================================================

Native options that cannot be combined with ``bispf_optimized`` (checked
against recipes.py 0.9.7; the Recipe constructor raises):

* ``beta2_end`` (scheduled Adam beta2): "scheduled beta2 requires
  optimizer_family='adam'"; this recipe is zero-momentum DualNorm.  No
  equivalent: DualNorm has no second moment.
* ``reg_coeff_end`` (scheduled penalty coefficient): "requires a fixed R1/R2
  or BCap reg_arm", and the bispf formulation refuses any reg_arm.
* ``optimizer_momentum`` 0.5/0.9: "constraint_geometry requires zero-momentum
  full DualNorm" (bispf_optimized uses direction_blend).  Nearest valid
  alternative for smoothing the iterate: t.ema / shadow EMA / snapshots.
* ``serve_average``: accepted by Recipe validation but INERT here: serving
  the average needs the continuous (dv12) stationarity table tester
  (``UpdatePolicy._serve_settled``), and ``continuous_policy`` is refused for
  plain optimizer families ("plain optimizers do not implement a continuous
  update policy"); without a tester the averaging rate falls back to
  1 - ema_decay.  Nearest: t.ema.
* ``input_noise_std`` / ``output_noise_std`` (scheduled noise): valid recipe
  fields, but the released adapter asserts zero noise (it does not apply
  the native noise in its caller-owned loop); not offered.
* ``lr_schedule='exponential'`` is valid natively but not requested; not
  offered.

Generation modes added in round 3 (built in, see generate.py)
=============================================================

``raw`` (live final weights; equals the default for fits without t.ema),
``ema@D`` (shadow EMA at decay D of the final update; D one of the fit's
``shadow_ema_decays``, round-3 fits only), ``snapens`` (M split evenly over
the snapshots at updates u >= 0.75 U and the raw final state, U = run
length; on a round-2 fit: its four snapshots + final, exactly the round-2
mode), ``snapens@K`` (the last K snapshots + raw final, K >= 1), ``snap@U``
(all M from snapshot U), ``snapema@U:D`` (all M from the shadow EMA at decay D
stored in snapshot U).  State names a mode plan may return:
``final`` (default served state: native EMA for t.ema fits, else the live
weights), ``raw``, ``ema_<D>``, ``snap_<U>``, ``snapema_<U>_<D>``.

Plug-in API (``Arm`` and ``FeatureBlock`` below; worked example:
``r3/arm_example.py``: flags ``example`` / ``example@s`` (round-2 hooks),
``g.example@h`` (hooks c/c2), ``d.example`` (hooks d/d2 with a per-cell
block), mode ``exshrink@v``)
==========================================================================

A plug-in file does::

    import arms
    from arms import Arm, FeatureBlock, register, register_mode, canonical_float

    @register
    class MyArm(Arm):
        name = 'myarm'              # flag name
        value_policy = 'optional'   # 'none' | 'optional' | 'required'
        requires = ()               # other flag NAMES that must be present
        conflicts = ()              # flag NAMES that must be absent
        needs_Z = False             # True -> ctx.Z (standardised train cells) is built
        @classmethod
        def parse_value(cls, text):  # -> (value, canonical_text); default float
            return canonical_float(text, lo=0.0, hi=1.0)

One instance is created per Context (training or generation) with
``self.value`` (parsed value or None), ``self.flag`` (canonical flag string),
``self.name`` and ``self.info`` (a JSON-serialisable dict written to
config.json ``r2_arms[flag]``; the key name is kept from round 2 for every
plug-in, including g./d. arms, which are also listed under ``departures``).
Hooks, all optional (default = no effect).
Plug-in arms run in sorted flag order; every hook output is shape-checked.

Setup (train-only fitting)
    ``setup(self, ctx) -> None``
    Called once in ``trainer.Context.__init__`` after the released data are
    loaded and BEFORE conditions and critic features are built.  Available:
    ``ctx.X`` (train counts [cells,G] float32 tensor on ctx.device),
    ``ctx.by`` (list of int row arrays per train donor, cache order),
    ``ctx.ids``, ``ctx.raw_c`` (float64 [n_train,7], bench_core encoding),
    ``ctx.fit_labels``, ``ctx.mean``/``ctx.sd`` (gene standardisation),
    ``ctx.W`` ([G,50] = W / pc_sd), ``ctx.trainpcs`` ([cells,50]),
    ``ctx.mu``/``ctx.theta``, ``ctx.fitlibs``, ``ctx.Z`` (only when some arm
    has needs_Z or genemean/donorpc is on), ``ctx.age_mean``/``ctx.age_sd``,
    ``ctx.genes``, ``ctx.n_genes``, ``ctx.fold``, ``ctx.panel``,
    ``ctx.flags``, ``ctx.device``, ``ctx.heldout_ids``, ``ctx.topt``
    (round-3 TrainOptions), ``ctx.batch_size`` (donors per batch) and
    ``ctx.bag`` (cells per bag).  setup must be a
    deterministic function of the TRAIN donors of the cache (it is recomputed
    at resume and at generation and compared with the checkpoint, hook (f));
    it must NOT read held-out counts and must not use any training random
    stream or the torch global RNG.  For fixed randomness use
    ``np.random.default_rng(np.random.SeedSequence([20261109, zlib.crc32(b'myarm'), k]))``.

(a) Per-donor condition matrix
    ``conditions(self, ctx, raw_c, cond) -> np.ndarray[n,7] float32``
    ``raw_c``: float64 [n,7] raw bench_core encoding (Braak/6, Thal/5,
    CERAD/3, ADNC/3, Male, APOE4 dose, raw age) of n donors; ``cond``: the
    condition built so far (round-1 ``build_conditions`` for the flags, then
    earlier plug-ins).  Return a new array (do not modify ``cond``).  Used for
    the train donors (``ctx.conditions``) AND for generation targets (train +
    held-out donors), so it must be a function of raw_c rows plus train-fitted
    setup state.  The condition is always 7 wide (generator donor layer and
    critic take [B,7]); slot 3 (ADNC/3) carries the stage, slots 0-2 are 0 in
    every round-1 arm and are free for plug-ins, slots 4-6 are used by covars.

(b) Per-batch condition transform during training (e.g. condition dropout)
    ``batch_condition(self, ctx, c, ds, rng, phase) -> Tensor[B,7]``
    ``c``: [B,7] float32 tensor on ctx.device (the batch donors' rows of
    ctx.conditions after earlier plug-ins); ``ds``: numpy int [B] train-donor
    indices of the batch; ``rng``: this arm's own ``numpy.random.Generator``;
    ``phase``: ``'critic'`` (first batch of an update) or ``'generator'``
    (second batch).  The returned condition is used for BOTH the generator
    (fake bags) and the critic (real and fake bags) in that half-update.
    Return a new tensor.  ``rng`` is created only for arms that override this
    hook: ``default_rng(SeedSequence([stream_root, 7, fold, seed,
    crc32(name), 0]))`` (``ctx.arm_seed(name, 0)``), independent of the 11
    released training streams, so arms without this hook keep the round-1
    streams bit for bit; its bit_generator state is checkpointed in
    recovery.pt / final.pt (``arm_streams``) and restored on resume.

(c) Generator class and parameters
    ``generator_class(self, ctx, cls) -> type``
    ``cls`` starts as ``HookedCountGenerator`` (a pg_model.CountGenerator
    subclass whose ``parameters_nb`` is split into overridable steps, forward
    bitwise identical to CountGenerator):

        film(z, c) -> (scale [B,h], shift [B,h])            donor FiLM
        cell_hidden(noise, scale, shift, z, c) -> h [B,K,h]
        logits(h, z, c) -> [B,K,G]                           before softmax
        dispersion(z, c) -> theta (broadcastable to [B,K,G])

    Return a subclass of ``cls`` (call ``super()`` to compose with other
    plug-ins; each plug-in wraps the class returned by the previous one, in
    sorted flag order).  The constructor signature stays
    ``(mean, theta, zd=4, zc=8, h=128)``; capture arm data by closure.  Add
    parameters in ``__init__`` AFTER ``super().__init__`` without drawing
    from the torch global RNG (zeros, constants, or a private
    ``torch.Generator().manual_seed(ctx.arm_seed(name, k))``), so the released
    parameters get bitwise the round-1 initial tensors.  New parameters land
    in the generator optimizer group (generator learning rate, released
    DualNorm update; no new loss term).  Not combinable with round-1
    ``offset`` (refused by parse_arm).
    ``init_generator(self, ctx, G) -> None``: called once on the constructed
    generator (CPU, before ``.to(device)``), inside ``torch.no_grad``, for
    every arm that overrides it (also when no arm changes the class); may
    initialise parameters the class added.  Never use the global RNG here.
    ``generator_grad_params(self, G) -> list[nn.Parameter]``: parameters
    asserted to receive a nonzero gradient at update 0 (default none).
    The released parameters asserted at update 0 are
    ``G.step0_grad_params()`` (HookedCountGenerator: ``out[-1].weight`` and
    ``log_theta``) plus ``P.z``; a class that replaces the head overrides
    ``step0_grad_params``.

(c2) Generator constructor arguments (round 3; width, cell latent, depth...)
    ``generator_kwargs(self, ctx, kwargs) -> dict``
    ``kwargs`` starts as ``{'zd': recipe.z_dim, 'zc': 8, 'h': 128}`` (the
    released CountGenerator arguments: donor latent width, cell-noise width,
    hidden width) after earlier plug-ins, in sorted flag order; return a new
    dict.  ``zd`` must stay ``recipe.z_dim`` (use ``t.zdim``).  Extra keys are
    allowed when the arm's ``generator_class`` accepts them (e.g. ``depth``).
    Any arm overriding (c) or (c2) makes the base class HookedCountGenerator
    (bitwise CountGenerator when nothing else changes).  Construction runs
    under ``torch.manual_seed(generator_init)``: a g. arm that changes the
    shapes of released layers changes their initial draws (that is the
    departure); it must not draw from any other stream.

(d) Critic feature blocks (appended after the released 150 features)
    ``feature_blocks(self, ctx) -> list[FeatureBlock]`` (called once, after
    setup).  Critic input = [150 released bag features, round-1 genemean or
    donorpc block if that flag is on, then each plug-in's blocks in sorted
    flag order and list order]; the critic is ``WideVectorCritic(total
    width)`` (released VectorCritic layers, h=128, wider input) whenever the
    width is not 150.  A FeatureBlock has ``name`` (str, unique within the
    arm), ``width`` (int > 0), ``info`` (JSON dict) and:

        real(self, ctx, idx, ds) -> Tensor[B,width]
            idx: [B,cells] long tensor of train cell rows (use ctx.X[idx],
            ctx.Z[idx], ctx.trainpcs[idx]); ds: numpy [B] train-donor index.
        fake(self, ctx, cf, rate, counts) -> Tensor[B,width]
            cf: straight-through critic_features [B,cells,G]
            (standardised log1p CP10k of the Poisson counts with the rate
            gradient); rate: Gamma-NB rate [B,cells,G] WITH gradient;
            counts: Poisson counts [B,cells,G] (no gradient).  A
            linear-in-counts block uses ``counts + (rate - rate.detach())``.
        state(self) -> dict   tensors/arrays/numbers for the consistency check.

    Real and fake must be the same function of (counts) for the same cells;
    the trainer asserts at update 0 that every block's critic input columns
    receive a nonzero critic gradient.  Feature blocks are not used at
    generation.

    Per-cell blocks (round 3, for cell-level critics).  A block with
    ``cell_dim = d`` (int, default None) is a per-cell block: ``width`` must
    be ``ctx.bag * d`` and real/fake return the per-cell features
    [B, cells, d] flattened cell-major (``x.reshape(B, cells * d)``).  Real
    cells come as row indices ``idx`` into the ctx tensors (``ctx.X[idx]``
    counts, ``ctx.Z[idx]`` standardised, ``ctx.trainpcs[idx]`` PCs), fake
    cells as the straight-through ``cf`` [B, cells, G] plus ``rate`` and
    ``counts``.  The per-cell features are part of the critic input vector
    x on purpose: the released BISPF critic penalty acts on dD/dx, so every
    input the critic sees stays under the native penalty.  A cell-level
    critic (d2) reads the block back with ``layout`` and reshapes it.
    Note: wide inputs change the penalty's RMS gradient-norm units
    (sqrt(dim) normalisation over the whole x).

(d2) Critic module (round 3: width, depth, block-wise, cell-level critics)
    ``critic_module(self, ctx, layout) -> nn.Module``
    At most ONE arm may override it (else ValueError).  Called in
    ``trainer.build`` under ``torch.manual_seed(critic_init)`` (the released
    critic stream; drawing from the global RNG here is expected), instead of
    VectorCritic / WideVectorCritic.  ``layout`` is a ``CriticLayout``:
    ``n_features`` (width of x), ``cond_dim`` (7), ``hidden`` (128, the
    released width), ``batch`` and ``cells`` (ctx.batch_size, ctx.bag) and
    ``blocks``: tuple of ``BlockSpec(flag, name, lo, hi, cell_dim)`` in
    column order: ``(None, 'released', 0, 150, None)``, then the round-1
    ``genemean``/``donorpc`` block if on (flag = that flag), then every
    plug-in block; ``layout.block(name, flag=None)`` finds one.  The module
    must implement ``forward(x [B, n_features], c [B, 7]) -> [B]`` (one logit
    per bag), be a deterministic function of (x, c) (no RNG, no dropout, no
    batch statistics or running buffers: the penalty re-evaluates it), and
    keep all trainable parameters in ``parameters()`` (they all go to the
    released critic optimizer: DualNorm, critic learning rate).  At update 0
    the trainer checks, for every non-released block, that dD/dx is nonzero
    on its columns (autograd.grad on a detached copy of the real batch; no
    parameter .grad is touched, no RNG drawn).

(e) Generation-time hooks
    ``generation_condition(self, ctx, raw_c, cond, is_heldout) -> np.ndarray[n,7]``
    Condition used by default generation (``generated.npz``) for each target
    donor, applied after the (a) chain to the targets' raw_c; default
    identity.  ``is_heldout``: bool [n].
    Named generation modes (``generated_{mode}.npz``) are registered with::

        register_mode(name, plan, *, value_policy='none', parse_value=None,
                      prepare=None, doc='')

    ``plan(ctx, value, target, rng, states) -> (cond, state_names)``
        target: ``GenTarget`` (index, donor, is_heldout, raw_c float64[7],
        stage, dev_index, default_cond float32[7], M, K, updates = the fit's
        run length or None); rng: numpy
        Generator seeded from SeedSequence([20261109, 5, fold, dev_index,
        crc32(mode name)]) (depends on fold, target and mode name only:
        same draws for every arm, seed and fit); states: tuple of model
        states present in the fit (``'snap_<u>'`` ascending, then (round 3)
        ``'snapema_<u>_<d>'``, ``'raw'``, ``'ema_<d>'``, then ``'final'``).
        Return cond float32 [M,7] (per generated donor) and a
        length-M sequence of state names.  Generated donors that use the same
        state should be contiguous (each contiguous run is generated in chunks
        of 8 in order of m; the per-target streams and library draws are the
        default ones, so a plan returning (default_cond x M, 'final' x M)
        reproduces generated.npz bitwise).
    ``prepare(ctx, value, state_name, G, P) -> None`` (optional): called once
        per loaded state on a private copy of the generator and prior, inside
        ``torch.no_grad``, before generation (e.g. scale a learned gain).
    Modes never retrain.  Built-in modes (here): ``snapens``, ``snapens@K``,
    ``snap@U``, ``raw``, ``ema@D``, ``snapema@U:D`` (see "Generation modes
    added in round 3" above; split rule: M // n + (1 if i < M % n), in
    ascending update order, contiguous).  A mode registered in a
    plug-in file is found by importing ``arm_<mode name>.py`` or, failing
    that, every ``arm_*.py``.  Mode names share the flag syntax
    (``name@value``, canonical value text).

(f) Checkpointed arm state
    ``state(self) -> dict`` (plus every block's ``state()``): deterministic
    tensors / numpy arrays / numbers / lists recomputed from the cache by
    setup; stored in recovery.pt, final.pt and snap_*.pt under
    ``arm_state['r2'][flag] = {'arm': ..., 'blocks': {name: ...}}`` and
    compared exactly (on CPU) with the recomputed value at resume and
    generation (mismatch -> ValueError).
    ``runtime_state(self)`` / ``load_runtime_state(self, state)``: optional
    mutable state that changes during training (default None); stored in
    recovery.pt/final.pt ``arm_runtime`` and restored on resume.

Worked example of the round-3 hooks (``r3/arm_example.py``)::

    @register
    class GExample(Arm):                       # flag g.example@h
        name = 'g.example'; value_policy = 'required'
        @classmethod
        def parse_value(cls, text): return canonical_int(text, lo=8, hi=1024)
        def generator_kwargs(self, ctx, kw): return {**kw, 'h': self.value, 'depth': 2}
        def generator_class(self, ctx, cls):
            class Deeper(cls):                 # one extra hidden layer in the head
                def __init__(self, mean, theta, zd=4, zc=8, h=128, depth=1):
                    super().__init__(mean, theta, zd=zd, zc=zc, h=h)
                    self.extra = nn.Sequential(*[nn.Sequential(nn.SiLU(), nn.Linear(h, h)) for _ in range(depth - 1)])
                def logits(self, h, z, c): return super().logits(self.extra(h) + h, z, c)
            return Deeper

    @register
    class DExample(Arm):                       # flag d.example
        name = 'd.example'
        def feature_blocks(self, ctx): return [CellPCBlock(ctx.bag)]   # cell_dim = 8
        def critic_module(self, ctx, layout):
            return CellSetCritic(layout)       # phi over the bag features + pooled per-cell MLP


Round-1 flags (copied code; semantics unchanged)
================================================

base
    Released formulation: CountGenerator (output bias = log pooled mean
    proportions, moment theta init), 150 bag features from 50 PCs
    (critic.bag_features), critic.VectorCritic(h=128), stage-only condition
    (stage_law.mask_adnc semantics), uniform donor sampling, Poisson forward
    with straight-through rate features (pg_model.critic_features).

offset
    ``OffsetCountGenerator``: a learnable table ``stage_offset`` [4, G] is
    added to the output logits before the softmax, row selected by the ADNC
    stage round(3*c[:,3]) and applied as ``one_hot(stage, 4) @ stage_offset``.
    Initialisation from TRAIN donors only (``offset_init``):

        C_d   = summed panel counts of donor d,  N_d = sum_g C_dg
        a     = 1 (pseudocount, counts per gene)
        q_g   = (sum_d C_dg + a) / (sum_d N_d + a G)          pooled proportions
        r_dg  = log((C_dg + a)/(N_d + a G)) - log q_g          donor log-ratios
        m_g   = mean over all n train donors of r_dg
        r_sg  = mean_{d in s} r_dg - m_g                       donor-averaged stage effect
        v_g   = pooled within-stage variance of r_dg
        e_s   = 1/n_s - 1/n
        tau2  = max(1e-6, mean over (s with n_s>0, g) of (r_sg^2 - v_g e_s))
        w_sg  = tau2 / (tau2 + v_g e_s)                        (0 when n_s = 0)
        offset_sg = w_sg * r_sg

genemean
    Critic input = [150 released features, s_gm * mean_cells(F)] where F is
    the SAME straight-through critic_features tensor [B, cells, G] whose PCA
    projection feeds the 150 features; for real bags F = standardize_counts
    (real counts).  s_gm = sqrt(E||f150||^2 / E||mean_cells F||^2) from 256
    real train bags (numpy default_rng(20261109)).  WideVectorCritic.

donorpc
    Critic input = [150 released features, ((mean_cells F) - mu) V / sd] with
    the top-K (K = min(16, n_train - 1)) right singular vectors of the train
    donor mean standardised profiles.  Mutually exclusive with genemean.

covars
    Condition slots: 3 = ADNC/3 (as released), 4 = Male, 5 = APOE4 dose, 6 =
    (age - mean)/sd with train-donor population mean/SD.
"""
from __future__ import annotations

import importlib
import math
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
# Plug-ins do ``import arms``; make sure that is THIS module even if it was
# imported under another name.
sys.modules.setdefault('arms', sys.modules[__name__])

ROUND1_FLAGS = ('covars', 'donorpc', 'genemean', 'offset')
FLAGS = ROUND1_FLAGS
OFFSET_PSEUDOCOUNT = 1.0
GENEMEAN_REF_BAGS = 256
GENEMEAN_REF_SEED = 20261109
DONORPC_MAX_K = 16
ARM_STREAM_TAG = 7      # SeedSequence([stream_root, 7, fold, seed, crc32(name), purpose])
MODE_STREAM_TAG = 5     # SeedSequence([20261109, 5, fold, dev_index, crc32(mode)])
GEN_ROOT = 20261109
_NAME_RE = re.compile(r'^(?:[tgd]\.)?[a-z][a-z0-9_]*$')   # flag names (round 3: optional t./g./d. prefix)
_MODE_RE = re.compile(r'^[a-z][a-z0-9_]*$')                # generation-mode names (no prefix)
TRAIN_PREFIX = 't.'


def crc(name):
    return zlib.crc32(str(name).encode())


# =========================================================================== flags / values
def split_flag(flag):
    """'name@value' -> ('name', 'value'); 'name' -> ('name', None)."""
    name, sep, vtext = str(flag).partition('@')
    if not _NAME_RE.match(name) or (sep and not vtext) or '@' in vtext or ',' in vtext or any(ch.isspace() for ch in vtext):
        raise ValueError(f'bad flag {flag!r}: want name or name@value, name ([tgd].)?[a-z][a-z0-9_]*')
    return name, (vtext if sep else None)


def flag_name(flag):
    return split_flag(flag)[0]


def canonical_float(text, lo=None, hi=None, *, lo_open=False, hi_open=False):
    """Parse a float value; canonical text is '%g' % v.  Returns (v, text)."""
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f'value {text!r} is not a number') from None
    if not math.isfinite(v):
        raise ValueError(f'value {text!r} must be finite')
    if lo is not None and (v < lo or (lo_open and v == lo)):
        raise ValueError(f'value {v} below {"(" if lo_open else "["}{lo}')
    if hi is not None and (v > hi or (hi_open and v == hi)):
        raise ValueError(f'value {v} above {hi}{")" if hi_open else "]"}')
    return v, '%g' % v


def canonical_int(text, lo=None, hi=None):
    if not re.fullmatch(r'-?[0-9]+', str(text)):
        raise ValueError(f'value {text!r} is not an integer')
    v = int(text)  # canonical text is str(v): parse_arm refuses '007' or '-0' as non-canonical
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f'value {v} out of range [{lo},{hi}]')
    return v, str(v)


# =========================================================================== round-3 training options (t.)
T005_RATES = {'generator_lr': '0.012', 'critic_lr': '0.018', 'particle_lr': '0.06'}   # decimal text of candidate t005
T005_PENALTY = {'reg_coeff': '1', 'reg_kappa': '1', 'prior_reg': '0.01'}            # decimal text of candidate t005 (integration, PROTOCOL_R3 addition 1)


@dataclass(frozen=True)
class TrainOptions:
    """Resolved round-3 training/recipe options (defaults = candidate t005 as released)."""
    steps: int = 20000
    anneal: float = None
    annealstart: float = None
    netfloor: float = None
    nethorizon: int = None
    ema: float = None
    glr: float = 1.0
    dlr: float = 1.0
    plr: float = 1.0
    particles: int = 512
    zdim: int = 8
    batch: int = 32
    bag: int = 128
    reg: float = 1.0
    kappa: float = 1.0
    priorreg: float = 1.0

    @property
    def is_default(self):
        return self == TrainOptions()

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _f01(text):
    return canonical_float(text, lo=0.0, hi=1.0, hi_open=True)


def _mult(text):
    return canonical_float(text, lo=0.0, lo_open=True, hi=100.0)


TRAIN_OPTIONS = {
    # name: (parser, requires, doc)
    'steps': (lambda t: canonical_int(t, lo=1, hi=200000), (), 'declared total_steps horizon and run length'),
    'anneal': (_f01, (), 'lr_floor = network_lr_floor = F (native cosine to F)'),
    'annealstart': (_f01, ('anneal',), 'lr_anneal_start = S'),
    'netfloor': (lambda t: canonical_float(t, lo=0.0, hi=1.0), ('anneal',), 'network_lr_floor = F (G/D only)'),
    'nethorizon': (lambda t: canonical_int(t, lo=1, hi=200000), ('anneal',), 'network_lr_horizon_cap = N'),
    'ema': (lambda t: canonical_float(t, lo=0.0, lo_open=True, hi=1.0, hi_open=True), (), 'ema_decay = D; default generation uses the native EMA'),
    'glr': (_mult, (), 'generator lr x M'),
    'dlr': (_mult, (), 'critic lr x M'),
    'plr': (_mult, (), 'particle lr x M'),
    'particles': (lambda t: canonical_int(t, lo=65, hi=8192), (), 'num_particles = N'),
    'zdim': (lambda t: canonical_int(t, lo=1, hi=64), (), 'z_dim = N'),
    'batch': (lambda t: canonical_int(t, lo=16, hi=256), (), 'donors per batch (recipe batch_size and caller batch)'),
    'bag': (lambda t: canonical_int(t, lo=16, hi=1024), (), 'cells per bag (caller bag)'),
    # integration (PROTOCOL_R3 addition 1): critic-penalty and prior-regulariser strength, recipe fields
    'reg': (_mult, (), 'reg_coeff = 1 x M (BISPF critic penalty coefficient)'),
    'kappa': (_mult, (), 'reg_kappa = 1 x M (BISPF gradient cap kappa)'),
    'priorreg': (_mult, (), 'prior_reg = .01 x M (SBNP particle regulariser weight)'),
}


def is_train_flag(flag):
    return str(flag).startswith(TRAIN_PREFIX)


def _parse_train_flag(flag):
    name, vtext = split_flag(flag)
    key = name[len(TRAIN_PREFIX):]
    if key not in TRAIN_OPTIONS:
        raise ValueError(f'unknown training option {name!r}; known: {sorted(TRAIN_PREFIX + k for k in TRAIN_OPTIONS)}')
    if vtext is None:
        raise ValueError(f'training option {name!r} needs a value ({name}@value)')
    v, canon = TRAIN_OPTIONS[key][0](vtext)
    if canon != vtext:
        raise ValueError(f'training option value {vtext!r} is not canonical; write {name}@{canon}')
    if getattr(TrainOptions(), key) == v:
        raise ValueError(f'{flag} is the t005 default; omit the flag')
    return key, v


def train_options(flags):
    """TrainOptions from the t. flags of a (validated) flag tuple."""
    vals = dict(_parse_train_flag(f) for f in flags if is_train_flag(f))
    return TrainOptions(**vals)


def train_flags(flags):
    return tuple(f for f in flags if is_train_flag(f))


def recipe_overrides(topt):
    """Recipe fields for get_recipe('gan', trainer='bispf_optimized', ...) (the released adapter's arguments + t. options).

    Learning rates are composed in Decimal exactly like released_update.resolved_recipe:
    lr = .012*glr, d_lr_mult = (.018*dlr)/lr, prior_lr_mult = (.06*plr)/lr.
    """
    from decimal import Decimal
    g = Decimal(T005_RATES['generator_lr']) * Decimal(repr(float(topt.glr)))
    d = Decimal(T005_RATES['critic_lr']) * Decimal(repr(float(topt.dlr)))
    pr = Decimal(T005_RATES['particle_lr']) * Decimal(repr(float(topt.plr)))
    q = {'total_steps': topt.steps, 'batch_size': topt.batch, 'num_particles': topt.particles, 'z_dim': topt.zdim,
         'lr': float(g), 'd_lr_mult': float(d / g), 'prior_lr_mult': float(pr / g)}
    for key, field in (('reg', 'reg_coeff'), ('kappa', 'reg_kappa'), ('priorreg', 'prior_reg')):
        q[field] = float(Decimal(T005_PENALTY[field]) * Decimal(repr(float(getattr(topt, key)))))
    if topt.anneal is not None:
        q['lr_floor'] = float(topt.anneal)
        q['network_lr_floor'] = float(topt.anneal if topt.netfloor is None else topt.netfloor)
        if topt.annealstart is not None:
            q['lr_anneal_start'] = float(topt.annealstart)
        if topt.nethorizon is not None:
            q['network_lr_horizon_cap'] = int(topt.nethorizon)
    if topt.ema is not None:
        q['ema_decay'] = float(topt.ema)
    return q


_DEPARTURE_FIELDS = {'steps': ('total_steps',), 'anneal': ('lr_floor', 'network_lr_floor'), 'annealstart': ('lr_anneal_start',),
                     'netfloor': ('network_lr_floor',), 'nethorizon': ('network_lr_horizon_cap',), 'ema': ('ema_decay',),
                     'glr': ('lr', 'd_lr_mult', 'prior_lr_mult'), 'dlr': ('d_lr_mult',), 'plr': ('prior_lr_mult',),
                     'particles': ('num_particles',), 'zdim': ('z_dim',), 'batch': ('batch_size',), 'bag': (),
                     'reg': ('reg_coeff',), 'kappa': ('reg_kappa',), 'priorreg': ('prior_reg',)}


def train_departures(flags, recipe_dict):
    """{t.flag: {'option', 'value', 'recipe_fields': {field: resolved value}, 'doc'}} for config.json ``departures``."""
    out = {}
    topt = train_options(flags)
    for f in train_flags(flags):
        key, v = _parse_train_flag(f)
        row = {'option': key, 'value': v, 'doc': TRAIN_OPTIONS[key][2],
               'recipe_fields': {k: recipe_dict.get(k) for k in _DEPARTURE_FIELDS[key]}}
        if key == 'bag':
            row['caller_fields'] = {'cells_per_bag': v}
        if key == 'batch':
            row['caller_fields'] = {'donors_per_batch': v}
        if key == 'particles' and v > 1024:
            row['prior_regulariser_rows'] = ('sampled unique rows (released adapter rule for num_particles > 1024): '
                                             'SBNP runs in its <=64-row regime (small repulsion margin, no active isotropy)')
        if key == 'ema':
            row['default_generation_state'] = 'native EMA (policy averages); mode raw = live weights'
        if key == 'steps':
            row['run_length'] = v
        out[f] = row
    if topt.anneal is not None:
        for f in out:
            if out[f]['option'] in ('anneal', 'annealstart', 'netfloor', 'nethorizon'):
                out[f]['schedule'] = {'lr_schedule': recipe_dict.get('lr_schedule', 'cosine'), 'lr_anneal_start': recipe_dict.get('lr_anneal_start'),
                                      'lr_floor': recipe_dict.get('lr_floor'), 'network_lr_floor': recipe_dict.get('network_lr_floor'),
                                      'network_lr_horizon_cap': recipe_dict.get('network_lr_horizon_cap'),
                                      'total_steps': recipe_dict.get('total_steps')}
    return out


# =========================================================================== plug-in API
class FeatureBlock:
    """One critic feature block (hook d).  Subclass and set name/width (and cell_dim for a per-cell block)."""
    name = None
    width = None
    cell_dim = None   # round 3: int d for a per-cell block of width ctx.bag * d (cell-major flattening)

    def __init__(self):
        self.info = {}

    def real(self, ctx, idx, ds):
        raise NotImplementedError

    def fake(self, ctx, cf, rate, counts):
        raise NotImplementedError

    def state(self):
        return {}


class Arm:
    """Base class of a round-2 plug-in arm.  See the module docstring."""
    name = None
    value_policy = 'none'      # 'none' | 'optional' | 'required'
    requires = ()
    conflicts = ()
    needs_Z = False

    @classmethod
    def parse_value(cls, text):
        return canonical_float(text)

    def __init__(self, value, flag):
        self.value = value
        self.flag = flag
        self.info = {}

    # ---- hooks (defaults: no effect)
    def setup(self, ctx):
        pass

    def conditions(self, ctx, raw_c, cond):
        return cond

    def batch_condition(self, ctx, c, ds, rng, phase):
        return c

    def generator_class(self, ctx, cls):
        return cls

    def generator_kwargs(self, ctx, kwargs):
        return kwargs

    def critic_module(self, ctx, layout):
        raise NotImplementedError

    def init_generator(self, ctx, G):
        pass

    def generator_grad_params(self, G):
        return []

    def feature_blocks(self, ctx):
        return []

    def generation_condition(self, ctx, raw_c, cond, is_heldout):
        return cond

    def state(self):
        return {}

    def runtime_state(self):
        return None

    def load_runtime_state(self, state):
        pass

    # ---- introspection used by the trainer
    @classmethod
    def overrides(cls, hook):
        return getattr(cls, hook) is not getattr(Arm, hook)


_ARMS = {}
_MODES = {}
_PLUGIN_ERRORS = {}
_SCANNED = [False]


def register(cls):
    """Class decorator: register a plug-in Arm subclass under ``cls.name``."""
    if not (isinstance(cls, type) and issubclass(cls, Arm)):
        raise TypeError('register expects an Arm subclass')
    name = cls.name
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f'bad arm name {name!r}')
    if name in ROUND1_FLAGS:
        raise ValueError(f'{name!r} is a round-1 flag; plug-ins cannot redefine it')
    if name.startswith(TRAIN_PREFIX):
        raise ValueError(f'{name!r}: t. flags are core training options (arms.TRAIN_OPTIONS); plug-ins cannot register them')
    if cls.value_policy not in ('none', 'optional', 'required'):
        raise ValueError('value_policy must be none, optional or required')
    old = _ARMS.get(name)
    if old is not None and (old.__module__, old.__qualname__) != (cls.__module__, cls.__qualname__):
        raise ValueError(f'arm {name!r} registered twice ({old.__module__} and {cls.__module__})')
    _ARMS[name] = cls
    return cls


@dataclass(frozen=True)
class ModeSpec:
    name: str
    plan: object
    value_policy: str = 'none'
    parse_value: object = None
    prepare: object = None
    doc: str = ''
    module: str = ''


@dataclass
class GenTarget:
    """What a mode plan sees about one generation target donor."""
    index: int
    donor: str
    is_heldout: bool
    raw_c: np.ndarray
    stage: int
    dev_index: int
    default_cond: np.ndarray
    M: int
    K: int
    updates: int = None     # round 3: the fit's run length (config 'updates'), None if unknown


def register_mode(name, plan, *, value_policy='none', parse_value=None, prepare=None, doc=''):
    """Register a generation mode (hook e).  See the module docstring."""
    if not isinstance(name, str) or not _MODE_RE.match(name):
        raise ValueError(f'bad mode name {name!r}')
    if value_policy not in ('none', 'optional', 'required'):
        raise ValueError('value_policy must be none, optional or required')
    mod = getattr(plan, '__module__', '')
    old = _MODES.get(name)
    if old is not None and old.module != mod:
        raise ValueError(f'mode {name!r} registered twice ({old.module} and {mod})')
    _MODES[name] = ModeSpec(name, plan, value_policy, parse_value or (lambda t: canonical_float(t)), prepare, doc, mod)
    return plan


def _import_plugin(path):
    modname = path.stem
    if modname in sys.modules:
        return
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        importlib.import_module(modname)
    except Exception as exc:  # noqa: BLE001 - reported when the requested name is missing
        sys.modules.pop(modname, None)
        _PLUGIN_ERRORS[modname] = repr(exc)


def _scan_plugins():
    if _SCANNED[0]:
        return
    for p in sorted(HERE.glob('arm_*.py')):
        _import_plugin(p)
    _SCANNED[0] = True


def _lookup(table, name, kind):
    if name not in table:
        p = HERE / f'arm_{name.replace(".", "_")}.py'
        if p.exists():
            _import_plugin(p)
    if name not in table:
        _scan_plugins()
    if name not in table:
        errs = '; '.join(f'{k}: {v}' for k, v in sorted(_PLUGIN_ERRORS.items()))
        raise ValueError(f'unknown {kind} {name!r}; registered {sorted(table)}' + (f'; plug-in import errors: {errs}' if errs else ''))
    return table[name]


def arm_class(name):
    return _lookup(_ARMS, name, 'arm flag')


def _check_value(name, vtext, policy, parser, kind):
    if vtext is None:
        if policy == 'required':
            raise ValueError(f'{kind} {name!r} needs a value ({name}@value)')
        return None
    if policy == 'none':
        raise ValueError(f'{kind} {name!r} takes no value')
    v, canon = parser(vtext)
    if canon != vtext:
        raise ValueError(f'{kind} value {vtext!r} is not canonical; write {name}@{canon}')
    return v


def flag_value(flag):
    """Parsed value of a canonical flag (None without a value)."""
    name, vtext = split_flag(flag)
    if vtext is None:
        return None
    if name == 'genemean':
        return canonical_float(vtext)[0]
    if is_train_flag(name):
        return _parse_train_flag(flag)[1]
    return arm_class(name).parse_value(vtext)[0]


def parse_arm(text):
    """Arm string -> sorted tuple of canonical flags; 'base' -> ().  Validates everything."""
    if text is None or text == '' or text == 'base':
        return ()
    if isinstance(text, (list, tuple)):
        text = ','.join(text)
    flags = [f.strip() for f in str(text).split(',') if f.strip()]
    if 'base' in flags:
        raise ValueError('base cannot be combined with other flags')
    names = []
    plug = []
    for f in flags:
        name, vtext = split_flag(f)
        names.append(name)
        if is_train_flag(name):
            _parse_train_flag(f)
            continue
        if name in ROUND1_FLAGS:
            if vtext is None:
                continue
            if name != 'genemean':
                raise ValueError(f'round-1 flag {name!r} takes no value')
            v, canon = canonical_float(vtext, lo=0.0, lo_open=True)
            if canon != vtext:
                raise ValueError(f'value {vtext!r} is not canonical; write genemean@{canon}')
            if v == 1.0:
                raise ValueError('genemean@1 is genemean; use the round-1 name')
        else:
            cls = arm_class(name)
            _check_value(name, vtext, cls.value_policy, cls.parse_value, 'arm flag')
            plug.append(cls)
    if len(set(names)) != len(names):
        raise ValueError('duplicate arm flag name')
    if 'genemean' in names and 'donorpc' in names:
        raise ValueError('genemean and donorpc are mutually exclusive')
    nameset = set(names)
    for n in nameset:
        if is_train_flag(n):
            miss = {TRAIN_PREFIX + r for r in TRAIN_OPTIONS[n[len(TRAIN_PREFIX):]][1]} - nameset
            if miss:
                raise ValueError(f'training option {n!r} requires {sorted(miss)}')
    if sum(c.overrides('critic_module') for c in plug) > 1:
        raise ValueError('at most one arm may provide critic_module (hook d2)')
    for cls in plug:
        miss = set(cls.requires) - nameset
        if miss:
            raise ValueError(f'arm {cls.name!r} requires {sorted(miss)}')
        bad = set(cls.conflicts) & nameset
        if bad:
            raise ValueError(f'arm {cls.name!r} conflicts with {sorted(bad)}')
    if 'offset' in nameset and any(c.overrides('generator_class') for c in plug):
        raise ValueError('round-1 offset cannot be combined with a plug-in that changes the generator class')
    return tuple(sorted(flags))


def arm_name(flags):
    return ','.join(sorted(flags)) if flags else 'base'


def round1_names(flags):
    return {flag_name(f) for f in flags if flag_name(f) in ROUND1_FLAGS}


def make_plugin_arms(flags):
    """Plug-in Arm instances for the plug-in flags, in sorted flag order."""
    out = []
    for f in sorted(flags):
        name, vtext = split_flag(f)
        if name in ROUND1_FLAGS or is_train_flag(name):
            continue
        cls = arm_class(name)
        out.append(cls(_check_value(name, vtext, cls.value_policy, cls.parse_value, 'arm flag'), f))
    return out


def parse_mode(text):
    """Mode string -> (ModeSpec, value, canonical text)."""
    name, vtext = split_flag(text)
    spec = _lookup(_MODES, name, 'generation mode')
    v = _check_value(name, vtext, spec.value_policy, spec.parse_value, 'mode')
    return spec, v, text


def mode_rng(fold, dev_index, mode_name):
    return np.random.default_rng(np.random.SeedSequence([GEN_ROOT, MODE_STREAM_TAG, int(fold), int(dev_index), crc(mode_name)]))


# =========================================================================== built-in modes
def snapshot_states(states):
    """Snapshot state names sorted by update."""
    snaps = [s for s in states if s.startswith('snap_')]
    return sorted(snaps, key=lambda s: int(s.split('_', 1)[1]))


def decay_text(d):
    """Canonical decay text used in state names and mode values ('%g')."""
    return '%g' % float(d)


def raw_final_state(ctx):
    """State name of the raw (live) final weights: 'raw' for t.ema fits, else 'final' (identical content)."""
    topt = getattr(ctx, 'topt', None)
    return 'raw' if topt is not None and topt.ema is not None else 'final'


def _split_counts(M, n):
    return [M // n + (1 if i < M % n else 0) for i in range(n)]


def _even_plan(target, order):
    counts = _split_counts(target.M, len(order))
    names = [s for s, c in zip(order, counts) for _ in range(c)]
    return np.repeat(target.default_cond[None], target.M, 0), names


def _snapens_plan(ctx, value, target, rng, states):
    """snapens: snapshots at u >= 0.75 U (U = run length) + raw final; snapens@K: last K snapshots + raw final."""
    snaps = snapshot_states(states)
    final = raw_final_state(ctx)
    if final not in states:
        raise ValueError(f'snapens needs the {final!r} state; fit has {list(states)}')
    if value is None:
        if target.updates is None:
            raise ValueError('snapens needs the fit run length (GenTarget.updates)')
        U = int(target.updates)
        use = [s for s in snaps if 4 * int(s.split('_', 1)[1]) >= 3 * U]
    else:
        if len(snaps) < value:
            raise ValueError(f'snapens@{value} needs {value} snapshots; fit has {snaps}')
        use = snaps[len(snaps) - value:]
    if not use:
        raise ValueError(f'snapens: no snapshot in the last 25% of training; fit has {list(states)}')
    return _even_plan(target, use + [final])


def _snap_plan(ctx, value, target, rng, states):
    name = f'snap_{value}'
    if name not in states:
        raise ValueError(f'snapshot {name} not in fit (have {list(states)})')
    return np.repeat(target.default_cond[None], target.M, 0), [name] * target.M


def _raw_plan(ctx, value, target, rng, states):
    name = raw_final_state(ctx)
    if name not in states:
        raise ValueError(f'raw state not in fit (have {list(states)})')
    return np.repeat(target.default_cond[None], target.M, 0), [name] * target.M


def _ema_plan(ctx, value, target, rng, states):
    name = f'ema_{decay_text(value)}'
    if name not in states:
        raise ValueError(f'shadow EMA state {name} not in fit (round-3 fits keep {SHADOW_EMA_DECAYS}; have {list(states)})')
    return np.repeat(target.default_cond[None], target.M, 0), [name] * target.M


def _parse_snapema(text):
    u, sep, d = str(text).partition(':')
    if not sep:
        raise ValueError(f'snapema value {text!r}: want U:D, e.g. 16000:0.999')
    uv, ut = canonical_int(u, lo=1)
    dv, dt = canonical_float(d, lo=0.0, lo_open=True, hi=1.0, hi_open=True)
    return (uv, dv), f'{ut}:{dt}'


def _snapema_plan(ctx, value, target, rng, states):
    u, d = value
    name = f'snapema_{u}_{decay_text(d)}'
    if name not in states:
        raise ValueError(f'state {name} not in fit (have {list(states)})')
    return np.repeat(target.default_cond[None], target.M, 0), [name] * target.M


SHADOW_EMA_DECAYS = (0.999, 0.9995, 0.9998)   # round-3 read-out EMA copies (trainer.ShadowEMA)

register_mode('snapens', _snapens_plan, value_policy='optional', parse_value=lambda t: canonical_int(t, lo=1),
              doc='M split evenly (M//n + (i < M%n), ascending) over snapshots at u >= 0.75 U and the raw final state; '
                  'snapens@K: the last K snapshots and the raw final state')
register_mode('snap', _snap_plan, value_policy='required', parse_value=lambda t: canonical_int(t, lo=1),
              doc='all M generated donors from snapshot snap_<U>')
register_mode('raw', _raw_plan, doc='all M from the raw (live) final weights; differs from default only for t.ema fits')
register_mode('ema', _ema_plan, value_policy='required',
              parse_value=lambda t: canonical_float(t, lo=0.0, lo_open=True, hi=1.0, hi_open=True),
              doc='all M from the shadow EMA (read-out, decay D) of the final update')
register_mode('snapema', _snapema_plan, value_policy='required', parse_value=_parse_snapema,
              doc='all M from the shadow EMA at decay D stored in snapshot U (value U:D)')


# =========================================================================== conditions
def stage_of(raw_c):
    v = np.asarray(raw_c, dtype=np.float64)[..., 3] * 3
    s = np.rint(v).astype(int)
    if np.any((s < 0) | (s > 3)) or not np.allclose(v, s, atol=1e-6, rtol=0):
        raise ValueError('ADNC must be one of four ordinal codes')
    return s


def build_conditions(raw_c, flags, age_mean=None, age_sd=None):
    """Seven-slot condition per donor as the ROUND-1 flags condition (copied)."""
    x = np.asarray(raw_c, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError('raw_c must be finite [n,7]')
    stage_of(x)
    out = np.zeros_like(x, dtype=np.float32)
    out[:, 3] = x[:, 3].astype(np.float32)  # identical to mask_adnc (float32 cast)
    if 'covars' in flags:
        if age_sd is None or not age_sd > 0:
            raise ValueError('covars arm needs a positive train age SD')
        out[:, 4] = x[:, 4]
        out[:, 5] = x[:, 5]
        out[:, 6] = (x[:, 6] - age_mean) / age_sd
    return out


def _checked_cond(arm, out, n, hook):
    out = np.asarray(out)
    if out.shape != (n, 7) or out.dtype != np.float32 or not np.isfinite(out).all():
        raise ValueError(f'{arm.flag}.{hook} must return finite float32 [{n},7], got {out.dtype} {out.shape}')
    return out


def plugin_conditions(plugins, ctx, raw_c, cond):
    """Hook (a) chain."""
    raw_c = np.asarray(raw_c, dtype=np.float64)
    for a in plugins:
        cond = _checked_cond(a, a.conditions(ctx, raw_c, cond.copy()), len(raw_c), 'conditions')
    return cond


def plugin_generation_conditions(plugins, ctx, raw_c, cond, is_heldout):
    raw_c = np.asarray(raw_c, dtype=np.float64)
    for a in plugins:
        cond = _checked_cond(a, a.generation_condition(ctx, raw_c, cond.copy(), np.asarray(is_heldout, bool)), len(raw_c), 'generation_condition')
    return cond


# =========================================================================== generator
def _count_generator_cls():
    from pg_model import CountGenerator  # imported lazily (sys.path set by trainer)
    return CountGenerator


_HOOKED = []


def hooked_count_generator():
    """HookedCountGenerator: CountGenerator with parameters_nb split into steps (hook c)."""
    if _HOOKED:
        return _HOOKED[0]
    CountGenerator = _count_generator_cls()

    class HookedCountGenerator(CountGenerator):
        """pg_model.CountGenerator; parameters_nb = film -> cell_hidden -> logits -> softmax/clamp -> dispersion.

        Same constructor (no new parameters, same RNG draws) and the same
        operations in the same order as CountGenerator.parameters_nb, so the
        forward is bitwise identical when no method is overridden.
        """

        def film(self, z, c):
            return self.donor(torch.cat([z, c], 1)).chunk(2, 1)

        def cell_hidden(self, noise, scale, shift, z, c):
            return self.cell(noise) * (1 + scale[:, None]) + shift[:, None]

        def logits(self, h, z, c):
            return self.out(h)

        def dispersion(self, z, c):
            return nn.functional.softplus(self.log_theta) + .05

        def step0_grad_params(self):
            """Released parameters asserted to get a gradient at update 0 (round 3; override when replacing the head)."""
            return [self.out[-1].weight, self.log_theta]

        def parameters_nb(self, z, c, k, library, gen=None):
            scale, shift = self.film(z, c)
            noise = torch.randn(len(z), k, self.zc, device=z.device, generator=gen)
            h = self.cell_hidden(noise, scale, shift, z, c)
            proportions = torch.softmax(self.logits(h, z, c), dim=-1).clamp_min(1e-8)
            proportions = proportions / proportions.sum(-1, keepdim=True)
            mu = library.clamp_min(1)[..., None] * proportions
            theta = self.dispersion(z, c)
            return mu, theta

    _HOOKED.append(HookedCountGenerator)
    return HookedCountGenerator


def make_offset_generator_class():
    CountGenerator = _count_generator_cls()

    class OffsetCountGenerator(CountGenerator):
        """CountGenerator + learnable stage x gene logit offset (round-1 copy)."""

        def __init__(self, mean, theta, offset, zd=4, zc=8, h=128):
            super().__init__(mean, theta, zd=zd, zc=zc, h=h)
            offset = torch.as_tensor(offset, dtype=torch.float32)
            if offset.shape != (4, len(mean)):
                raise ValueError('offset table must be [4,G]')
            self.stage_offset = nn.Parameter(offset.clone())

        def parameters_nb(self, z, c, k, library, gen=None):
            scale, shift = self.donor(torch.cat([z, c], 1)).chunk(2, 1)
            noise = torch.randn(len(z), k, self.zc, device=z.device, generator=gen)
            h = self.cell(noise) * (1 + scale[:, None]) + shift[:, None]
            stage = torch.round(c[:, 3] * 3).long()
            onehot = nn.functional.one_hot(stage, 4).to(self.stage_offset.dtype)
            logits = self.out(h) + (onehot @ self.stage_offset)[:, None, :]
            proportions = torch.softmax(logits, dim=-1).clamp_min(1e-8)
            proportions = proportions / proportions.sum(-1, keepdim=True)
            mu = library.clamp_min(1)[..., None] * proportions
            theta = nn.functional.softplus(self.log_theta) + .05
            return mu, theta

    return OffsetCountGenerator


def offset_init(X, by, stages, pseudocount=OFFSET_PSEUDOCOUNT):
    """Shrunken per-stage pseudobulk log-ratio table [4,G] (train donors only; round-1 copy)."""
    Xt = torch.as_tensor(X)
    C = torch.stack([Xt[torch.as_tensor(r, device=Xt.device)].sum(0) for r in by]).double().cpu().numpy()
    stages = np.asarray(stages, dtype=int)
    n, G = C.shape
    a = float(pseudocount)
    N = C.sum(1)
    logq = np.log(C.sum(0) + a) - np.log(N.sum() + a * G)
    r_d = np.log(C + a) - np.log(N[:, None] + a * G) - logq
    m = r_d.mean(0)
    n_s = np.array([(stages == s).sum() for s in range(4)])
    present = [s for s in range(4) if n_s[s] > 0]
    r_s = np.zeros((4, G))
    ss = np.zeros(G)
    for s in present:
        msk = stages == s
        r_s[s] = r_d[msk].mean(0) - m
        ss += ((r_d[msk] - r_d[msk].mean(0)) ** 2).sum(0)
    df = n - len(present)
    if df < 1:
        raise ValueError('offset init needs more train donors than stages')
    v = ss / df
    e = np.array([1.0 / n_s[s] - 1.0 / n if n_s[s] else 0.0 for s in range(4)])
    excess = [r_s[s] ** 2 - v * e[s] for s in present]
    tau2 = max(1e-6, float(np.mean(excess)))
    w = np.zeros((4, G))
    for s in present:
        w[s] = tau2 / (tau2 + v * e[s])
    table = (w * r_s).astype(np.float32)
    info = {
        'pseudocount': a, 'tau2': tau2, 'donors_per_stage': n_s.tolist(), 'within_stage_df': int(df),
        'mean_within_stage_variance': float(v.mean()),
        'mean_weight_per_stage': [float(w[s].mean()) if n_s[s] else 0.0 for s in range(4)],
        'raw_rms_per_stage': [float(np.sqrt((r_s[s] ** 2).mean())) for s in range(4)],
        'shrunk_rms_per_stage': [float(np.sqrt((table[s].astype(np.float64) ** 2).mean())) for s in range(4)],
        'formula': 'offset_sg = tau2/(tau2 + v_g (1/n_s - 1/n)) * r_sg, r_sg = donor-averaged stage log-ratio minus '
                   'all-donor average (fixer revision v2); see arms.py docstring',
        'version': 'offset_init-v2-donor-averaged',
    }
    return table, info


# =========================================================================== critic
class WideVectorCritic(nn.Module):
    """critic.VectorCritic with an arbitrary input width (same layers, h=128)."""

    def __init__(self, n_features, h=128):
        super().__init__()
        self.n_features = int(n_features)
        self.phi = nn.Sequential(nn.Linear(n_features + 7, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.rho = nn.Sequential(nn.Linear(h + 7, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, c):
        if x.ndim != 2 or x.shape[1] != self.n_features or c.shape != (len(x), 7):
            raise ValueError(f'Expected donor vectors [B,{self.n_features}] and conditions [B,7]')
        f = self.phi(torch.cat([x, c], 1))
        return self.rho(torch.cat([f, c], 1)).squeeze(1)


@dataclass(frozen=True)
class BlockSpec:
    """One column range of the critic input x (hook d2 layout)."""
    flag: object      # None for the released block, else the round-1 or plug-in flag
    name: str
    lo: int
    hi: int
    cell_dim: object = None   # int for a per-cell block (width = cells * cell_dim, cell-major)

    @property
    def width(self):
        return self.hi - self.lo


@dataclass(frozen=True)
class CriticLayout:
    """Critic input layout passed to ``Arm.critic_module`` (hook d2)."""
    n_features: int
    cond_dim: int
    hidden: int
    batch: int
    cells: int
    blocks: tuple

    def block(self, name, flag=None):
        hits = [b for b in self.blocks if b.name == name and (flag is None or b.flag == flag)]
        if len(hits) != 1:
            raise KeyError(f'{len(hits)} critic blocks named {name!r} (flag {flag!r})')
        return hits[0]

    def as_dict(self):
        return {'n_features': self.n_features, 'cond_dim': self.cond_dim, 'hidden': self.hidden, 'batch': self.batch, 'cells': self.cells,
                'blocks': [{'flag': b.flag, 'name': b.name, 'columns': [b.lo, b.hi], 'cell_dim': b.cell_dim} for b in self.blocks]}


# =========================================================================== features
class Features:
    """Critic observation: released 150 + round-1 block (genemean/donorpc) + plug-in blocks.

    ``real(idx, ds=None)``: [B, cells] long tensor of train cell rows -> [B, F];
    ``fake(cf, rate=None, counts=None)``: straight-through critic_features
    [B, cells, G] (+ rate with gradient and Poisson counts) -> [B, F].  With
    round-1 flags only, both are the round-1 code path.  ``blocks`` lists
    (arm, block, first column, last column) of plug-in blocks.
    """

    def __init__(self, flags, ctx, plugins=()):
        from released_update import bag_features, CONFIG
        if CONFIG.get('fourier_feature_scale', 1.0) != 1.0:
            raise ValueError('released bag feature scale must be 1')
        self.bag_features = bag_features
        self.flags = flags
        self.ctx = ctx
        names = round1_names(flags)
        self.kind = 'genemean' if 'genemean' in names else 'donorpc' if 'donorpc' in names else 'base'
        # A critic-only projector must never replace ctx.W: generation and
        # evaluation continue to use the original, train-fitted PCA space.
        self.W = getattr(ctx, 'critic_W', ctx.W)
        self.trainpcs = getattr(ctx, 'critic_trainpcs', ctx.trainpcs)
        self.base_scale = getattr(ctx, 'critic_feature_scale', 1.)
        self.info = {'kind': self.kind}
        if self.kind == 'base':
            self.n_extra = 0
        elif self.kind == 'genemean':
            gm = [f for f in flags if flag_name(f) == 'genemean'][0]
            mult = flag_value(gm)
            self.n_extra = ctx.n_genes
            self.scale = self._genemean_scale()
            self.info.update(genemean_scale=self.scale, reference_bags=GENEMEAN_REF_BAGS,
                             reference_seed=GENEMEAN_REF_SEED)
            if mult is not None:  # genemean@s
                self.scale = self.scale * mult
                self.info.update(genemean_multiplier=mult, genemean_scale_effective=self.scale)
        else:
            self._fit_donorpc()
        self.n_round1 = 150 + self.n_extra
        self.blocks = []
        col = self.n_round1
        for a in plugins:
            for b in a.feature_blocks(ctx):
                if not isinstance(b, FeatureBlock) or not isinstance(b.name, str) or not (isinstance(b.width, int) and b.width > 0):
                    raise ValueError(f'{a.flag}: feature blocks must be FeatureBlock with name and positive int width')
                if any(x[0] is a and x[1].name == b.name for x in self.blocks):
                    raise ValueError(f'{a.flag}: duplicate block name {b.name}')
                cd = getattr(b, 'cell_dim', None)
                if cd is not None and not (isinstance(cd, int) and cd > 0 and b.width == cd * int(getattr(ctx, 'bag', 128))):
                    raise ValueError(f'{a.flag}/{b.name}: per-cell block needs int cell_dim > 0 and width = ctx.bag * cell_dim')
                self.blocks.append((a, b, col, col + b.width))
                col += b.width
        self.n_extra = col - 150
        self.n_features = col
        if self.blocks:
            self.info['r2_blocks'] = [{'flag': a.flag, 'block': b.name, 'columns': [lo, hi], 'info': b.info}
                                      for a, b, lo, hi in self.blocks]

    def layout(self, hidden=128):
        """CriticLayout of the critic input x (hook d2)."""
        blocks = [BlockSpec(None, 'released', 0, 150)]
        if self.kind != 'base':
            f = [x for x in self.flags if flag_name(x) == self.kind][0]
            blocks.append(BlockSpec(f, self.kind, 150, self.n_round1))
        blocks += [BlockSpec(a.flag, b.name, lo, hi, getattr(b, 'cell_dim', None)) for a, b, lo, hi in self.blocks]
        return CriticLayout(self.n_features, 7, int(hidden), int(getattr(self.ctx, 'batch_size', 32)), int(getattr(self.ctx, 'bag', 128)),
                            tuple(blocks))

    def _extra(self, gene_mean):
        if self.kind == 'genemean':
            return gene_mean * self.scale
        return ((gene_mean - self.pc_mu) @ self.pc_V) / self.pc_sd

    def _check(self, a, b, t):
        if t.ndim != 2 or t.shape[1] != b.width:
            raise ValueError(f'{a.flag}/{b.name} returned {tuple(t.shape)}, want [B,{b.width}]')
        return t

    def real(self, idx, ds=None):
        base = self.bag_features(self.trainpcs[idx])
        if self.base_scale != 1.:
            base = base * self.base_scale
        if self.kind == 'base' and not self.blocks:
            return base
        parts = [base]
        if self.kind != 'base':
            parts.append(self._extra(self.ctx.Z[idx].mean(1)))
        parts += [self._check(a, b, b.real(self.ctx, idx, ds)) for a, b, _, _ in self.blocks]
        return torch.cat(parts, dim=1)

    def fake(self, cf, rate=None, counts=None):
        base = self.bag_features(cf @ self.W)
        if self.base_scale != 1.:
            base = base * self.base_scale
        if self.kind == 'base' and not self.blocks:
            return base
        parts = [base]
        if self.kind != 'base':
            parts.append(self._extra(cf.mean(1)))
        parts += [self._check(a, b, b.fake(self.ctx, cf, rate, counts)) for a, b, _, _ in self.blocks]
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def _genemean_scale(self):
        ctx = self.ctx
        rng = np.random.default_rng(GENEMEAN_REF_SEED)
        num = den = 0.0
        for _ in range(GENEMEAN_REF_BAGS // 32):
            ds = rng.integers(len(ctx.by), size=32)
            idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], 128, replace=True) for i in ds]), device=ctx.device)
            num += float(self.bag_features(ctx.trainpcs[idx]).double().square().sum())
            den += float(ctx.Z[idx].mean(1).double().square().sum())
        return float(math.sqrt(num / den))

    @torch.no_grad()
    def _fit_donorpc(self):
        ctx = self.ctx
        P = torch.stack([ctx.Z[torch.as_tensor(r, device=ctx.device)].double().mean(0) for r in ctx.by]).cpu().numpy()
        n = len(P)
        K = min(DONORPC_MAX_K, n - 1)
        mu = P.mean(0)
        U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
        V = Vt[:K].T
        signs = np.sign(V[np.abs(V).argmax(0), np.arange(K)])
        V = V * signs
        scores = (P - mu) @ V
        sd = scores.std(0)
        if not (sd > 0).all():
            raise ValueError('degenerate donor PC')
        dev = ctx.device
        self.pc_mu = torch.as_tensor(mu, dtype=torch.float32, device=dev)
        self.pc_V = torch.as_tensor(V, dtype=torch.float32, device=dev)
        self.pc_sd = torch.as_tensor(sd, dtype=torch.float32, device=dev)
        self.n_extra = K
        self.info.update(K=int(K), explained_variance_fraction=(S[:K] ** 2 / (S ** 2).sum()).tolist(),
                         train_donor_score_sd=sd.tolist())

    def state(self):
        """Round-1 block tensors for the exact consistency check (plug-in blocks: Context.arm_state)."""
        if self.kind == 'genemean':
            return {'genemean_scale': self.scale}
        if self.kind == 'donorpc':
            return {'pc_mu': self.pc_mu.cpu(), 'pc_V': self.pc_V.cpu(), 'pc_sd': self.pc_sd.cpu()}
        return {}
