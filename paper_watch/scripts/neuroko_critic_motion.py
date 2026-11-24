"""Compare the native v11 and v12 critic updates on fixed probes from the NeuroKO training cells."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
import multiprocessing
import os
from pathlib import Path
from statistics import median

from compare_neuroko_v12 import GROUPS, receipt, write_frozen
from neuroko_rate_noise import NOISY_PRIOR, REPLICATION_SEEDS


STEPS = 64
RECIPES = ("unified_v11", "unified_v12")


@contextmanager
def particle_gradients(model):
    """Observe the two native loss derivatives with respect to the learned particle table."""
    import torch

    grads = {}
    trainer = model.trainer
    original = trainer.loss.g_loss

    def capture(name, value):
        if value.requires_grad:
            grad, = torch.autograd.grad(value, model.prior.z, retain_graph=True, allow_unused=True)
            if grad is not None:
                if name in grads:
                    raise ValueError(f"Multiple {name} particle gradients in a generator update")
                grads[name] = grad.detach().flatten()

    def adversarial(*args, **kwargs):
        value = original(*args, **kwargs)
        capture("adversarial", value)
        return value

    def prior_hook(module, inputs, output):
        capture("prior", output)

    def read():
        a = grads.pop("adversarial")
        p = trainer.recipe.prior_reg * grads.pop("prior")
        adv_norm, prior_norm = a.norm().item(), p.norm().item()
        return dict(adversarial=adv_norm, weighted_prior=prior_norm, ratio=prior_norm / adv_norm if adv_norm else None,
                    cosine=(a @ p).item() / (adv_norm * prior_norm) if adv_norm and prior_norm else None)

    handle = trainer.prior_regularizer.register_forward_hook(prior_hook)
    trainer.loss.g_loss = adversarial
    try:
        yield read
    finally:
        handle.remove()
        trainer.loss.g_loss = original


def measure(x, labels, probes, seed, recipe, device, critic_lr, prior_weight, *, monitor=True, measure_gradients=False,
            steps=STEPS, fit_context=None):
    import torch
    from paper_watch.alztest.harness import ConditionalPGAN, TrainingSettings
    from paper_watch.alztest.neuroko_duration import optimizer_rates
    from paper_watch.alztest.neuroko_tune import verify_fixed_schedule

    overrides = {k: v for k, v in NOISY_PRIOR.items() if k not in ("name", "lr")}
    overrides["d_lr_mult"] = critic_lr / NOISY_PRIOR["lr"]
    overrides["prior_reg"] = prior_weight
    settings = TrainingSettings(**overrides)
    torch.manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = ConditionalPGAN(x.shape[1], len(probes), seed=seed, recipe_name=recipe, device=device,
                                learning_rate=NOISY_PRIOR["lr"], settings=settings)
    verify_fixed_schedule(model.trainer.recipe, steps)
    rates = optimizer_rates(model)
    if rates != {"generator": .014, "critic": critic_lr, "particles": .0406}:
        raise ValueError(f"Unexpected initial rates {rates}")

    def weights_hash():
        digest = sha256()
        for module in (model.G, model.D, model.prior):
            for p in module.parameters():
                digest.update(p.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def outputs():
        saved = model.D.cond.clone()
        try:
            values = []
            for i, rows in enumerate(probes):
                model.D.cond.zero_()
                model.D.cond[i] = 1
                values.append(model.D(rows).flatten())
            return torch.stack(values)
        finally:
            model.D.cond.copy_(saved)

    initial = weights_hash()
    batches = sha256()

    def record(label, rows):
        batches.update(label.to_bytes(4, "little"))
        batches.update(rows.astype("<i8", copy=False).tobytes())

    first = None
    preceding = outputs() if monitor else None
    centered_rms, raw_rms, loss_gradients = [], [], []

    def after(step):
        nonlocal first, preceding
        if optimizer_rates(model) != rates:
            raise ValueError(f"Changed learning rate at step {step}")
        if read_gradients is not None:
            loss_gradients.append(read_gradients())
        if monitor:
            current = outputs()
            change = current - preceding
            centered = change - change.mean(dim=1, keepdim=True)
            raw_rms.append(change.square().mean().sqrt().item())
            centered_rms.append(centered.square().mean().sqrt().item())
            if step == 1:
                first = centered.flatten().cpu().double().numpy()
            preceding = current

    observation = particle_gradients(model) if measure_gradients else nullcontext()
    with ExitStack() as stack:
        read_gradients = stack.enter_context(observation)
        if fit_context is not None:
            stack.enter_context(fit_context(model))
        model.fit(x, labels, steps=steps, after_step=after, on_batch=record)
    row = dict(initial_parameters_sha256=initial, final_parameters_sha256=weights_hash(),
               training_batches_sha256=batches.hexdigest(), rates=rates, steps=model.trainer.completed_steps,
               centered_rms=centered_rms, raw_rms=raw_rms)
    if measure_gradients:
        row["particle_loss_gradients"] = loss_gradients
    return row, first


def probe_folds(args, gpu):
    import numpy as np
    import torch
    from paper_watch.alztest.neuroko import LAYOUT, load_comparison, open_screen, project_fold

    torch.set_num_threads(2)
    torch.cuda.set_device(gpu)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda", gpu)
    with ExitStack() as stack:
        screens = [stack.enter_context(open_screen(args.data, name, args.window_cells)) for name in LAYOUT]
        conditions, panels = load_comparison(screens, 100, 160, 6)
    panel = panels["iNeurons"]
    for i, group in enumerate(GROUPS):
        training = (panel.groups != "Screen1_0") & (panel.groups != group)
        x, y = project_fold(panel.x[training], panel.x[panel.groups == group])
        labels = panel.labels[training]
        data_hash = sha256()
        for arr in (x, labels, y, panel.labels[panel.groups == group]):
            data_hash.update(np.ascontiguousarray(arr).view(np.uint8))
        rng = np.random.default_rng(5847)
        indices = np.stack([rng.choice(np.flatnonzero(labels == k), 32, replace=False) for k in range(len(conditions))])
        probes = torch.as_tensor(x[indices], device=device, dtype=torch.float32)
        yield i, group, x, labels, probes, data_hash.hexdigest(), sha256(indices.astype("<i8").tobytes()).hexdigest()


def run_gpu(args, gpu):
    import numpy as np

    if version("particlegan") != "0.12.0":
        raise ValueError("This optimizer comparison requires ParticleGAN 0.12.0")
    cells = []
    for i, group, x, labels, probes, data_hash, probe_hash in probe_folds(args, gpu):
        device = probes.device
        for seed in REPLICATION_SEEDS:
            if (seed + i) % 2 != gpu:
                continue
            rows, first = {}, {}
            for recipe in RECIPES:
                critic_lr = .021 if recipe == "unified_v11" else args.v12_critic_lr
                weight = NOISY_PRIOR["prior_reg"] if recipe == "unified_v11" else args.v12_prior_weight
                row, first[recipe] = measure(x, labels, probes, seed, recipe, device, critic_lr, weight,
                                             measure_gradients=args.particle_gradients)
                path = receipt(args.reference, "validation", "0.12.0", recipe, NOISY_PRIOR, seed, group)
                ref = json.loads(path.read_text())
                if (ref["initial_parameters_sha256"] != row["initial_parameters_sha256"]
                        or ref["projected_data_sha256"] != data_hash
                        or ref["visible_devices"] != str(gpu)
                        or ref["initial_rates"] != {"generator": .014, "critic": .021, "particles": .0406}
                        or row["steps"] != STEPS):
                    raise ValueError(f"Probes use a different training configuration from {path}")
                if seed == REPLICATION_SEEDS[0] and i == 0:
                    plain, _ = measure(x, labels, probes, seed, recipe, device, critic_lr, weight, monitor=False)
                    for field in ("final_parameters_sha256", "training_batches_sha256"):
                        if plain[field] != row[field]:
                            raise ValueError(f"Measuring critic outputs changed {field}")
                    row["unobserved_replay_verified"] = True
                rows[recipe] = row
            v11, v12 = (rows[r] for r in RECIPES)
            if any(v11[k] != v12[k] for k in ("initial_parameters_sha256", "training_batches_sha256")):
                raise ValueError("Different initial parameters or training minibatches between formulations")
            a, b = (first[r] for r in RECIPES)
            cell = dict(seed=seed, group=group, gpu=gpu, projected_data_sha256=data_hash,
                        probe_indices_sha256=probe_hash, rows=rows,
                        first_step_centered_cosine=float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                        first_step_ratio=v12["centered_rms"][0] / v11["centered_rms"][0],
                        median_step_ratio=median(v12["centered_rms"]) / median(v11["centered_rms"]))
            write_frozen(args.output / f"seed{seed}-{group}.json", cell)
            cells.append(cell)
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--v12-critic-lr", type=float, default=.021, help="verify the measured movement at a new fixed v12 critic rate")
    parser.add_argument("--particle-gradients", action="store_true",
                        help="also record adversarial and weighted prior gradients on the learned particles")
    parser.add_argument("--v12-prior-weight", type=float, default=NOISY_PRIOR["prior_reg"],
                        help="verify the effective particle gradient strength at another native prior weight")
    args = parser.parse_args()
    if not isfinite(args.v12_critic_lr) or args.v12_critic_lr <= 0:
        parser.error("--v12-critic-lr must be positive and finite")
    if not isfinite(args.v12_prior_weight) or args.v12_prior_weight < 0 or (args.particle_gradients and args.v12_prior_weight == 0):
        parser.error("use a finite nonnegative prior weight; measuring its gradient requires an active prior penalty")
    args.output.mkdir(parents=True, exist_ok=True)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    protocol = dict(release="0.12.0", recipes=RECIPES, settings=NOISY_PRIOR, seeds=REPLICATION_SEEDS, groups=GROUPS,
                    steps=STEPS, probe_seed=5847, real_training_probes_per_condition=32,
                    rule="Compare the RMS change in critic scores after subtracting each condition's mean change. "
                         "Record the first step at common initial parameters and per-fit medians over 64 steps. "
                         "Compare ratios across nine paired fits. These measurements do not select a winner.")
    if args.v12_critic_lr != .021:
        protocol["v12_critic_lr"] = args.v12_critic_lr
    if args.v12_prior_weight != NOISY_PRIOR["prior_reg"]:
        protocol["v12_prior_weight"] = args.v12_prior_weight
    if args.particle_gradients:
        protocol["particle_gradients"] = "Record the two native loss gradients on all 64 x 16 particle parameters each update."
    write_frozen(args.output / "protocol.json", protocol)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(run_gpu, args, gpu) for gpu in (0, 1)]
        cells = sorted([row for job in jobs for row in job.result()], key=lambda row: (row["seed"], row["group"]))
    summary = {key: median(cell[key] for cell in cells)
               for key in ("first_step_centered_cosine", "first_step_ratio", "median_step_ratio")}
    write_frozen(args.output / "result.json", dict(protocol=protocol, summary=summary, cells=cells))
    print("Across nine paired training probes", summary, flush=True)


if __name__ == "__main__":
    main()
