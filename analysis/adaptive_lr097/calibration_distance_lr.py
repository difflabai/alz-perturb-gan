"""One calibration scalar controls an external LR schedule, not the PG objective.

The callback is compatible with released UpdatePolicy(schedule=...). The caller
must checkpoint this state separately and evaluate with isolated random streams.
No data loading, GPU allocation, fitting, or live-policy mutation occurs here.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import math
from statistics import median

@dataclass(frozen=True)
class Config:
    interval: int = 1000
    warmup: int = 2000
    window: int = 3
    patience: int = 3
    cooldown: int = 1
    factor: float = 0.5
    min_scale: float = 0.125
    relative_improvement: float = 0.01
    def __post_init__(self):
        for k in ('interval', 'window', 'patience'):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError('positive integer cadence/window/patience required')
        for k in ('warmup', 'cooldown'):
            if type(getattr(self, k)) is not int or getattr(self, k) < 0:
                raise ValueError('nonnegative integer warmup/cooldown required')
        for k in ('factor', 'min_scale', 'relative_improvement'):
            v = getattr(self, k)
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError('finite numeric controller configuration required')
        if not 0 < self.factor < 1 or not 0 < self.min_scale <= 1 or not 0 <= self.relative_improvement < 1:
            raise ValueError('invalid LR multiplier or improvement threshold')

class CalibrationDistanceLR:
    metric = 'calibration_cell_energy_distance'
    def __init__(self, config=Config()):
        if not isinstance(config, Config):
            raise TypeError('Config required')
        self.config = config
        self.history = []
        self.scale = 1.0
        self.best = None
        self.bad = 0
        self.cooldown_left = 0
        self.last_step = -1
        self.decisions = []

    def observe(self, completed_updates, value, *, split_role):
        """Observe only a finite calibration ED at a completed-update boundary."""
        if split_role != 'calibration':
            raise ValueError('LR decisions may only use calibration observations')
        if (type(completed_updates) is not int or completed_updates <= self.last_step
                or completed_updates < 0 or completed_updates % self.config.interval):
            raise ValueError('strictly increasing, cadence-aligned completed updates required')
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative calibration distance required')
        # Do not let missed observations silently shorten the plateau duration.
        if self.last_step >= 0 and completed_updates != self.last_step + self.config.interval:
            raise ValueError('missing calibration observation; keep the declared cadence')
        if self.last_step < 0 and completed_updates not in (0, self.config.interval):
            raise ValueError('first observation must be zero or the first declared interval')
        self.last_step = completed_updates
        self.history.append(float(value))
        smooth = median(self.history[-self.config.window:])
        action = 'hold'
        if completed_updates >= self.config.warmup and len(self.history) >= self.config.window:
            if self.best is None or smooth < self.best * (1 - self.config.relative_improvement):
                self.best, self.bad = smooth, 0
                action = 'improved'
            elif self.cooldown_left == 0:
                self.bad += 1
            if self.cooldown_left:
                self.cooldown_left -= 1
            elif self.bad >= self.config.patience:
                reduced = max(self.config.min_scale, self.scale * self.config.factor)
                action = 'reduce' if reduced < self.scale else 'at_floor'
                self.scale, self.bad = reduced, 0
                self.cooldown_left = self.config.cooldown
        result = dict(completed_updates=completed_updates, metric=self.metric,
                      raw_value=float(value), smoothed_value=smooth,
                      action=action, lr_scale=self.scale)
        self.decisions.append(result)
        return deepcopy(result)

    def __call__(self, step, recipe):
        """Released public schedule callback: multiply G/D/prior equally."""
        from particlegan import learning_rate_scales
        if type(step) is not int or step < 0 or step < self.last_step:
            raise ValueError('schedule must follow observed completed updates')
        if recipe.continuous_policy is not None:
            raise ValueError('this external schedule requires the fixed BISPF recipe')
        network, prior = learning_rate_scales(step, recipe)
        return network * self.scale, prior * self.scale

    def state_dict(self):
        return deepcopy(dict(schema=1, metric=self.metric, config=asdict(self.config),
                             decisions=self.decisions))

    def load_state_dict(self, state):
        """Replay and validate before mutation; retain exact future decisions."""
        if (not isinstance(state, dict) or set(state) != {'schema','metric','config','decisions'}
                or type(state['schema']) is not int or state['schema'] != 1
                or state['metric'] != self.metric or state['config'] != asdict(self.config)
                or not isinstance(state['decisions'], list)):
            raise ValueError('controller checkpoint identity mismatch')
        try:
            saved_config = Config(**state['config'])
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid controller configuration in checkpoint') from exc
        if saved_config != self.config:
            raise ValueError('controller configuration differs')
        other = type(self)(self.config)
        for saved in state['decisions']:
            if not isinstance(saved, dict) or 'completed_updates' not in saved or 'raw_value' not in saved:
                raise ValueError('invalid controller history')
            actual = other.observe(saved['completed_updates'], saved['raw_value'], split_role='calibration')
            if saved != actual:
                raise ValueError('controller checkpoint decisions disagree with history')
        self.__dict__.update(deepcopy(other.__dict__))
