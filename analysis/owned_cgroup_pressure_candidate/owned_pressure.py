"""Fail-closed memory-pressure telemetry for one inherited cgroup-v2 scope."""
import math
from pathlib import Path


class PressureRefusal(RuntimeError):
    pass


def full_avg10(text):
    try:
        lines = [line for line in text.splitlines() if line.startswith('full ')]
        if len(lines) != 1:
            raise ValueError('full line is missing or duplicated')
        fields = dict(item.split('=', 1) for item in lines[0].split()[1:])
        value = float(fields['avg10'])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('invalid pressure value')
        return value
    except (ValueError, KeyError) as error:
        raise PressureRefusal('Malformed memory pressure telemetry') from error


class OwnedPressure:
    def __init__(self, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
        self.proc = Path(proc_root)
        self.base = Path(cgroup_root).resolve(strict=True)
        self.membership = self._membership('self')
        self.directory = (self.base / self.membership.lstrip('/')).resolve(strict=True)
        if not self.directory.is_relative_to(self.base):
            raise PressureRefusal('Cgroup directory escapes its mounted root')
        if not (self.directory / 'memory.pressure').is_file():
            raise PressureRefusal('Owned cgroup memory pressure is unavailable')

    def _membership(self, pid):
        try:
            text = (self.proc / str(pid) / 'cgroup').read_text()
            rows = [row[3:] for row in text.splitlines() if row.startswith('0::')]
            if len(rows) != 1:
                raise ValueError('single unified cgroup membership required')
            value = rows[0]
            if not value.startswith('/') or '..' in Path(value).parts or '\x00' in value:
                raise ValueError('invalid unified cgroup path')
            return value
        except (OSError, ValueError) as error:
            raise PressureRefusal('Owned cgroup membership is unavailable or invalid') from error

    def check_worker(self, pid):
        if self._membership('self') != self.membership or self._membership(pid) != self.membership:
            raise PressureRefusal('Worker or guard moved outside the measured inherited cgroup')

    def sample(self):
        if self._membership('self') != self.membership:
            raise PressureRefusal('Guard cgroup changed after pressure scope was bound')
        if (self.base / self.membership.lstrip('/')).resolve(strict=True) != self.directory:
            raise PressureRefusal('Measured cgroup path changed after binding')
        try:
            owned = full_avg10((self.directory / 'memory.pressure').read_text())
            global_value = full_avg10((self.proc / 'pressure/memory').read_text())
        except OSError as error:
            raise PressureRefusal('Pressure telemetry disappeared; no zero-pressure fallback') from error
        return {'owned_cgroup_full_avg10': owned, 'global_full_avg10': global_value,
                'cgroup_relative_path': self.membership}

    def full_pressure(self):
        return self.sample()['owned_cgroup_full_avg10']
