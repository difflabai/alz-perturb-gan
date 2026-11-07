"""Pin the actual imported official release, independently of historical Python sites."""
from pathlib import Path
import importlib.metadata
import sys
SITE=Path('/ml2/alz/artifacts/native-releases/particlegan-0.9.7/site-packages')
VERSION='0.9.7'
PENALTY_OPTIONS={'input_domain':'continuous'}
def activate():
    if 'particlegan' not in sys.modules:
        sys.path.insert(0,str(SITE))
    import particlegan
    if not Path(particlegan.__file__).resolve().is_relative_to(SITE/'particlegan'):
        raise RuntimeError('Historical or unqualified ParticleGAN import origin')
    if importlib.metadata.version('particlegan')!=VERSION:
        raise RuntimeError('ParticleGAN distribution version differs')
    return particlegan
