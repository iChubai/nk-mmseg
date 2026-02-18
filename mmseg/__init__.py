import warnings

import mmcv
from packaging.version import parse

from .version import __version__, version_info


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a comparable tuple."""
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))

    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(
                    f'unknown prerelease version {version.pre[0]}, '
                    'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])
    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])

    return tuple(release)


def _safe_mmcv_version():
    raw = getattr(mmcv, '__version__', '0.0.0')
    # local compatibility builds may use suffixes like "2.0.0-jittor-lite"
    base = raw.split('-')[0]
    try:
        return digit_version(base)
    except Exception:
        return digit_version('0.0.0')


mmcv_version = _safe_mmcv_version()


__all__ = ['__version__', 'version_info', 'digit_version', 'mmcv_version']
