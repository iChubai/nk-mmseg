"""Minimal config objects compatible with common mmengine usage."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


class ConfigDict(dict):

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class Config(ConfigDict):

    @staticmethod
    def _to_config_data(obj):
        if isinstance(obj, dict):
            out = ConfigDict()
            for k, v in obj.items():
                out[k] = Config._to_config_data(v)
            return out
        if isinstance(obj, list):
            return [Config._to_config_data(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(Config._to_config_data(x) for x in obj)
        return obj

    @classmethod
    def fromfile(cls, filename: str | Path) -> 'Config':
        filename = str(filename)
        ext = os.path.splitext(filename)[-1]
        if ext == '.py':
            raw = None

            # Prefer mmseg config loader to support `_base_` inheritance.
            try:
                from mmseg.engine.config import load_cfg
                raw = load_cfg(filename)
            except Exception:
                raw = None

            if raw is None:
                # Fallback path: package import first (supports relative imports
                # when config is under repo root), then raw file import.
                cfg_path = Path(filename).resolve()
                repo_root = Path(__file__).resolve().parents[1]
                mod = None
                try:
                    rel = cfg_path.relative_to(repo_root)
                    if rel.suffix == '.py':
                        module_name = '.'.join(rel.with_suffix('').parts)
                        if str(repo_root) not in sys.path:
                            sys.path.insert(0, str(repo_root))
                        mod = __import__(module_name, fromlist=['*'])
                except Exception:
                    mod = None

                if mod is None:
                    spec = importlib.util.spec_from_file_location(
                        'mmengine_cfg', filename)
                    mod = importlib.util.module_from_spec(spec)
                    assert spec.loader is not None
                    spec.loader.exec_module(mod)

                if hasattr(mod, 'cfg'):
                    raw = getattr(mod, 'cfg')
                elif hasattr(mod, 'C'):
                    raw = getattr(mod, 'C')
                else:
                    raw = {
                        k: v
                        for k, v in mod.__dict__.items()
                        if not k.startswith('_')
                    }
        elif ext in ('.json',):
            with open(filename, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        elif ext in ('.yml', '.yaml'):
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError('pyyaml is required to read yaml config files') from exc
            with open(filename, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
        else:
            raise NotImplementedError(f'Unsupported config format: {ext}')

        cfg = cls()
        if isinstance(raw, dict):
            cfg.update(cls._to_config_data(raw))
        else:
            cfg['cfg'] = cls._to_config_data(raw)
        return cfg

    def copy(self) -> 'Config':
        out = Config()
        out.update(self._to_config_data(self.to_dict()))
        return out

    def merge_from_dict(self, options: dict[str, Any]) -> None:
        for k, v in options.items():
            self[k] = self._to_config_data(v)

    def to_dict(self):
        def _plain(x):
            if isinstance(x, dict):
                return {k: _plain(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_plain(v) for v in x]
            if isinstance(x, tuple):
                return tuple(_plain(v) for v in x)
            return x

        return _plain(dict(self))
