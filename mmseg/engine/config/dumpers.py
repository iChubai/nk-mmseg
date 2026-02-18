import json

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def yaml_dumper(cfg, filepath):
    if yaml is None:
        raise RuntimeError('pyyaml is required to dump yaml config files')
    cfg = cfg.to_dict()
    with open(filepath, 'w') as f:
        yaml.dump(cfg, f)


def json_dumper(cfg, filepath):
    cfg = cfg.to_dict()
    with open(filepath, 'w') as f:
        json.dump(cfg, f, indent=4)


cfg_dumpers = {
    '.yml': yaml_dumper,
    '.yaml': yaml_dumper,
    '.json': json_dumper,
}
