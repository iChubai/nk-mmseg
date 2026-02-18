import ast
import importlib.util
import json
import os.path as osp
import sys
import types

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def yaml_reader(filepath):
    if yaml is None:
        raise RuntimeError('pyyaml is required to read yaml config files')
    with open(filepath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg


def json_reader(filepath):
    with open(filepath, 'r') as f:
        cfg = json.load(f)
    return cfg


def python_reader(filepath):
    """Reader python type config.

    Refer to mmcv.utils.config.
    """
    # validate python syntax
    with open(filepath, 'r', encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError('There are syntax errors in config '
                          f'file {filepath}: {e}')

    abs_path = osp.abspath(filepath)
    dir_name = osp.dirname(abs_path)
    module_name = f'_mmseg_cfg_{abs(hash(abs_path))}'
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load config module from {filepath}')

    sys.path.insert(0, dir_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    cfg = {
        name: value
        for name, value in mod.__dict__.items() if not name.startswith('__')
        and not isinstance(value, types.ModuleType)
        and not isinstance(value, types.FunctionType)
    }
    # delete imported module
    if module_name in sys.modules:
        del sys.modules[module_name]
    return cfg


cfg_readers = {
    '.yml': yaml_reader,
    '.ymal': yaml_reader,
    '.json': json_reader,
    '.py': python_reader
}
