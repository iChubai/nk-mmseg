from collections.abc import Mapping


def iter_leaves(obj):
    """A generator to visit all leaves of obj."""
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            for k, v in iter_leaves(value):
                k.insert(0, key)
                yield (k, v)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            for k, v in iter_leaves(value):
                k.insert(0, i)
                yield (k, v)
    else:
        yield [], obj


def set_leaf(obj, keys, value):
    if isinstance(keys, str):
        keys = keys.split('.')
    if not keys:
        return

    parents = []
    cur = obj
    for key in keys[:-1]:
        parents.append((cur, key))
        cur = cur[key]

    leaf_key = keys[-1]

    def _set_item(container, key, new_value):
        if isinstance(container, tuple):
            tmp = list(container)
            tmp[key] = new_value
            return tuple(tmp)
        container[key] = new_value
        return container

    updated = _set_item(cur, leaf_key, value)
    if not isinstance(cur, tuple):
        return

    # Propagate reconstructed tuples back to the root.
    child = updated
    for parent, key in reversed(parents):
        if isinstance(parent, tuple):
            child = _set_item(parent, key, child)
            continue
        parent[key] = child
        return


def delete_node(obj, keys):
    if isinstance(keys, (tuple, list)):
        for key in keys[:-1]:
            obj = obj[key]
        del obj[keys[-1]]
    else:
        assert isinstance(keys, str), 'only support search str node'
        if isinstance(obj, Mapping):
            if keys in obj:
                del obj[keys]
            for value in obj.values():
                delete_node(value, keys)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                delete_node(value, keys)
