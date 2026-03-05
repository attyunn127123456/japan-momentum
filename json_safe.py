"""json の safe wrapper — bool/NaN/Inf を自動変換"""
import json as _json
import math


def _sanitize(obj):
    """再帰的にbool/NaN/Infを変換"""
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return obj


def dumps(obj, **kwargs):
    return _json.dumps(_sanitize(obj), **kwargs)


def dump(obj, fp, **kwargs):
    fp.write(dumps(obj, **kwargs))


def loads(s, **kwargs):
    return _json.loads(s, **kwargs)


def load(fp, **kwargs):
    return _json.load(fp, **kwargs)
