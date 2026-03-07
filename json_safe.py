"""json の safe wrapper — bool/NaN/Inf/numpy型 を自動変換"""
import json as _json
import math

# numpy が使える場合だけ型を取得
try:
    import numpy as _np
    _NP_INT   = (_np.integer,)
    _NP_FLOAT = (_np.floating,)
    _NP_BOOL  = (_np.bool_,)
except ImportError:
    _NP_INT   = ()
    _NP_FLOAT = ()
    _NP_BOOL  = ()


def _sanitize(obj):
    """再帰的にbool/NaN/Inf/numpy型を変換"""
    if _NP_BOOL and isinstance(obj, _NP_BOOL):
        return bool(obj)
    if isinstance(obj, bool):
        return int(obj)
    if _NP_INT and isinstance(obj, _NP_INT):
        return int(obj)
    if _NP_FLOAT and isinstance(obj, _NP_FLOAT):
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
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
