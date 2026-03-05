"""json の safe wrapper — bool/NaN/Inf を自動変換"""
import json as _json
import math

class SafeEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return super().default(obj)
        return super().default(obj)
    
    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(self._sanitize(obj), _one_shot)
    
    def _sanitize(self, obj):
        if isinstance(obj, bool):
            return int(obj)
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(i) for i in obj]
        return obj

def dumps(obj, **kwargs):
    kwargs.setdefault('cls', SafeEncoder)
    kwargs.setdefault('ensure_ascii', False)
    return _json.dumps(obj, **kwargs)

def dump(obj, fp, **kwargs):
    fp.write(dumps(obj, **kwargs))
