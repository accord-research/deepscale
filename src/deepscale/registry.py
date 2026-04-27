_METHODS = {}
_METRICS = {}
_STRATEGIES = {}

def register_method(name):
    def decorator(cls):
        _METHODS[name] = cls
        return cls
    return decorator

def register_metric(name):
    def decorator(cls):
        _METRICS[name] = cls
        return cls
    return decorator

def register_strategy(name):
    def decorator(cls):
        _STRATEGIES[name] = cls
        return cls
    return decorator

def get_method(name):
    if name not in _METHODS:
        raise KeyError(f"Unknown method: {name}")
    return _METHODS[name]

def get_metric(name):
    if name not in _METRICS:
        raise KeyError(f"Unknown metric: {name}")
    return _METRICS[name]

def get_strategy(name):
    if name not in _STRATEGIES:
        raise KeyError(f"Unknown strategy: {name}")
    return _STRATEGIES[name]
