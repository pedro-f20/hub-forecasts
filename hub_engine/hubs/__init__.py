"""Hub adapters. Each module exposes SPECS, history(hub, spec, ref, backtest) and latest_truth(hub, spec)."""
from hub_engine.hubs import ai4casting, cdc, respicast

ADAPTERS = {name: module for module in (ai4casting, cdc, respicast) for name in module.SPECS}


def get(name: str):
    """(spec, adapter module) for a hub name, e.g. 'flusight'."""
    if name not in ADAPTERS:
        raise KeyError(f"[hubs] unknown hub {name}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[name].SPECS[name], ADAPTERS[name]
