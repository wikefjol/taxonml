# schedulers.py
import math, torch

def _req(d, key, msg=None):
    if key not in d:
        raise ValueError(msg or f"Missing '{key}' in schedule config")
    return d[key]

def validate_schedule_cfg(s: dict) -> dict:
    s = {**s}
    s.setdefault("base", "epoch")
    if s["base"] not in ("epoch", "step"):
        raise ValueError("schedule.base must be 'epoch' or 'step'")
    s.setdefault("floor", {}); s["floor"].setdefault("min_factor", 1e-3)
    s.setdefault("warmup", {"type": "linear", "duration": 0})
    s["warmup"].setdefault("type", "linear")
    s["warmup"].setdefault("duration", 0)

    main = _req(s, "main", "schedule.main is required")
    t = _req(main, "type", "schedule.main.type is required")
    if t == "tri":
        _req(main, "plateau"); _req(main, "decay")
    elif t == "cosine":
        _req(main, "epochs")
    elif t == "cosine_restarts":
        _req(main, "cycle_epochs"); main.setdefault("num_cycles", 1)
        main.setdefault("t_mult", 1.0); main.setdefault("peak_decay", 1.0)
    else:
        raise ValueError(f"Unknown schedule.main.type: {t}")
    return s

def build_scheduler_unified(optimizer, steps_per_epoch: int, schedule: dict):
    s = validate_schedule_cfg(schedule)
    base = s["base"]; wu = s["warmup"]["duration"]; minf = s["floor"]["min_factor"]
    main = s["main"]; t = main["type"]

    def step_to_epoch(gs: int) -> float:
        return gs if base == "epoch" else gs / max(1, steps_per_epoch)

    def tri_lambda(gs: int):
        e = step_to_epoch(gs)
        if e <= wu: return e / max(1e-9, wu)
        p, d = float(main["plateau"]), float(main["decay"])
        if e <= wu + p: return 1.0
        if e <= wu + p + d:
            prog = (e - wu - p) / max(1e-9, d)
            return 1.0 - prog * (1.0 - minf)
        return minf

    def cosine_lambda(gs: int):
        e = step_to_epoch(gs)
        if e <= wu: return e / max(1, wu) if wu > 0 else 1.0
        T = float(main["epochs"])
        t_ = min(T, e - wu)
        cos_out = 0.5 * (1.0 + math.cos(math.pi * t_ / max(1e-9, T)))
        return minf + (1.0 - minf) * cos_out

    def cosine_restarts_lambda(gs: int):
        e = step_to_epoch(gs)
        if e <= wu: return e / max(1, wu) if wu > 0 else 1.0
        e -= wu
        L = float(main["cycle_epochs"]); k = int(main["num_cycles"])
        t_mult = float(main["t_mult"]); peak_decay = float(main["peak_decay"])
        cycle_len, idx = L, 0
        while e >= cycle_len and idx < max(0, k-1):
            e -= cycle_len; idx += 1; cycle_len *= t_mult
        cos_out = 0.5 * (1.0 + math.cos(math.pi * min(e, cycle_len) / max(1e-9, cycle_len)))
        peak = peak_decay ** idx
        return minf + peak * (1.0 - minf) * cos_out

    fn = {"tri": tri_lambda, "cosine": cosine_lambda, "cosine_restarts": cosine_restarts_lambda}[t]
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    sched.by_step = (base == "step")   # <- tell the trainer how to step
    sched.base = base                  # optional: helpful for debugging
    sched.steps_per_epoch = steps_per_epoch  # optional: doc/debug
    return sched
