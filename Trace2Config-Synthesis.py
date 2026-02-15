# pip install z3-solver pandas lxml
from z3 import *
import pandas as pd
from lxml import etree
from dataclasses import dataclass

# =============================
# Input artifacts
# =============================
FM_XML_PATH = "/mnt/data/FM.xml"
CAN_LOGS    = "/mnt/data/Simulink_CAN_Logs.csv"
HIL_LAT     = "/mnt/data/Simulink_HIL_Latency.csv"
VAR_TIMING  = "/mnt/data/Simulink_Variant_Timing.csv"
REPLAY_LOGS = "/mnt/data/Simulink_Replay_Attacks.csv"

# =============================
# Empirical bounds container
# =============================
@dataclass
class EmpiricalBounds:
    d1_mean_period_ms: float
    d1_min_interarrival_ms: float
    d1_max_interarrival_ms: float
    d2_max_auth_latency_ms: float
    d2_max_jitter_ms: float
    d3_max_latency_by_variant: dict
    d4_min_replay_interval_ms: float

def derive_empirical_bounds() -> EmpiricalBounds:
    can = pd.read_csv(CAN_LOGS)
    ia = can["inter_arrival_ms"].dropna()
    d1_mean = float(ia.mean())
    d1_min  = float(ia.min())
    d1_max  = float(ia.max())

    hil = pd.read_csv(HIL_LAT)
    d2_max_lat = float(hil["latency_ms"].max())
    d2_max_jit = float(hil["jitter_ms"].max()) if "jitter_ms" in hil.columns else 0.0

    vt = pd.read_csv(VAR_TIMING)
    d3 = (vt.groupby("variant")["latency_ms"].max()).to_dict()
    d3 = {str(k): float(v) for k, v in d3.items()}

    rp = pd.read_csv(REPLAY_LOGS)
    d4_min = float(rp["replay_interval_ms"].dropna().min())

    return EmpiricalBounds(
        d1_mean_period_ms=d1_mean,
        d1_min_interarrival_ms=d1_min,
        d1_max_interarrival_ms=d1_max,
        d2_max_auth_latency_ms=d2_max_lat,
        d2_max_jitter_ms=d2_max_jit,
        d3_max_latency_by_variant=d3,
        d4_min_replay_interval_ms=d4_min
    )

# =============================
# Feature model compiler
# =============================
def compile_feature_model(fm_xml_path: str):
    tree = etree.parse(fm_xml_path)
    root = tree.getroot()

    feature_names = set()
    for f in root.xpath(".//*[@name]"):
        feature_names.add(f.get("name"))

    feature_vars = {name: Bool(name) for name in sorted(feature_names)}
    fm_constraints = []

    # Example: alternative groups (exactly one)
    for alt in root.xpath(".//alt"):
        children = [feature_vars[c.get("name")]
                    for c in alt.xpath("./feature[@name]")]
        if children:
            fm_constraints += [
                Or(children),
                Sum([If(c, 1, 0) for c in children]) == 1
            ]

    # Example: requires / excludes
    for req in root.xpath(".//requires"):
        a, b = req.get("a"), req.get("b")
        if a in feature_vars and b in feature_vars:
            fm_constraints.append(Implies(feature_vars[a], feature_vars[b]))

    for exc in root.xpath(".//excludes"):
        a, b = exc.get("a"), exc.get("b")
        if a in feature_vars and b in feature_vars:
            fm_constraints.append(Not(And(feature_vars[a], feature_vars[b])))

    return feature_vars, fm_constraints

# =============================
# Variant loader
# =============================
def load_variant(variant_name: str):
    if variant_name == "V1":
        return {"CAN": True, "AES_128": True}
    if variant_name == "V2":
        return {"CAN_FD": True, "AES_256": True}
    if variant_name == "V3":
        return {"SecOC_Protection": True}
    return {}

# =============================
# Latency model (trace-fitted placeholder)
# =============================
def build_latency_model(bounds: EmpiricalBounds, feature_vars: dict):
    MAC_32  = feature_vars.get("MAC_32", BoolVal(False))
    MAC_64  = feature_vars.get("MAC_64", BoolVal(False))
    MAC_128 = feature_vars.get("MAC_128", BoolVal(False))

    AES_128 = feature_vars.get("AES_128", BoolVal(False))
    AES_256 = feature_vars.get("AES_256", BoolVal(False))

    Fresh_Counter   = feature_vars.get("Fresh_Counter", BoolVal(False))
    Fresh_Timestamp = feature_vars.get("Fresh_Timestamp", BoolVal(False))

    latency_ms = Real("latency_ms")

    mac_cost = If(MAC_32, 1.2, If(MAC_64, 2.2, 3.2))
    aes_cost = If(AES_128, 0.9, 1.6)
    fr_cost  = If(Fresh_Counter, 0.7, 1.1)

    return latency_ms, [latency_ms == mac_cost + aes_cost + fr_cost]

# =============================
# End-to-end synthesis
# =============================
def synthesize_variant(variant_name: str):
    bounds = derive_empirical_bounds()
    feature_vars, fm_constraints = compile_feature_model(FM_XML_PATH)
    vsel = load_variant(variant_name)

    freshness_window_ms = Int("freshness_window_ms")
    latency_ms, latency_constraints = build_latency_model(bounds, feature_vars)

    req_constraints = []
    if "SecOC_Protection" in feature_vars:
        req_constraints.append(feature_vars["SecOC_Protection"])

    variant_constraints = []
    for fname, val in vsel.items():
        if fname in feature_vars:
            variant_constraints.append(feature_vars[fname] == BoolVal(val))

    empirical_constraints = [
        freshness_window_ms >= int(bounds.d1_min_interarrival_ms),
        latency_ms <= min(
            bounds.d2_max_auth_latency_ms,
            bounds.d3_max_latency_by_variant.get(
                variant_name, bounds.d2_max_auth_latency_ms
            )
        )
    ]

    s = Solver()
    s.add(fm_constraints +
          req_constraints +
          variant_constraints +
          latency_constraints +
          empirical_constraints)

    if s.check() == sat:
        m = s.model()
        return {k: m.evaluate(v, model_completion=True)
                for k, v in feature_vars.items()}
    else:
        return None