"""
build_model.py  —  Step 5 (probabilistic-model 6.3.0 / nx backend)

We do NOT learn a JPT here (the external `jpt` package doesn't build on
aarch64). Instead we *construct* a probabilistic circuit by hand that
represents the distribution of "known kitchen objects".

The circuit is a mixture (SumUnit) over three components, one per object
type. Each component is a ProductUnit of independent Gaussians over
[weight, size, material]:

        SumUnit  (the mixture: "a kitchen object is one of these")
        ├── 1/3 · ProductUnit(cup)      N(weight) · N(size) · N(material)
        ├── 1/3 · ProductUnit(pitcher)  N(weight) · N(size) · N(material)
        └── 1/3 · ProductUnit(pot)      N(weight) · N(size) · N(material)

material is kept numeric for now (ceramic=0, glass=1, metal=2) so we
don't get blocked on symbolic-variable encoding. Each component uses a
tight Gaussian around that code, so "glass" objects cluster at ~1.

This whole circuit IS the "trained" model: it supports log_likelihood,
marginal, conditional, and truncated — the exact API you need for
Steps 8, 9, and 10.

Run:
    python build_model.py
"""


import numpy as np

from random_events.variable import Continuous
from probabilistic_model.distributions import GaussianDistribution
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    ProductUnit,
    leaf,
)

# ---------------------------------------------------------------------------
# 1. Declare the variables  (ORDER MATTERS — this is the model's contract)
# ---------------------------------------------------------------------------
weight   = Continuous("weight")     # kg
size     = Continuous("size")       # m (longest dimension)
material = Continuous("material")   # ceramic=0, glass=1, metal=2  (numeric for now)

VARIABLE_ORDER = [weight, size, material]

MATERIAL_CODE = {"ceramic": 0.0, "glass": 1.0, "metal": 2.0}


# ---------------------------------------------------------------------------
# 2. Describe each known object type as (mean, std) per variable
#    These play the role of "what the robot has seen during training".
# ---------------------------------------------------------------------------
#                   weight (kg)     size (m)       material code
COMPONENTS = {
    "cup":     [(0.25, 0.08),   (0.10, 0.02),   (0.5, 0.5)],   # light, small, ceramic/glass
    "pitcher": [(2.50, 0.30),   (0.25, 0.03),   (1.0, 0.15)],  # heavy, medium, glass
    "pot":     [(3.00, 0.40),   (0.30, 0.03),   (2.0, 0.15)],  # heavy, medium, metal
}


def build_model() -> ProbabilisticCircuit:
    """Construct the mixture circuit by hand and return it."""
    circuit = ProbabilisticCircuit()
    mixture = SumUnit(probabilistic_circuit=circuit)

    log_weight = np.log(1.0 / len(COMPONENTS))   # equal prior per component

    for name, params in COMPONENTS.items():
        product = ProductUnit(probabilistic_circuit=circuit)
        for var, (mean, std) in zip(VARIABLE_ORDER, params):
            gaussian = GaussianDistribution(var, mean, std)
            product.add_subcircuit(leaf(gaussian, circuit))
        mixture.add_subcircuit(product, log_weight)

    return circuit


def encode(weight_kg: float, size_m: float, material_name: str) -> np.ndarray:
    """Turn human-readable object properties into the model's input row."""
    code = MATERIAL_CODE[material_name]
    return np.array([[weight_kg, size_m, code]])


if __name__ == "__main__":
    model = build_model()

    # --- Sanity check: normal objects score high, the impossible cup scores low
    tests = [
        ("normal cup",     encode(0.20, 0.10, "ceramic")),
        ("normal pitcher", encode(2.50, 0.25, "glass")),
        ("normal pot",     encode(3.00, 0.30, "metal")),
        ("IMPOSSIBLE cup", encode(50.0, 0.10, "glass")),   # 50 kg, 10 cm
    ]

    print("Log-likelihoods (higher = more familiar):")
    for label, x in tests:
        log_p = float(model.log_likelihood(x)[0])
        print(f"  {label:16s} log P(x) = {log_p:9.2f}")


