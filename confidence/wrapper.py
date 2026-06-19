"""
Step 7 — The confidence-aware wrapper.

Scores a kitchen object against the joint probabilistic circuit using a
FULL-EVIDENCE query (log_likelihood over the whole [weight, size, material]
vector). If the log-likelihood falls below a configurable threshold, it
records an UnfamiliarSampleWarning.

This is the "basic" version your plan targets for the supervisor meeting:
- full evidence only (no marginal / conditional yet),
- one check per object,
- a single configurable threshold.

Steps 9 and 10 will refine WHERE this is called (per node) and WHICH slice
of the model it queries (marginal / conditional). The plumbing here stays.
"""

import numpy as np
from dataclasses import dataclass, field
from typing_extensions import List

from .warning import UnfamiliarSampleWarning

# ceramic=0, glass=1, metal=2  — must match build_model.py
MATERIAL_CODE = {"ceramic": 0.0, "glass": 1.0, "metal": 2.0}


@dataclass
class ConfidenceAwareEvaluator:
    model: object                 # a probabilistic circuit with .log_likelihood
    threshold: float = -40.0      # familiarity threshold (see note below)
    warnings: List[UnfamiliarSampleWarning] = field(default_factory=list)

    def check(self, node_name: str, instance) -> float:
        """Score one instance; record a warning if it is out-of-distribution.

        Returns the log-likelihood so callers can log / inspect it.
        """
        # 1. Handle a missing semantic tag (the "missing material" anomaly)
        if getattr(instance, "material", None) not in MATERIAL_CODE:
            self.warnings.append(UnfamiliarSampleWarning(
                node_name=node_name,
                log_p=None,
                reason=f"missing/unknown material {getattr(instance, 'material', None)!r}",
            ))
            return float("-inf")

        # 2. Encode the instance into the model's variable order
        x = np.array([[
            instance.weight,
            instance.size,
            MATERIAL_CODE[instance.material],
        ]])

        # 3. Full-evidence query: L = log P(x)
        log_p = float(self.model.log_likelihood(x)[0])

        # 4. Threshold and (maybe) warn
        if log_p < self.threshold:
            self.warnings.append(UnfamiliarSampleWarning(
                node_name=node_name,
                log_p=log_p,
                reason=f"log P(x)={log_p:.2f} below threshold {self.threshold}",
            ))
        return log_p
