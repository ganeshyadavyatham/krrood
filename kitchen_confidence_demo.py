

from dataclasses import dataclass
from typing_extensions import Optional

from build_model import build_model
from confidence.wrapper import ConfidenceAwareEvaluator


# ---------------------------------------------------------------------------
# A plain kitchen object (stand-in for a KRROOD Symbol / a perceived object).
# material can be None to simulate a missing semantic tag.
# ---------------------------------------------------------------------------
@dataclass
class KitchenObject:
    name: str
    weight: float
    size: float
    material: Optional[str]


def main() -> None:
    # --- Step 5 model (built, not loaded — avoids the pickle issue)
    model = build_model()
    evaluator = ConfidenceAwareEvaluator(model=model, threshold=-40.0)

    # --- The world: normal objects + two anomalies
    objects = [
        KitchenObject("cup",            weight=0.20, size=0.10, material="ceramic"),
        KitchenObject("pitcher",        weight=2.50, size=0.25, material="glass"),
        KitchenObject("pot",            weight=3.00, size=0.30, material="metal"),
        KitchenObject("impossible_cup", weight=50.0, size=0.10, material="glass"),
        KitchenObject("tagless_object", weight=0.30, size=0.09, material=None),
    ]

    # --- Run the confidence check on every object
    print(f"{'object':16s} {'log P(x)':>12s}   status")
    print("-" * 48)
    for o in objects:
        log_p = evaluator.check(node_name="grasp_rule_root", instance=o)
        # Was a warning produced for THIS object (the most recent one)?
        flagged = evaluator.warnings and evaluator.warnings[-1].node_name == "grasp_rule_root" \
                  and (log_p == float("-inf") or log_p < evaluator.threshold)
        status = "OUT-OF-DISTRIBUTION" if flagged else "familiar"
        shown = "  -inf" if log_p == float("-inf") else f"{log_p:12.2f}"
        print(f"{o.name:16s} {shown}   {status}")

    # --- Report all warnings (the traceable output)
    print("\nWarnings raised:")
    if not evaluator.warnings:
        print("  (none)")
    for w in evaluator.warnings:
        print(" ", w)


if __name__ == "__main__":
    main()
