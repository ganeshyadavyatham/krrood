from dataclasses import dataclass, field
from typing_extensions import List

from krrood.entity_query_language.entity import entity, let, Symbol, inference
from krrood.entity_query_language.quantify_entity import an
from krrood.entity_query_language.rule import refinement
from krrood.entity_query_language.conclusion import Add


@dataclass
class KitchenObject(Symbol):
    name: str
    weight: float
    size: float
    material: str


@dataclass
class GraspChoice(Symbol):
    obj: KitchenObject
    style: str


@dataclass
class World(Symbol):
    id_: int
    objects: List[KitchenObject]
    choices: List[GraspChoice] = field(default_factory=list)


# Build a small world of normal kitchen objects
cup     = KitchenObject("cup",     weight=0.2, size=0.10, material="ceramic")
pitcher = KitchenObject("pitcher", weight=2.5, size=0.25, material="glass")
pot     = KitchenObject("pot",     weight=3.0, size=0.30, material="metal")
world   = World(1, [cup, pitcher, pot])

obj     = let(type_=KitchenObject, domain=world.objects)
choices = inference(GraspChoice)()

query = an(entity(choices, obj.weight > 0))   # everything has positive weight; selects all

with query:
    # Default: one-handed grasp
    Add(choices, inference(GraspChoice)(obj=obj, style="one_handed"))
    # Refinement: if heavy AND glass, two-handed
    with refinement(obj.weight > 2.0, obj.material == "glass"):
        Add(choices, inference(GraspChoice)(obj=obj, style="two_handed"))

results = list(query.evaluate())
for r in results:
    print(r.obj.name, "->", r.style)
