

from dataclasses import dataclass, field
from typing_extensions import List

from krrood.entity_query_language.entity import entity, let, Symbol, inference
from krrood.entity_query_language.quantify_entity import an
from krrood.entity_query_language.rule import refinement, alternative
from krrood.entity_query_language.conclusion import Add
from krrood.entity_query_language.predicate import HasType


# ---------------------------------------------------------------------------
# Code cell 1: Domain model and small world
# ---------------------------------------------------------------------------

@dataclass
class Body(Symbol):
    name: str
    size: int = 1


@dataclass
class Container(Body):
    ...


@dataclass
class Handle(Body):
    ...


@dataclass
class Connection(Symbol):
    parent: Body
    child: Body


@dataclass
class FixedConnection(Connection):
    ...


@dataclass
class RevoluteConnection(Connection):
    ...


@dataclass
class World(Symbol):
    id_: int
    bodies: List[Body]
    connections: List[Connection] = field(default_factory=list)


@dataclass
class View(Symbol):  # Common super-type for Drawer/Door/Wardrobe
    ...


@dataclass
class Drawer(View):
    handle: Body
    container: Body


@dataclass
class Door(View):
    handle: Body
    body: Body


@dataclass
class Wardrobe(View):
    handle: Body
    body: Body
    container: Body


def main() -> None:
    # --- Build a small "world"
    container1 = Container("Container1")
    body2 = Body("Body2", size=2)
    body3 = Body("Body3")
    container2 = Container("Container2")
    handle1 = Handle("Handle1")
    handle2 = Handle("Handle2")
    handle3 = Handle("Handle3")

    world = World(
        1,
        [container1, container2, body2, body3, handle1, handle2, handle3],
    )

    # Connections between bodies/handles
    fixed_1 = FixedConnection(container1, handle1)
    fixed_2 = FixedConnection(body2, handle2)
    fixed_3 = FixedConnection(body3, handle3)
    revolute_1 = RevoluteConnection(container2, body3)
    world.connections = [fixed_1, fixed_2, fixed_3, revolute_1]

    # -----------------------------------------------------------------------
    # Code cell 2: Build the starting query
    # -----------------------------------------------------------------------

    # Declare the variables
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)
    views = inference(View)()

    # Aliases for convenience
    handle = fixed_connection.child
    body = fixed_connection.parent
    container = revolute_connection.parent

    # Base query: select things that are handles attached via a fixed connection
    query = an(entity(views, HasType(fixed_connection.child, Handle)))

    # -----------------------------------------------------------------------
    # Code cell 3: Build the rule tree
    # -----------------------------------------------------------------------

    with query:
        # Base conclusion: if a fixed connection exists between body and handle,
        # default to a Drawer.
        Add(views, inference(Drawer)(handle=handle, container=body))

        # Refinement (exception): if the body is "bigger" (size > 1), add a Door
        # instead. This is a more specific case that overrides the base rule.
        with refinement(body.size > 1):
            Add(views, inference(Door)(handle=handle, body=body))

            # Alternative refinement: if the body is *also* connected to a
            # parent container via a revolute connection, add a Wardrobe.
            with alternative(
                body == revolute_connection.child,
                container == revolute_connection.parent,
            ):
                Add(views, inference(Wardrobe)(handle=handle, body=body, container=container))

    # -----------------------------------------------------------------------
    # Code cell 4: Evaluate the rule tree
    # -----------------------------------------------------------------------

    results = list(query.evaluate())

    # Expectations:
    # - Handle1 → Drawer (default rule, body size=1)
    # - Handle2 → Door   (body2.size > 1, no revolute connection)
    # - Handle3 → Wardrobe (body3.size > 1 AND revolute connection to container2)
    assert len(results) == 3, f"expected 3 results, got {len(results)}"
    assert any(isinstance(v, Drawer)   and v.handle.name == "Handle1" for v in results)
    assert any(isinstance(v, Door)     and v.handle.name == "Handle2" for v in results)
    assert any(isinstance(v, Wardrobe) and v.handle.name == "Handle3" for v in results)

    print("Rule tree evaluated successfully. Results:")
    print(*results, sep="\n")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Notes (from the original markdown):
#   - refinement(*conditions): narrows the context with an additional condition
#     (like an exception/specialization).
#   - alternative(*conditions): introduces a sibling branch with its own
#     conditions; only contributes conclusions if those are satisfied.
#   - Add(target, value): materializes a conclusion into the selected variable
#     (here, a collection-like placeholder `views`).
# ---------------------------------------------------------------------------
