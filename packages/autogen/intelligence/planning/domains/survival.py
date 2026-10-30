"""World-linked Survival and Crafting STRIPS domain (SYS-08, Module M09)."""

from __future__ import annotations

from autogen.intelligence.planning.contracts import Fact, GroundAction, PlanningProblem


def create_survival_crafting_problem(
    agent_start: str = "Camp",
    forest_loc: str = "Forest",
    food_loc: str = "Meadow",
    shelter_loc: str = "Camp",
) -> PlanningProblem:
    """Creates a Survival & Crafting STRIPS planning problem instance:

    Gather wood -> Build shelter -> Gather food -> Store food -> Light fire -> Survive winter.
    """
    initial_facts = frozenset(
        {
            Fact("At", (agent_start,)),
            Fact("WoodAvailable", (forest_loc,)),
            Fact("FoodAvailable", (food_loc,)),
        }
    )

    goal_facts = frozenset(
        {
            Fact("WinterSurvived", ()),
        }
    )

    locations = {agent_start, forest_loc, food_loc, shelter_loc}
    actions: list[GroundAction] = []

    # Movement actions between locations
    for loc1 in locations:
        for loc2 in locations:
            if loc1 != loc2:
                actions.append(
                    GroundAction(
                        name=f"Move({loc1}, {loc2})",
                        preconditions=frozenset({Fact("At", (loc1,))}),
                        add_effects=frozenset({Fact("At", (loc2,))}),
                        delete_effects=frozenset({Fact("At", (loc1,))}),
                        cost=1.0,
                    )
                )

    # Gather Wood
    actions.append(
        GroundAction(
            name=f"GatherWood({forest_loc})",
            preconditions=frozenset({Fact("At", (forest_loc,)), Fact("WoodAvailable", (forest_loc,))}),
            add_effects=frozenset({Fact("HasWood", ())}),
            delete_effects=frozenset({Fact("WoodAvailable", (forest_loc,))}),
            cost=1.0,
        )
    )

    # Gather Food
    actions.append(
        GroundAction(
            name=f"GatherFood({food_loc})",
            preconditions=frozenset({Fact("At", (food_loc,)), Fact("FoodAvailable", (food_loc,))}),
            add_effects=frozenset({Fact("HasFood", ())}),
            delete_effects=frozenset({Fact("FoodAvailable", (food_loc,))}),
            cost=1.0,
        )
    )

    # Build Shelter (requires wood at shelter_loc)
    actions.append(
        GroundAction(
            name=f"BuildShelter({shelter_loc})",
            preconditions=frozenset({Fact("At", (shelter_loc,)), Fact("HasWood", ())}),
            add_effects=frozenset({Fact("ShelterBuilt", ()), Fact("ShelterAt", (shelter_loc,))}),
            delete_effects=frozenset({Fact("HasWood", ())}),
            cost=2.0,
        )
    )

    # Store Food (requires shelter built and food held)
    actions.append(
        GroundAction(
            name=f"StoreFood({shelter_loc})",
            preconditions=frozenset({Fact("At", (shelter_loc,)), Fact("ShelterBuilt", ()), Fact("HasFood", ())}),
            add_effects=frozenset({Fact("FoodStored", ())}),
            delete_effects=frozenset({Fact("HasFood", ())}),
            cost=1.0,
        )
    )

    # Light Campfire (requires wood or shelter)
    actions.append(
        GroundAction(
            name=f"LightCampfire({shelter_loc})",
            preconditions=frozenset({Fact("At", (shelter_loc,)), Fact("ShelterBuilt", ())}),
            add_effects=frozenset({Fact("FireLit", ())}),
            delete_effects=frozenset(),
            cost=1.0,
        )
    )

    # Survive Winter (requires shelter built, food stored, fire lit)
    actions.append(
        GroundAction(
            name=f"SurviveWinter({shelter_loc})",
            preconditions=frozenset(
                {
                    Fact("At", (shelter_loc,)),
                    Fact("ShelterBuilt", ()),
                    Fact("FoodStored", ()),
                    Fact("FireLit", ()),
                }
            ),
            add_effects=frozenset({Fact("WinterSurvived", ())}),
            delete_effects=frozenset(),
            cost=5.0,
        )
    )

    return PlanningProblem(
        initial_facts=initial_facts,
        goal_facts=goal_facts,
        actions=tuple(actions),
    )
