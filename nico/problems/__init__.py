from .base import Instance, index_instance, repeat_instance, tile_instance, to_device
from .cvrp import CVRP, capacity_for_size
from .tsp import TSP


def make_problem(name: str, **kwargs):
    if name == "tsp":
        return TSP()
    if name == "cvrp":
        return CVRP(**kwargs)
    raise ValueError(f"Unknown problem {name!r}")


def problem_from_config(cfg):
    if cfg.problem == "cvrp":
        return CVRP(extra_routes=cfg.data.extra_routes, operators=tuple(cfg.model.operators))
    if tuple(cfg.model.operators) != ("2opt",):
        raise ValueError("The TSP supports only the 2opt operator")
    return make_problem(cfg.problem)


__all__ = [
    "CVRP", "TSP", "Instance", "capacity_for_size", "index_instance", "make_problem", "problem_from_config",
    "repeat_instance", "tile_instance", "to_device",
]
