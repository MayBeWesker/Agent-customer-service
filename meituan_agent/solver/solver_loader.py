import importlib.util
from functools import lru_cache

from utils.path_tool import get_abs_path


SOLVER_PATHS = {
    "sa": get_abs_path("example/25号-15-sa.py"),
    "baseline": get_abs_path("example/example_solution.py"),
}


@lru_cache(maxsize=None)
def _load_solver_module(solver_name: str):
    if solver_name not in SOLVER_PATHS:
        raise ValueError(f"Unsupported solver: {solver_name}")

    module_path = SOLVER_PATHS[solver_name]
    spec = importlib.util.spec_from_file_location(f"meituan_{solver_name}_solver", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load solver from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_solver(solver_name: str = "sa"):
    module = _load_solver_module(solver_name)
    solve = getattr(module, "solve", None)
    if solve is None:
        raise AttributeError(f"Solver {solver_name} does not define solve(input_text)")
    return solve

