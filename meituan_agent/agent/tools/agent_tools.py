import os
import re
from collections import Counter

from langchain_core.tools import tool

from solver.solver_loader import get_solver
from utils.path_tool import get_abs_path


DEFAULT_CASE_PATH = get_abs_path("example/large_seed301.txt")
CASE_PATH_PATTERN = re.compile(r"([^\s\"']+\.txt)")

CURRENT_CASE_INPUT_TEXT = None
CURRENT_CASE_SOURCE = None
CURRENT_PREFERRED_SOLVER = "sa"
LAST_ASSIGNMENTS = None
LAST_ASSIGNMENT_SUMMARY = None


def set_case_context(
    input_text: str | None = None,
    case_source: str | None = None,
    preferred_solver: str | None = None,
):
    global CURRENT_CASE_INPUT_TEXT, CURRENT_CASE_SOURCE, CURRENT_PREFERRED_SOLVER
    CURRENT_CASE_INPUT_TEXT = input_text if input_text and input_text.strip() else None
    CURRENT_CASE_SOURCE = case_source
    if preferred_solver:
        CURRENT_PREFERRED_SOLVER = preferred_solver
    else:
        CURRENT_PREFERRED_SOLVER = "sa"


def detect_solver_name(query: str, default: str = "sa") -> str:
    query_lower = query.lower()
    if "baseline" in query_lower or "贪心" in query:
        return "baseline"
    if "sa" in query_lower or "模拟退火" in query:
        return "sa"
    return default


def detect_case_path(query: str) -> str | None:
    match = CASE_PATH_PATTERN.search(query)
    if not match:
        return None
    raw_path = match.group(1)
    if os.path.isabs(raw_path):
        return raw_path
    return get_abs_path(raw_path)


def _resolve_case_path(case_path: str | None = None) -> str:
    resolved_path = case_path or CURRENT_CASE_SOURCE or DEFAULT_CASE_PATH
    if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
        resolved_path = get_abs_path(resolved_path)
    return resolved_path


def load_case_text(case_path: str | None = None, input_text: str | None = None) -> tuple[str, str]:
    effective_input_text = input_text or CURRENT_CASE_INPUT_TEXT
    if effective_input_text is not None and effective_input_text.strip():
        return effective_input_text, CURRENT_CASE_SOURCE or "uploaded_case"

    resolved_path = _resolve_case_path(case_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Case file does not exist: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        return f.read(), resolved_path


def solve_dispatch_case(
    solver_name: str = "sa",
    case_path: str | None = None,
    input_text: str | None = None,
) -> tuple[list, str]:
    solver = get_solver(solver_name)
    case_text, case_source = load_case_text(case_path=case_path, input_text=input_text)
    return solver(case_text), case_source


def format_assignments(assignments: list, solver_name: str, case_source: str, limit: int = 20) -> str:
    total_bundles = len(assignments)
    total_tasks = sum(len(task_id_list_str.split(",")) for task_id_list_str, _ in assignments)

    lines = [
        f"solver: {solver_name}",
        f"case: {case_source}",
        f"bundle_count: {total_bundles}",
        f"covered_task_count: {total_tasks}",
        "selections:",
    ]

    for idx, (task_id_list_str, courier_ids) in enumerate(assignments[:limit], start=1):
        lines.append(f"{idx}. {task_id_list_str} -> {', '.join(courier_ids)}")

    if total_bundles > limit:
        lines.append(f"... truncated {total_bundles - limit} more selections")

    return "\n".join(lines)


def _parse_case_stats(input_text: str) -> dict:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    row_count = 0
    tasks = set()
    couriers = set()
    bundle_sizes = []
    willingness_values = []

    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, _, willingness_str = parts[:4]
        task_ids = [task_id.strip() for task_id in task_id_list_str.split(",") if task_id.strip()]
        row_count += 1
        couriers.add(courier_id.strip())
        bundle_sizes.append(len(task_ids))
        tasks.update(task_ids)
        try:
            willingness_values.append(float(willingness_str))
        except ValueError:
            pass

    bundle_counter = Counter(bundle_sizes)
    avg_willingness = sum(willingness_values) / len(willingness_values) if willingness_values else 0.0

    return {
        "row_count": row_count,
        "task_count": len(tasks),
        "courier_count": len(couriers),
        "avg_willingness": avg_willingness,
        "bundle_counter": dict(sorted(bundle_counter.items())),
    }


@tool
def solve_dispatch_case_tool(
    solver_name: str = "",
    case_path: str = "",
    limit: int = 20,
) -> str:
    """
    对调度 case 运行求解器，返回任务分配选择结果。
    solver_name 可选值：sa、baseline。若为空，默认使用当前上下文或 sa。
    case_path 可选；如果为空，则优先使用当前上传的 case，否则使用默认样例。
    """
    global LAST_ASSIGNMENTS, LAST_ASSIGNMENT_SUMMARY

    resolved_solver_name = solver_name.strip() or CURRENT_PREFERRED_SOLVER or "sa"
    resolved_case_path = case_path.strip() or None
    assignments, case_source = solve_dispatch_case(
        solver_name=resolved_solver_name,
        case_path=resolved_case_path,
    )
    summary = format_assignments(
        assignments=assignments,
        solver_name=resolved_solver_name,
        case_source=case_source,
        limit=limit,
    )
    LAST_ASSIGNMENTS = assignments
    LAST_ASSIGNMENT_SUMMARY = summary
    return summary


@tool
def inspect_case_tool(case_path: str = "") -> str:
    """
    查看当前 case 的基础统计信息，例如候选行数、任务数、骑手数、平均意愿值、bundle 大小分布。
    如果 case_path 为空，则优先检查当前上传的 case，否则使用默认样例。
    """
    input_text, case_source = load_case_text(case_path=case_path.strip() or None)
    stats = _parse_case_stats(input_text)
    return (
        f"case: {case_source}\n"
        f"candidate_rows: {stats['row_count']}\n"
        f"task_count: {stats['task_count']}\n"
        f"courier_count: {stats['courier_count']}\n"
        f"avg_willingness: {stats['avg_willingness']:.4f}\n"
        f"bundle_size_distribution: {stats['bundle_counter']}"
    )


@tool
def list_solver_tool() -> str:
    """
    查看可用 solver 以及它们的适用场景。
    """
    return (
        "sa: 模拟退火增强版主算法，通常结果更强，适合正式求解。\n"
        "baseline: 贪心基线算法，适合快速对照和基准比较。"
    )


@tool
def get_last_solution_tool() -> str:
    """
    获取最近一次求解结果，适合用户追问“刚才那个结果再展示一下”。
    """
    if LAST_ASSIGNMENT_SUMMARY is None:
        return "当前还没有已缓存的求解结果。请先运行一次求解。"
    return LAST_ASSIGNMENT_SUMMARY
