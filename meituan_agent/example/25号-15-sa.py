"""
从全局的角度来看，我们需要构建一个 agent 进行这个问题的求解，
这个算法是作为我们 agent 的核心算法进行计算的

模拟退火增强版 Solver
===================
接口保持不变：
    solve(input_text: str) -> list

输出格式：
    [(task_id_list_str, [courier_id]), ...]


平均惩罚分数
748.06
完成算例
10 / 10
high_noise_seed601
517.64
30/30(100%)
8496ms
large_seed301
718.42
40/40(100%)
8943ms
large_seed302
680.89
40/40(100%)
8962ms
low_willingness_seed501
1,840.90
30/30(100%)
9226ms
medium_seed201
529.52
30/30(100%)
8672ms
medium_seed202
541.23
30/30(100%)
8591ms
medium_seed203
529.22
30/30(100%)
8532ms
scarce_couriers_seed401
1,638.75
40/40(100%)
8455ms
small_seed100
322.34
15/15(100%)
7961ms
tiny_seed42
161.71
6/6(100%)
7923ms

"""

import math
import random
import time


PROXY_WILLINGNESS_WEIGHT = 45.0
PAIR_SEARCH_BETAS = (0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 60.0)
# 当前额外时间没有稳定转化为收益，先把总预算回收到更稳的档位。
TIME_LIMIT_SEC = 9.8
FINAL_RESERVE_SEC = 0.40
BASE_PRIMARY_RATIO = 0.68
# 给专项分支留出明确时间窗，避免 base 吃满之后 scarce/low 实际上跑不到。
BASE_ANNEAL_RATIO = 0.80
SCARCE_BRANCH_RATIO = 0.93
SCARCE_BRANCH_BUDGET_SEC = 0.60
ENABLE_LOW_PORTFOLIO_SEED = False
BIG_COST = 10 ** 12
ANNEALING_START_TEMP = 18.0
ANNEALING_END_TEMP = 0.15
ANNEALING_FREE_COURIERS = 5
ANNEALING_PER_BUNDLE_LIMIT = 4
ANNEALING_LOCAL_OPTION_LIMIT = 96
BACKUP_PER_TASK_LIMIT = 3
SMALL_CASE_SCAN_WEIGHTS = (20.0, 35.0, 45.0, 60.0, 80.0, 110.0)
LARGE_CASE_SCAN_WEIGHTS = (30.0, 45.0, 65.0)
FRONTIER_EXTRA_TOPK = 4
SINGLE_MATRIX_RETAIN_WEIGHT = 90.0
BACKUP_GAIN_ALPHA = 2.0
SCARCE_BIG_MOVE_PROB = 0.20
SCARCE_BIG_REMOVE_CHOICES = (4, 5)
SCARCE_ANNEAL_OPTION_LIMIT = 64
ENABLE_SCARCE_GREEDY_REFILL = False
LOW_PRIMARY_SWAP_SCORE_SLACK = 16.0
LOW_PRIMARY_SWAP_W_GAIN = 0.08
LOW_SINGLE_TASK_TOPK = 20
LOW_SPLIT_SINGLE_TOPK = 6
LOW_SINGLE_W_GAIN = 0.05
LOW_SINGLE_SCORE_BASE = 30.0
LOW_SINGLE_SCORE_GAIN = 80.0
LOW_RISK_ALPHA = 2.0
LOW_RISK_WEIGHT = 180.0
LOW_STATIC_AVG_BEST_SINGLE_THRESHOLD = 0.45
LOW_STATIC_AVG_CANDIDATE_THRESHOLD = 0.35
LOW_CANDIDATE_RATIO_THRESHOLD = 0.70
LOW_COURIER_RATIO_THRESHOLD = 0.70
LOW_TASK_RATIO_THRESHOLD = 0.50
# 本地 ILP 分支扫描后，-2.5 在当前样例上是比较稳的一档。
LOW_ALPHA_SCORE = -2.5
LOW_MATCHING_PARAMS = ((90.0, 80.0), (120.0, 120.0), (150.0, 160.0))
LOW_ILP_TOPK_PER_BUNDLE = 2
LOW_ILP_TOPK_PER_TASK = 18
LOW_ILP_SINGLETON_GUARD_K = 4
LOW_ILP_BRANCH_OPTION_LIMIT = 12
LOW_MATCHING_TIME_BUDGET = 0.30
LOW_MATCHING_TIME_BUDGET_STRONG = 0.50
LOW_BEAM_SIZE = 16
LOW_BEAM_SIZE_STRONG = 20
LOW_BEAM_PER_ANCHOR_TOPK = 8
LOW_BEAM_ACCEPT_WEIGHT = 90.0
LOW_BEAM_RISK_WEIGHT = 120.0
LOW_BEAM_BETA = 90.0
LOW_BEAM_GAMMA = 120.0
LOW_BEAM_BUNDLE_BONUS = 4.0
LOW_BEAM_TIME_BUDGET = 0.30
LOW_BEAM_TIME_BUDGET_STRONG = 0.45
LOW_ILP_TIME_BUDGET_STRONG = 0.55
LOW_ILP_TIME_BUDGET_WEAK = 0.25
ENABLE_LOW_MATCHING = True
ENABLE_LOW_BEAM = True
ENABLE_LOW_RISK_REPLACEMENT = False
ENABLE_LOW_ILP = False
SCARCE_BEAM_SIZE = 6
SCARCE_BEAM_PER_ANCHOR = 6
SCARCE_DESTROY_CHOICES = (3, 4)
SCARCE_MAX_REPAIR_TASK_COUNT = 8
SCARCE_VALUE_MARGIN = 30.0
ENABLE_SCARCE_BEAM_REPAIR = False
SCARCE_USE_EXACT_REPAIR = True
SCARCE_EXACT_MAX_TASK_COUNT = 8
SCARCE_EXACT_MAX_CANDIDATES = 90
SCARCE_EXACT_TOP_RESULTS = 3
SCARCE_EXACT_NODE_LIMIT = 6000
SCARCE_EXACT_BUNDLE_GAIN_WEIGHT = 4.0
SCARCE_EXACT_SELECTED_PENALTY = 3.0
DEBUG_SCENE = False


# candidate tuple layout
# 0 task_id_list_str
# 1 task_ids(tuple)
# 2 courier_id
# 3 total_score
# 4 willingness
# 5 idx
# 6 task_mask
# 7 courier_idx
# 8 task_cnt


def _make_candidate(task_str, task_ids, courier_id, score, willingness, idx, task_mask, courier_idx):
    return (
        task_str,
        task_ids,
        courier_id,
        score,
        willingness,
        idx,
        task_mask,
        courier_idx,
        len(task_ids),
    )


def _alpha_rank_cost(cand, alpha=LOW_ALPHA_SCORE):
    willingness = max(cand[4], 1e-6)
    return cand[3] * (willingness ** alpha)


def _low_candidate_value(cand):
    return 100.0 * cand[4] * cand[8] - cand[3]


def _parse_input(input_text):
    lines = input_text.strip().splitlines()
    if not lines:
        return {
            "candidates": [],
            "task_names": [],
            "courier_names": [],
            "bundle_candidates": {},
            "bundle_frontier": {},
            "task_to_bundle_candidates": {},
            "bundle_backup_potential": {},
            "single_matrix": [],
            "single_task_options": [],
            "candidate_lookup": {},
        }

    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    raw_rows = []
    all_tasks = set()
    all_couriers = set()

    for idx, line in enumerate(lines[start:]):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            parts = line.split()
            if len(parts) < 4:
                continue

        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue

        task_ids = tuple(t.strip() for t in task_id_list_str.split(",") if t.strip())
        courier_id = courier_id.strip()
        if not task_ids or not courier_id:
            continue

        if willingness < 0.0:
            willingness = 0.0
        elif willingness > 1.0:
            willingness = 1.0

        task_str = task_id_list_str.strip()
        raw_rows.append((idx, task_str, task_ids, courier_id, score, willingness))
        all_tasks.update(task_ids)
        all_couriers.add(courier_id)

    task_names = sorted(all_tasks)
    courier_names = sorted(all_couriers)
    task_to_idx = {}
    courier_to_idx = {}

    for i, task_id in enumerate(task_names):
        task_to_idx[task_id] = i
    for i, courier_id in enumerate(courier_names):
        courier_to_idx[courier_id] = i

    candidates = []
    bundle_candidates = {}
    single_matrix = [[None for _ in courier_names] for _ in task_names]
    candidate_lookup = {}

    for idx, task_str, task_ids, courier_id, score, willingness in raw_rows:
        task_mask = 0
        for task_id in task_ids:
            task_mask |= 1 << task_to_idx[task_id]
        candidate = _make_candidate(
            task_str,
            task_ids,
            courier_id,
            score,
            willingness,
            idx,
            task_mask,
            courier_to_idx[courier_id],
        )
        candidates.append(candidate)
        candidate_lookup[(task_str, courier_id)] = (score, willingness)
        if task_mask not in bundle_candidates:
            bundle_candidates[task_mask] = []
        bundle_candidates[task_mask].append(candidate)
        if len(task_ids) == 1:
            task_idx = task_to_idx[task_ids[0]]
            old = single_matrix[task_idx][candidate[7]]
            # 这里不再让“后出现的候选”直接覆盖前面的候选，
            # 而是保留 score / willingness 组合更优的那个，
            # 这样匈牙利匹配看到的单订单输入会更干净。
            if old is None:
                single_matrix[task_idx][candidate[7]] = candidate
            else:
                old_key = old[3] - SINGLE_MATRIX_RETAIN_WEIGHT * old[4]
                new_key = candidate[3] - SINGLE_MATRIX_RETAIN_WEIGHT * candidate[4]
                if new_key < old_key:
                    single_matrix[task_idx][candidate[7]] = candidate

    bundle_frontier = {}
    task_to_bundle_candidates = {}
    for task_mask, items in bundle_candidates.items():
        # The framework note about noisy candidates is valid: for the same
        # bundle, many couriers are dominated by another courier with both
        # lower score and higher willingness. Removing them makes local search
        # both faster and more stable.
        frontier_items = _pareto_frontier(items)
        bundle_frontier[task_mask] = frontier_items
        for task_idx in _mask_to_indices(task_mask):
            if task_idx not in task_to_bundle_candidates:
                task_to_bundle_candidates[task_idx] = []
            task_to_bundle_candidates[task_idx].append((task_mask, frontier_items))

    bundle_backup_potential = _build_bundle_backup_potential(bundle_frontier)
    single_task_options = [[] for _ in task_names]
    for cand in candidates:
        if cand[8] != 1:
            continue
        task_idx = cand[6].bit_length() - 1
        single_task_options[task_idx].append(cand)
    for task_idx in range(len(task_names)):
        # 低意愿修复里优先保留 alpha-score 更优的单单候选，
        # 会比只按 willingness 排序更容易留下“高意愿但分数还能接受”的替代项。
        single_task_options[task_idx].sort(key=lambda c: (_alpha_rank_cost(c), c[3], -c[4], c[5]))
        single_task_options[task_idx] = single_task_options[task_idx][:LOW_SINGLE_TASK_TOPK]

    return {
        "candidates": candidates,
        "task_names": task_names,
        "courier_names": courier_names,
        "bundle_candidates": bundle_candidates,
        "bundle_frontier": bundle_frontier,
        "task_to_bundle_candidates": task_to_bundle_candidates,
        "bundle_backup_potential": bundle_backup_potential,
        "single_matrix": single_matrix,
        "single_task_options": single_task_options,
        "candidate_lookup": candidate_lookup,
    }


def _pareto_frontier(items):
    ordered = sorted(items, key=lambda c: (-c[4], c[3], c[7], c[5]))
    frontier = []
    best_score = float("inf")
    for cand in ordered:
        if cand[3] + 1e-12 < best_score:
            frontier.append(cand)
            best_score = cand[3]

    frontier_ids = set((cand[7], cand[5]) for cand in frontier)
    extra = []
    for cand in sorted(items, key=lambda c: (c[3] - 90.0 * c[4] * c[8], c[3], -c[4], c[5])):
        cand_id = (cand[7], cand[5])
        if cand_id in frontier_ids:
            continue
        extra.append(cand)
        if len(extra) >= FRONTIER_EXTRA_TOPK:
            break

    # 只保 Pareto 会有点过度剪枝；补一小批 adjusted-score 很强的候选，
    # 能给组合搜索保留一些“骑手身份有优势”的替代项。
    combined = frontier + extra
    combined.sort(key=lambda c: (c[3] - 90.0 * c[4] * c[8], c[3], -c[4], c[5]))
    return combined


def _build_bundle_backup_potential(bundle_frontier):
    potential = {}
    for task_mask, items in bundle_frontier.items():
        willingness_values = []
        for cand in items[: max(BACKUP_PER_TASK_LIMIT, 3)]:
            willingness_values.append(cand[4])
        willingness_values.sort(reverse=True)
        if not willingness_values:
            potential[task_mask] = 0.0
            continue
        fail_prob = 1.0
        for w in willingness_values[:3]:
            fail_prob *= (1.0 - w)
        combined_prob = 1.0 - fail_prob
        potential[task_mask] = max(0.0, combined_prob - willingness_values[0])
    return potential


def _evaluate(selected):
    covered = 0
    expected_accept = 0.0
    total_score = 0.0
    total_willingness = 0.0

    for cand in selected:
        covered += cand[8]
        expected_accept += cand[4] * cand[8]
        total_score += cand[3]
        total_willingness += cand[4]

    return {
        "selected": list(selected),
        "covered": covered,
        "expected_accept": expected_accept,
        "total_score": total_score,
        "total_willingness": total_willingness,
    }


def _ranking_key(solution):
    adjusted_cost = solution["total_score"] - PROXY_WILLINGNESS_WEIGHT * solution["expected_accept"]
    return (
        solution["covered"],
        -adjusted_cost,
        solution["expected_accept"],
        -solution["total_score"],
        solution["total_willingness"],
        -len(solution["selected"]),
    )


def _solution_value(solution):
    return (
        solution["covered"] * 100000.0
        + solution["expected_accept"] * PROXY_WILLINGNESS_WEIGHT
        - solution["total_score"]
        + 0.01 * solution["total_willingness"]
        - 0.001 * len(solution["selected"])
    )


def _get_scene_flags(problem, solution=None):
    env = _estimate_environment(problem)
    low_willingness_mode = (
        env["avg_best_single_w"] < LOW_STATIC_AVG_BEST_SINGLE_THRESHOLD
        or env["avg_candidate_w"] < LOW_STATIC_AVG_CANDIDATE_THRESHOLD
        or env.get("low_candidate_ratio", 0.0) >= LOW_CANDIDATE_RATIO_THRESHOLD
        or env.get("low_courier_ratio", 0.0) >= LOW_COURIER_RATIO_THRESHOLD
        or env.get("low_task_ratio", 0.0) >= LOW_TASK_RATIO_THRESHOLD
    )
    scarce_mode = env["courier_task_ratio"] < 1.20
    avg_primary_w = None
    if solution is not None and solution["covered"] > 0:
        avg_primary_w = solution["expected_accept"] / float(solution["covered"])
        risky_assignments = 0
        for cand in solution["selected"]:
            if cand[4] < 0.28 or (cand[8] >= 2 and cand[4] < 0.38):
                risky_assignments += 1
        if avg_primary_w < 0.60 or risky_assignments >= max(2, len(solution["selected"]) // 5):
            low_willingness_mode = True
    return env, low_willingness_mode, scarce_mode, avg_primary_w


def _solution_value_for_scene(problem, solution, backup_gain_hint=0.0):
    _, low_willingness_mode, scarce_mode, _ = _get_scene_flags(problem, solution)
    if low_willingness_mode:
        # 低意愿场景更怕失败，所以更强调接单概率和 backup 的收益。
        return (
            solution["covered"] * 100000.0
            + solution["expected_accept"] * 100.0
            - solution["total_score"]
            + backup_gain_hint * 200.0
            + 0.01 * solution["total_willingness"]
            - 0.001 * len(solution["selected"])
        )
    if scarce_mode:
        # 文档里的判断是对的：scarce 场景如果用过强的 bundle 偏置，
        # SA 会持续朝“更少 assignment”而不是“更低真实惩罚”走。
        return _solution_value(solution)
    return _solution_value(solution)


def _low_solution_value(solution):
    return (
        solution["covered"] * 100000.0
        + solution["expected_accept"] * 100.0
        - solution["total_score"]
        + 0.01 * solution["total_willingness"]
        - 0.001 * len(solution["selected"])
    )


def _is_strong_low_mode(env, avg_primary_w=None):
    return (
        env["avg_best_single_w"] < 0.50
        or env["avg_candidate_w"] < 0.35
        or env.get("low_courier_ratio", 0.0) >= 0.70
        or env.get("low_task_ratio", 0.0) >= 0.50
        or (avg_primary_w is not None and avg_primary_w < 0.60)
    )


def _scarce_solution_value(solution):
    bundle_gain = solution["covered"] - len(solution["selected"])
    return (
        solution["covered"] * 100000.0
        - solution["total_score"]
        + solution["expected_accept"] * 45.0
        - len(solution["selected"]) * 5.0
        + bundle_gain * 3.0
    )


def _bundle_sort_key(cand, beta):
    return (cand[3] - beta * cand[4] * cand[8], cand[3], -cand[4], cand[5])


def _get_search_profile(problem):
    candidate_count = len(problem["candidates"])
    task_count = len(problem["task_names"])

    if task_count <= 16 or candidate_count <= 4000:
        return {
            "scan_weights": (25.0, 45.0, 70.0, 100.0),
            "scan_bundle_bonus": (0.0, 6.0, 10.0),
            "scan_rarity_bonus": (0.0, 4.0),
            "anneal_option_limit": 144,
        }
    if task_count <= 30 or candidate_count <= 18000:
        return {
            "scan_weights": (30.0, 50.0, 75.0),
            "scan_bundle_bonus": (0.0, 6.0),
            "scan_rarity_bonus": (0.0, 3.0),
            "anneal_option_limit": 120,
        }
    return {
        "scan_weights": (35.0, 55.0),
        "scan_bundle_bonus": (0.0, 6.0),
        "scan_rarity_bonus": (0.0, 2.0),
        "anneal_option_limit": ANNEALING_LOCAL_OPTION_LIMIT,
    }


def _construct_greedy(candidates, key_fn):
    selected = []
    used_task_mask = 0
    used_courier_mask = 0

    for cand in sorted(candidates, key=key_fn):
        courier_bit = 1 << cand[7]
        if used_courier_mask & courier_bit:
            continue
        if used_task_mask & cand[6]:
            continue
        selected.append(cand)
        used_courier_mask |= courier_bit
        used_task_mask |= cand[6]

    return _evaluate(selected)


def _make_static_features(candidates):
    task_freq = {}
    courier_freq = {}
    for cand in candidates:
        courier_id = cand[2]
        courier_freq[courier_id] = courier_freq.get(courier_id, 0) + 1
        for task_id in cand[1]:
            task_freq[task_id] = task_freq.get(task_id, 0) + 1
    return task_freq, courier_freq


def _run_greedy_portfolio(problem, deadline):
    candidates = problem["candidates"]
    if not candidates:
        return _evaluate([])
    profile = _get_search_profile(problem)
    bundle_backup_potential = problem["bundle_backup_potential"]
    env = _estimate_environment(problem)
    scarce_mode = env["courier_task_ratio"] < 1.2

    task_freq, courier_freq = _make_static_features(candidates)

    def rarity(cand):
        value = 0.0
        for task_id in cand[1]:
            value += 1.0 / max(1, task_freq[task_id])
        value += 0.1 / max(1, courier_freq[cand[2]])
        return value

    def backup_potential(cand):
        return bundle_backup_potential.get(cand[6], 0.0)

    def greedy_backup_weight(cand):
        if scarce_mode:
            # scarce 场景本来就缺备用骑手，初始化阶段过早追逐 backup 潜力
            # 容易把主解带向“看起来可补、实际上补不到”的伪收益结构。
            return 0.0
        return backup_potential(cand)

    def alpha_score(cand):
        return _alpha_rank_cost(cand)

    key_functions = [
        # 主 key 继续保留当前最稳的 adjusted-score 思路。
        lambda c: (c[3] - 45.0 * c[4] * c[8] - 22.0 * greedy_backup_weight(c) * c[8], c[5]),
        # alpha-score 会把高 willingness 候选更自然地提前，
        # 比纯 score key 更适合 low-like 场景，也能减少无效的纯成本偏置。
        lambda c: (alpha_score(c) - 10.0 * greedy_backup_weight(c) * c[8] - 2.0 * (c[8] - 1), c[3], c[5]),
        # rarity-aware alpha key 保留一点“稀缺任务优先”的能力，
        # 但不再像旧 key 那样叠太多分量，避免把主解带偏。
        lambda c: (alpha_score(c) - 4.0 * rarity(c) - 8.0 * greedy_backup_weight(c) * c[8], c[3], c[5]),
    ]

    best = _construct_greedy(candidates, key_functions[0])
    for key_fn in key_functions[1:]:
        if time.perf_counter() >= deadline:
            break
        current = _construct_greedy(candidates, key_fn)
        if _ranking_key(current) > _ranking_key(best):
            best = current

    # Time-adaptive parameter scan from the framework notes:
    # smaller instances deserve a wider sweep of willingness / bundle weights,
    # while large instances keep more time for exact repair and annealing.
    for weight in profile["scan_weights"]:
        for bundle_bonus in profile["scan_bundle_bonus"]:
            for rarity_bonus in profile["scan_rarity_bonus"]:
                if time.perf_counter() >= deadline:
                    return best
                current = _construct_greedy(
                    candidates,
                    lambda c, weight=weight, bundle_bonus=bundle_bonus, rarity_bonus=rarity_bonus: (
                        alpha_score(c)
                        - weight * c[4] * c[8]
                        - bundle_bonus * (c[8] - 1)
                        - rarity_bonus * rarity(c),
                        - 18.0 * greedy_backup_weight(c) * c[8],
                        c[3],
                        c[5],
                    ),
                )
                if _ranking_key(current) > _ranking_key(best):
                    best = current
    return best


def _can_run_exact_singletons(problem):
    task_names = problem["task_names"]
    courier_names = problem["courier_names"]
    single_matrix = problem["single_matrix"]
    if not task_names or len(task_names) > len(courier_names):
        return False
    for row in single_matrix:
        has_candidate = False
        for cand in row:
            if cand is not None:
                has_candidate = True
                break
        if not has_candidate:
            return False
    return True


def _hungarian_assign(problem, remaining_task_indices, blocked_courier_mask, beta):
    if not remaining_task_indices:
        return []

    courier_names = problem["courier_names"]
    single_matrix = problem["single_matrix"]
    available_couriers = []
    for courier_idx in range(len(courier_names)):
        if not (blocked_courier_mask & (1 << courier_idx)):
            available_couriers.append(courier_idx)

    if len(remaining_task_indices) > len(available_couriers):
        return None

    n = len(remaining_task_indices)
    m = len(available_couriers)
    cost = [[BIG_COST for _ in range(m)] for _ in range(n)]

    for row_idx, task_idx in enumerate(remaining_task_indices):
        matrix_row = single_matrix[task_idx]
        for col_idx, courier_idx in enumerate(available_couriers):
            cand = matrix_row[courier_idx]
            if cand is not None:
                cost[row_idx][col_idx] = cand[3] - beta * cand[4]

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for row in range(1, n + 1):
        p[0] = row
        col0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[col0] = True
            row0 = p[col0]
            delta = float("inf")
            col1 = 0
            for col in range(1, m + 1):
                if used[col]:
                    continue
                cur = cost[row0 - 1][col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col

            for col in range(m + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta

            col0 = col1
            if p[col0] == 0:
                break

        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = [-1] * n
    for col in range(1, m + 1):
        row = p[col]
        if row:
            assignment[row - 1] = col - 1

    selected = []
    for row_idx, assigned_col in enumerate(assignment):
        if assigned_col < 0:
            return None
        courier_idx = available_couriers[assigned_col]
        cand = single_matrix[remaining_task_indices[row_idx]][courier_idx]
        if cand is None:
            return None
        selected.append(cand)

    return selected


def _local_merge_key(selected, weight):
    solution = _evaluate(selected)
    adjusted_cost = solution["total_score"] - weight * solution["expected_accept"]
    return (
        -adjusted_cost,
        solution["expected_accept"],
        -solution["total_score"],
        solution["total_willingness"],
        -len(solution["selected"]),
    )


def _find_best_pair_merge(problem, singles, blocked_courier_mask, beta):
    bundle_frontier = problem["bundle_frontier"]
    used_courier_mask = blocked_courier_mask
    for cand in singles:
        used_courier_mask |= 1 << cand[7]

    best_pair = None
    best_gain_key = None

    for i, left in enumerate(singles):
        for right in singles[i + 1:]:
            pair_mask = left[6] | right[6]
            pair_options = bundle_frontier.get(pair_mask)
            if not pair_options:
                continue

            current_key = _local_merge_key((left, right), beta)
            current_bits = (1 << left[7]) | (1 << right[7])
            for pair_cand in pair_options:
                pair_bit = 1 << pair_cand[7]
                if used_courier_mask & pair_bit and not (current_bits & pair_bit):
                    continue
                new_key = _local_merge_key((pair_cand,), beta)
                if new_key <= current_key:
                    continue
                gain_key = (
                    new_key[0] - current_key[0],
                    new_key[1] - current_key[1],
                    new_key[2] - current_key[2],
                    new_key[3] - current_key[3],
                    new_key[4] - current_key[4],
                )
                if best_gain_key is None or gain_key > best_gain_key:
                    best_gain_key = gain_key
                    best_pair = pair_cand
                break

    return best_pair


def _solve_singleton_plus_pairs(problem, beta, deadline):
    fixed_pairs = []
    fixed_task_mask = 0
    blocked_courier_mask = 0
    task_count = len(problem["task_names"])

    while time.perf_counter() < deadline:
        remaining_task_indices = []
        for task_idx in range(task_count):
            if not (fixed_task_mask & (1 << task_idx)):
                remaining_task_indices.append(task_idx)

        singleton_solution = _hungarian_assign(problem, remaining_task_indices, blocked_courier_mask, beta)
        if singleton_solution is None:
            return None

        merge = _find_best_pair_merge(problem, singleton_solution, blocked_courier_mask, beta)
        if merge is None:
            return _evaluate(fixed_pairs + singleton_solution)

        fixed_pairs.append(merge)
        fixed_task_mask |= merge[6]
        blocked_courier_mask |= 1 << merge[7]

    remaining_task_indices = []
    for task_idx in range(task_count):
        if not (fixed_task_mask & (1 << task_idx)):
            remaining_task_indices.append(task_idx)

    singleton_solution = _hungarian_assign(problem, remaining_task_indices, blocked_courier_mask, beta)
    if singleton_solution is None:
        return None

    return _evaluate(fixed_pairs + singleton_solution)


def _run_exact_portfolio(problem, deadline):
    if not _can_run_exact_singletons(problem):
        return None

    betas = _make_beta_grid(problem)
    best = None
    for beta in betas:
        if time.perf_counter() >= deadline:
            break
        current = _solve_singleton_plus_pairs(problem, beta, deadline)
        if current is None:
            continue
        if best is None or _ranking_key(current) > _ranking_key(best):
            best = current
    return best


def _estimate_environment(problem):
    task_count = len(problem["task_names"])
    courier_count = len(problem["courier_names"])
    single_matrix = problem["single_matrix"]
    candidates = problem["candidates"]
    best_single_w = []

    for row in single_matrix:
        row_best = None
        for cand in row:
            if cand is None:
                continue
            if row_best is None or cand[4] > row_best:
                row_best = cand[4]
        if row_best is not None:
            best_single_w.append(row_best)

    avg_best_single_w = 0.0
    if best_single_w:
        avg_best_single_w = sum(best_single_w) / len(best_single_w)

    avg_candidate_w = 0.0
    if candidates:
        avg_candidate_w = sum(cand[4] for cand in candidates) / len(candidates)

    low_candidate_ratio = 0.0
    if candidates:
        low_candidate_ratio = sum(1 for cand in candidates if cand[4] < LOW_STATIC_AVG_CANDIDATE_THRESHOLD) / len(candidates)

    courier_w = {}
    for cand in candidates:
        if cand[7] not in courier_w:
            courier_w[cand[7]] = []
        courier_w[cand[7]].append(cand[4])

    low_courier_ratio = 0.0
    if courier_w:
        low_courier_count = 0
        for ws in courier_w.values():
            if sum(ws) / len(ws) < LOW_STATIC_AVG_CANDIDATE_THRESHOLD:
                low_courier_count += 1
        low_courier_ratio = low_courier_count / len(courier_w)

    task_best_w = [0.0 for _ in problem["task_names"]]
    for cand in candidates:
        for task_idx in _mask_to_indices(cand[6]):
            if cand[4] > task_best_w[task_idx]:
                task_best_w[task_idx] = cand[4]

    low_task_ratio = 0.0
    if task_best_w:
        low_task_ratio = sum(1 for w in task_best_w if w < LOW_STATIC_AVG_BEST_SINGLE_THRESHOLD) / len(task_best_w)

    return {
        "task_count": task_count,
        "courier_count": courier_count,
        "courier_task_ratio": float(courier_count) / max(1, task_count),
        "avg_best_single_w": avg_best_single_w,
        "avg_candidate_w": avg_candidate_w,
        "low_candidate_ratio": low_candidate_ratio,
        "low_courier_ratio": low_courier_ratio,
        "low_task_ratio": low_task_ratio,
    }


def _make_beta_grid(problem):
    env = _estimate_environment(problem)
    betas = list(PAIR_SEARCH_BETAS)

    if env["avg_best_single_w"] < 0.35:
        betas.extend([80.0, 100.0, 130.0, 160.0])
    elif env["avg_best_single_w"] < 0.50:
        betas.extend([70.0, 90.0, 120.0])
    elif env["avg_best_single_w"] < 0.65:
        betas.extend([60.0, 80.0])

    if env["courier_task_ratio"] < 1.15:
        betas.extend([55.0, 75.0])

    betas = sorted(set(betas))
    return betas


def _augment_with_backup_couriers(problem, solution):
    selected = solution["selected"]
    if not selected:
        return []

    env, low_willingness_mode, _, avg_primary_w = _get_scene_flags(problem, solution)
    total_couriers = len(problem["courier_names"])
    spare_couriers = total_couriers - len(selected)

    assignments = []
    used_courier_mask = 0
    for cand in selected:
        assignments.append({
            "task_str": cand[0],
            "task_mask": cand[6],
            "task_cnt": cand[8],
            "courier_ids": [cand[2]],
            "courier_indices": [cand[7]],
            "success_prob": cand[4],
        })
        used_courier_mask |= 1 << cand[7]

    if spare_couriers <= 0:
        return assignments

    if avg_primary_w is None:
        avg_primary_w = 0.0

    target_prob = 0.92
    if env["courier_task_ratio"] >= 1.8:
        target_prob = 0.97
    elif env["courier_task_ratio"] >= 1.35:
        target_prob = 0.95
    if env["avg_best_single_w"] < 0.45:
        target_prob = max(target_prob, 0.985)
    elif env["avg_best_single_w"] < 0.60:
        target_prob = max(target_prob, 0.965)

    backup_weight = 140.0
    backup_weight += 180.0 * max(0.0, 0.75 - env["avg_best_single_w"])
    backup_weight += 40.0 * max(0.0, env["courier_task_ratio"] - 1.0)
    backup_weight += 30.0 * max(0.0, 0.70 - avg_primary_w)

    severe_mode = low_willingness_mode
    desired_extra = min(
        spare_couriers,
        len(assignments) if not severe_mode else len(assignments) * 2,
    )
    used_extra = 0

    while spare_couriers > 0 and used_extra < desired_extra:
        best_choice = None

        for idx, assignment in enumerate(assignments):
            local_backup_limit = BACKUP_PER_TASK_LIMIT
            if severe_mode and assignment["success_prob"] < 0.65:
                local_backup_limit = 5
            elif severe_mode and assignment["success_prob"] < 0.78:
                local_backup_limit = 4

            # 低意愿场景对少数高风险任务放宽 backup 上限，
            # 会比平均分配备用骑手更容易把失败概率压下去。
            if len(assignment["courier_ids"]) >= local_backup_limit:
                continue

            current_prob = assignment["success_prob"]
            fail_prob = 1.0 - current_prob
            if fail_prob <= 1e-9:
                continue

            if not severe_mode and current_prob >= target_prob and spare_couriers <= len(assignments) // 2:
                continue

            # Backup riders also use the frontier view. This keeps the extra
            # couriers focused on real acceptance gain instead of noisy copies
            # with worse score and no willingness benefit.
            bundle_items = problem["bundle_frontier"].get(assignment["task_mask"], [])
            local_best = None

            for cand in bundle_items:
                courier_bit = 1 << cand[7]
                if used_courier_mask & courier_bit:
                    continue

                delta_accept = fail_prob * cand[4]
                new_fail_prob = fail_prob * (1.0 - cand[4])
                nonlinear_gain = (fail_prob ** BACKUP_GAIN_ALPHA) - (new_fail_prob ** BACKUP_GAIN_ALPHA)
                expected_score_cost = fail_prob * cand[4] * cand[3]
                # 非线性失败概率收益会更优先补强“非常危险”的任务，
                # 比线性 delta_accept 更适合 low-willingness 场景。
                utility = nonlinear_gain * backup_weight * assignment["task_cnt"] - 0.70 * expected_score_cost

                if current_prob < target_prob:
                    utility += 30.0 * (target_prob - current_prob) * nonlinear_gain * assignment["task_cnt"]
                if severe_mode:
                    utility += 20.0 * nonlinear_gain * assignment["task_cnt"]

                if local_best is None or utility > local_best[0]:
                    local_best = (utility, cand)

            if local_best is None:
                continue
            if best_choice is None or local_best[0] > best_choice[0]:
                best_choice = (local_best[0], idx, local_best[1])

        if best_choice is None or best_choice[0] <= 0.0:
            break

        _, idx, cand = best_choice
        assignment = assignments[idx]
        assignment["courier_ids"].append(cand[2])
        assignment["courier_indices"].append(cand[7])
        assignment["success_prob"] = 1.0 - (1.0 - assignment["success_prob"]) * (1.0 - cand[4])

        used_courier_mask |= 1 << cand[7]
        spare_couriers -= 1
        used_extra += 1

    return assignments


def _refine_primary_couriers_for_low_willingness(problem, solution):
    env, low_willingness_mode, _, avg_primary_w = _get_scene_flags(problem, solution)
    if not solution["selected"] or not low_willingness_mode:
        return solution
    if avg_primary_w is not None and avg_primary_w >= 0.72:
        return solution

    selected = list(solution["selected"])
    used_couriers = set()
    for cand in selected:
        used_couriers.add(cand[7])

    ordered_indices = list(range(len(selected)))
    ordered_indices.sort(key=lambda idx: (selected[idx][4], -selected[idx][8], selected[idx][3]))

    improved = False
    for idx in ordered_indices:
        base = selected[idx]
        if base[4] >= 0.82:
            continue

        best_alt = None
        best_gain = 0.0
        for cand in problem["bundle_frontier"].get(base[6], []):
            if cand[7] == base[7]:
                continue
            if cand[7] in used_couriers:
                continue
            willingness_gain = cand[4] - base[4]
            score_delta = cand[3] - base[3]
            if willingness_gain < LOW_PRIMARY_SWAP_W_GAIN:
                continue
            if score_delta > LOW_PRIMARY_SWAP_SCORE_SLACK * base[8]:
                continue

            proxy_gain = PROXY_WILLINGNESS_WEIGHT * base[8] * willingness_gain - score_delta
            if env["avg_best_single_w"] < 0.45:
                proxy_gain += 18.0 * willingness_gain * base[8]
            if proxy_gain > best_gain:
                best_gain = proxy_gain
                best_alt = cand

        if best_alt is None or best_gain <= 0.0:
            continue

        # 低意愿场景先把主骑手从很差的 willingness 拉上来，
        # 后面的 backup 才不会一直在补一个先天过弱的主结构。
        used_couriers.remove(base[7])
        used_couriers.add(best_alt[7])
        selected[idx] = best_alt
        improved = True

    if not improved:
        return solution
    return _evaluate(selected)


def _mask_to_indices(mask):
    indices = []
    while mask:
        lowbit = mask & -mask
        indices.append(lowbit.bit_length() - 1)
        mask ^= lowbit
    return indices


def _build_owner_map(selected, task_count):
    owner = [-1] * task_count
    for bundle_idx, cand in enumerate(selected):
        for task_idx in _mask_to_indices(cand[6]):
            owner[task_idx] = bundle_idx
    return owner


def _local_signature(selected):
    signature = []
    for cand in selected:
        signature.append((cand[0], cand[2]))
    signature.sort()
    return tuple(signature)


def _enumerate_local_replacements(problem, subset_task_mask, allowed_courier_mask, beta):
    bundle_frontier = problem["bundle_frontier"]
    profile = _get_search_profile(problem)
    _, _, scarce_mode, _ = _get_scene_flags(problem)
    candidate_pool = []

    for bundle_mask, items in bundle_frontier.items():
        if bundle_mask & ~subset_task_mask:
            continue
        filtered = []
        for cand in items:
            if allowed_courier_mask & (1 << cand[7]):
                filtered.append(cand)
        if not filtered:
            continue
        filtered.sort(key=lambda cand: _bundle_sort_key(cand, beta))
        candidate_pool.extend(filtered[:ANNEALING_PER_BUNDLE_LIMIT])

    if not candidate_pool:
        return []

    options = []
    current = []
    subset_indices = _mask_to_indices(subset_task_mask)
    local_task_count = len(subset_indices)

    def dfs(uncovered_mask, used_courier_mask):
        option_limit = profile["anneal_option_limit"]
        if scarce_mode:
            option_limit = min(option_limit, SCARCE_ANNEAL_OPTION_LIMIT)
        if len(options) >= option_limit:
            return
        if uncovered_mask == 0:
            local_solution = _evaluate(current)
            local_score = _solution_value_for_scene(problem, local_solution)
            if scarce_mode:
                local_score = _solution_value(local_solution)
            options.append({
                "selected": list(current),
                "score": local_score,
            })
            return

        anchor = uncovered_mask & -uncovered_mask
        for cand in candidate_pool:
            if not (cand[6] & anchor):
                continue
            if cand[6] & ~uncovered_mask:
                continue
            courier_bit = 1 << cand[7]
            if used_courier_mask & courier_bit:
                continue
            current.append(cand)
            dfs(uncovered_mask ^ cand[6], used_courier_mask | courier_bit)
            current.pop()

    dfs(subset_task_mask, 0)
    options.sort(key=lambda item: item["score"], reverse=True)
    return options


def _choose_random_neighbor(problem, current_selected, rng, beta):
    if not current_selected:
        return None

    task_count = len(problem["task_names"])
    env = _estimate_environment(problem)
    scarce_mode = env["courier_task_ratio"] < 1.2
    owner = _build_owner_map(current_selected, task_count)

    remove_count = 1
    if scarce_mode and len(current_selected) >= 5:
        # scarce repair 还不够强时，大邻域太频繁会变成“破坏多、修复少”，
        # 这里保持更保守的触发概率，让搜索先稳住结构。
        if rng.random() < SCARCE_BIG_MOVE_PROB:
            remove_count = min(len(current_selected), rng.choice(SCARCE_BIG_REMOVE_CHOICES))
        else:
            remove_count = 2 if rng.random() < 0.70 else 3
    else:
        roll = rng.random()
        if len(current_selected) >= 3 and roll < 0.25:
            remove_count = 3
        elif len(current_selected) >= 2 and roll < 0.80:
            remove_count = 2

    first_idx = rng.randrange(len(current_selected))
    removed_indices = {first_idx}

    while len(removed_indices) < remove_count:
        seed_idx = rng.choice(list(removed_indices))
        seed_tasks = _mask_to_indices(current_selected[seed_idx][6])
        candidate_idx = -1

        if seed_tasks:
            for _ in range(3):
                anchor_task = rng.choice(seed_tasks)
                partner_task = rng.randrange(task_count)
                if partner_task == anchor_task:
                    continue
                owner_idx = owner[partner_task]
                if owner_idx >= 0 and owner_idx not in removed_indices:
                    candidate_idx = owner_idx
                    break

        if candidate_idx < 0:
            candidate_idx = rng.randrange(len(current_selected))
            while candidate_idx in removed_indices:
                candidate_idx = rng.randrange(len(current_selected))

        removed_indices.add(candidate_idx)

    removed_indices = sorted(removed_indices)
    removed = [current_selected[idx] for idx in removed_indices]

    subset_task_mask = 0
    allowed_courier_mask = 0
    current_local = []
    remaining_selected = []

    for idx, cand in enumerate(current_selected):
        if idx in removed_indices:
            subset_task_mask |= cand[6]
            allowed_courier_mask |= 1 << cand[7]
            current_local.append(cand)
        else:
            remaining_selected.append(cand)

    free_couriers = []
    used_courier_mask = 0
    for cand in remaining_selected:
        used_courier_mask |= 1 << cand[7]
    for courier_idx in range(len(problem["courier_names"])):
        courier_bit = 1 << courier_idx
        if not (used_courier_mask & courier_bit) and not (allowed_courier_mask & courier_bit):
            free_couriers.append(courier_idx)

    rng.shuffle(free_couriers)
    for courier_idx in free_couriers[:ANNEALING_FREE_COURIERS]:
        allowed_courier_mask |= 1 << courier_idx

    options = _enumerate_local_replacements(problem, subset_task_mask, allowed_courier_mask, beta)
    if not options:
        return None

    current_signature = _local_signature(current_local)
    filtered = []
    for option in options:
        if _local_signature(option["selected"]) != current_signature:
            filtered.append(option)
    if not filtered:
        return None

    return {
        "remaining_selected": remaining_selected,
        "current_local": current_local,
        "options": filtered,
    }


def _sample_option(options, temperature, rng):
    if not options:
        return None
    top_count = min(len(options), 8 if temperature > 1.0 else 3)
    top_options = options[:top_count]
    if len(top_options) == 1:
        return top_options[0]

    best_score = top_options[0]["score"]
    weights = []
    denom = max(temperature, 1e-6)
    total = 0.0
    for option in top_options:
        weight = math.exp((option["score"] - best_score) / denom)
        weights.append(weight)
        total += weight

    pick = rng.random() * total
    acc = 0.0
    for option, weight in zip(top_options, weights):
        acc += weight
        if acc >= pick:
            return option
    return top_options[-1]


def _greedy_refill_from_global(problem, remaining_selected, beta):
    used_task_mask = 0
    used_courier_mask = 0
    for cand in remaining_selected:
        used_task_mask |= cand[6]
        used_courier_mask |= 1 << cand[7]

    task_count = len(problem["task_names"])
    full_task_mask = (1 << task_count) - 1
    uncovered_mask = full_task_mask ^ used_task_mask
    if uncovered_mask == 0:
        return list(remaining_selected)

    bundle_backup_potential = problem["bundle_backup_potential"]
    bundle_frontier = problem["bundle_frontier"]
    candidate_pool = []
    for items in bundle_frontier.values():
        candidate_pool.extend(items)

    def refill_key(cand):
        backup_bonus = bundle_backup_potential.get(cand[6], 0.0)
        return (
            cand[3] / cand[8]
            - beta * cand[4]
            - 7.0 * (cand[8] - 1)
            - 10.0 * backup_bonus * cand[8],
            cand[3],
            cand[5],
        )

    selected = list(remaining_selected)
    for cand in sorted(candidate_pool, key=refill_key):
        courier_bit = 1 << cand[7]
        if used_courier_mask & courier_bit:
            continue
        if cand[6] & used_task_mask:
            continue
        if cand[6] & ~uncovered_mask:
            continue
        selected.append(cand)
        used_courier_mask |= courier_bit
        used_task_mask |= cand[6]
        uncovered_mask ^= cand[6]
        if uncovered_mask == 0:
            break

    if uncovered_mask != 0:
        remaining_task_indices = _mask_to_indices(uncovered_mask)
        singleton_fill = _hungarian_assign(problem, remaining_task_indices, used_courier_mask, beta)
        if singleton_fill is None:
            return None
        selected.extend(singleton_fill)

    return selected


def _run_simulated_annealing(problem, initial_solution, deadline):
    if time.perf_counter() >= deadline:
        return initial_solution

    rng = random.Random(20260525)
    current_solution = _evaluate(initial_solution["selected"])
    best_solution = current_solution
    current_value = _solution_value(current_solution)
    best_value = current_value

    total_window = max(0.01, deadline - time.perf_counter())
    while time.perf_counter() < deadline:
        progress = 1.0 - max(0.0, deadline - time.perf_counter()) / total_window
        temperature = ANNEALING_START_TEMP * ((ANNEALING_END_TEMP / ANNEALING_START_TEMP) ** progress)
        beta = 35.0 + 20.0 * progress

        move = _choose_random_neighbor(problem, current_solution["selected"], rng, beta)
        if move is None:
            continue

        env, _, scarce_mode, _ = _get_scene_flags(problem, current_solution)
        proposal = None
        if ENABLE_SCARCE_GREEDY_REFILL and scarce_mode and len(move["current_local"]) >= 4 and rng.random() < 0.65:
            refill_selected = _greedy_refill_from_global(problem, move["remaining_selected"], beta)
            if refill_selected is not None:
                proposal = {
                    "selected": refill_selected[len(move["remaining_selected"]):],
                    "refill_full_selected": refill_selected,
                }

        if proposal is None:
            proposal = _sample_option(move["options"], temperature, rng)
        if proposal is None:
            continue

        if "refill_full_selected" in proposal:
            proposal_solution = _evaluate(proposal["refill_full_selected"])
            proposal_value = _solution_value(proposal_solution)
        else:
            proposal_solution = _evaluate(move["remaining_selected"] + proposal["selected"])
            proposal_value = _solution_value(proposal_solution)
        delta = proposal_value - current_value

        if delta >= 0.0 or rng.random() < math.exp(delta / max(temperature, 1e-6)):
            current_solution = proposal_solution
            current_value = proposal_value

            if _ranking_key(current_solution) > _ranking_key(best_solution):
                best_solution = current_solution
                best_value = current_value
            elif current_value > best_value and _ranking_key(current_solution) == _ranking_key(best_solution):
                best_solution = current_solution
                best_value = current_value

    return best_solution


def _run_base_solver(problem, start_time, deadline, special_mode):
    # 10 秒总预算下，主干仍然是最稳定的收益来源，所以把大部分时间给 base。
    primary_deadline = min(deadline, start_time + TIME_LIMIT_SEC * BASE_PRIMARY_RATIO)
    anneal_deadline = min(deadline, start_time + TIME_LIMIT_SEC * BASE_ANNEAL_RATIO)

    best = _run_greedy_portfolio(problem, primary_deadline)
    exact_solution = _run_exact_portfolio(problem, primary_deadline)
    if exact_solution is not None and _ranking_key(exact_solution) > _ranking_key(best):
        best = exact_solution

    best = _run_simulated_annealing(problem, best, anneal_deadline)
    best = _refine_primary_couriers_for_low_willingness(problem, best)
    return best


def _run_low_willingness_portfolio(problem, deadline):
    candidates = problem["candidates"]
    if not candidates:
        return None

    candidate_solutions = []
    key_functions = [
        lambda c: (-c[4] * c[8], c[3] / c[8], c[3], c[5]),
        lambda c: (c[3] - 120.0 * c[4] * c[8], c[3], c[5]),
    ]

    # low 场景单靠 score-first 很容易把 willingness 太低的候选提前选走，
    # 这里补 willingness-first / high-beta 候选，只作为额外种子，不替换 base。
    for key_fn in key_functions:
        if time.perf_counter() >= deadline:
            break
        current = _construct_greedy(candidates, key_fn)
        candidate_solutions.append(current)

    best = None
    for solution in candidate_solutions:
        if best is None or _prefer_low_solution(solution, best):
            best = solution
    return best


def _hungarian_assign_low_risk(problem, remaining_task_indices, blocked_courier_mask, beta, gamma):
    if not remaining_task_indices:
        return []

    courier_names = problem["courier_names"]
    single_matrix = problem["single_matrix"]
    available_couriers = []
    for courier_idx in range(len(courier_names)):
        if not (blocked_courier_mask & (1 << courier_idx)):
            available_couriers.append(courier_idx)

    if len(remaining_task_indices) > len(available_couriers):
        return None

    n = len(remaining_task_indices)
    m = len(available_couriers)
    cost = [[BIG_COST for _ in range(m)] for _ in range(n)]

    for row_idx, task_idx in enumerate(remaining_task_indices):
        matrix_row = single_matrix[task_idx]
        for col_idx, courier_idx in enumerate(available_couriers):
            cand = matrix_row[courier_idx]
            if cand is not None:
                failure_risk = (1.0 - cand[4]) ** LOW_RISK_ALPHA
                cost[row_idx][col_idx] = cand[3] - beta * cand[4] + gamma * failure_risk

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for row in range(1, n + 1):
        p[0] = row
        col0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[col0] = True
            row0 = p[col0]
            delta = float("inf")
            col1 = 0
            for col in range(1, m + 1):
                if used[col]:
                    continue
                cur = cost[row0 - 1][col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col

            for col in range(m + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta

            col0 = col1
            if p[col0] == 0:
                break

        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = [-1] * n
    for col in range(1, m + 1):
        row = p[col]
        if row:
            assignment[row - 1] = col - 1

    selected = []
    for row_idx, assigned_col in enumerate(assignment):
        if assigned_col < 0:
            return None
        courier_idx = available_couriers[assigned_col]
        cand = single_matrix[remaining_task_indices[row_idx]][courier_idx]
        if cand is None:
            return None
        selected.append(cand)

    return selected


def _low_conservative_pair_merge(problem, singles, deadline):
    selected = list(singles)
    current = _evaluate(selected)
    stats = {"attempt_merge": 0, "accept_merge": 0}

    while time.perf_counter() < deadline:
        best_merge = None
        best_ids = None
        selected_len = len(selected)
        if selected_len < 2:
            break

        used_all = set(cand[7] for cand in selected)
        for left_idx in range(selected_len):
            left = selected[left_idx]
            if left[8] != 1:
                continue
            for right_idx in range(left_idx + 1, selected_len):
                right = selected[right_idx]
                if right[8] != 1:
                    continue
                if left[6] & right[6]:
                    continue
                pair_mask = left[6] | right[6]
                options = problem["bundle_frontier"].get(pair_mask, [])
                if not options:
                    continue
                stats["attempt_merge"] += 1
                used_without_pair = used_all - {left[7], right[7]}
                min_old_w = min(left[4], right[4])
                old_score = left[3] + right[3]
                old_risk = _assignment_failure_risk(left) + _assignment_failure_risk(right)

                for pair in options:
                    if pair[7] in used_without_pair:
                        continue
                    if pair[4] < min_old_w - 0.05:
                        continue
                    if pair[3] > old_score + 20.0:
                        continue
                    new_risk = _assignment_failure_risk(pair)
                    risk_gain = old_risk - new_risk
                    score_delta = pair[3] - old_score
                    utility = LOW_RISK_WEIGHT * risk_gain - score_delta
                    if utility <= 0.0:
                        continue

                    candidate_selected = selected[:left_idx] + selected[left_idx + 1:right_idx] + selected[right_idx + 1:] + [pair]
                    proposal = _evaluate(candidate_selected)
                    if _low_solution_value(proposal) <= _low_solution_value(current):
                        continue
                    if best_merge is None or utility > best_merge[0]:
                        best_merge = (utility, proposal, candidate_selected)
                        best_ids = (left_idx, right_idx)

        if best_merge is None:
            break

        _, current, selected = best_merge
        stats["accept_merge"] += 1

    _debug_scene_info("low_matching_merge", list(stats.items()))
    return current


def _run_low_aware_matching_candidates(problem, deadline, strong_low_mode=False):
    all_tasks = list(range(len(problem["task_names"])))
    params = LOW_MATCHING_PARAMS if strong_low_mode else LOW_MATCHING_PARAMS[:2]
    if strong_low_mode:
        # 只在强 low 下补一组更激进的 matching 参数，扩大高 willingness 主骨架的多样性。
        params = params + ((180.0, 220.0),)
    candidates = []
    stats = {"configs": 0, "full_cover": 0, "best_expected_accept": None, "best_total_score": None, "best_selected_count": None}
    best = None

    for beta, gamma in params:
        if time.perf_counter() >= deadline:
            break
        stats["configs"] += 1
        singles = _hungarian_assign_low_risk(problem, all_tasks, 0, beta, gamma)
        if singles is None:
            continue
        stats["full_cover"] += 1
        solution = _evaluate(singles)
        if time.perf_counter() < deadline:
            merged = _low_conservative_pair_merge(problem, singles, deadline)
            if merged is not None:
                solution = merged
        candidates.append((f"low_matching_{int(beta)}_{int(gamma)}", solution))
        if best is None or _low_solution_value(solution) > _low_solution_value(best):
            best = solution
            stats["best_expected_accept"] = round(solution["expected_accept"], 4)
            stats["best_total_score"] = round(solution["total_score"], 4)
            stats["best_selected_count"] = len(solution["selected"])

    _debug_scene_info("low_matching", list(stats.items()))
    return candidates


def _run_low_aware_matching_solver(problem, deadline, strong_low_mode=False):
    candidates = _run_low_aware_matching_candidates(problem, deadline, strong_low_mode)
    best = None
    for _, solution in candidates:
        if best is None or _low_solution_value(solution) > _low_solution_value(best):
            best = solution
    return best


def _low_beam_state_value(state):
    return (
        state["covered"] * 100000.0
        - state["total_score"]
        + LOW_BEAM_ACCEPT_WEIGHT * state["expected_accept"]
        - LOW_BEAM_RISK_WEIGHT * state["failure_risk"]
        - 3.0 * len(state["selected"])
    )


def _choose_low_anchor_task(problem, state):
    best_key = None
    best_task = None
    task_to_bundle_candidates = problem["task_to_bundle_candidates"]
    uncovered_mask = state["uncovered_mask"]
    for task_idx in _mask_to_indices(uncovered_mask):
        feasible_count = 0
        best_w = 0.0
        for bundle_mask, items in task_to_bundle_candidates.get(task_idx, []):
            if bundle_mask & ~uncovered_mask:
                continue
            for cand in items:
                courier_bit = 1 << cand[7]
                if state["used_courier_mask"] & courier_bit:
                    continue
                feasible_count += 1
                if cand[4] > best_w:
                    best_w = cand[4]
        if feasible_count == 0:
            return None
        key = (feasible_count, best_w)
        if best_key is None or key < best_key:
            best_key = key
            best_task = task_idx
    return best_task


def _run_low_failure_risk_beam_search(problem, deadline, strong_low_mode=False):
    full_task_mask = (1 << len(problem["task_names"])) - 1
    beam = [{
        "selected": [],
        "uncovered_mask": full_task_mask,
        "used_courier_mask": 0,
        "covered": 0,
        "expected_accept": 0.0,
        "total_score": 0.0,
        "failure_risk": 0.0,
    }]
    completed = []
    task_to_bundle_candidates = problem["task_to_bundle_candidates"]
    beam_size = LOW_BEAM_SIZE_STRONG if strong_low_mode else LOW_BEAM_SIZE
    stats = {
        "expanded_count": 0,
        "beam_completed_count": 0,
        "best_expected_accept": None,
        "best_total_score": None,
        "best_failure_risk": None,
    }

    while beam and time.perf_counter() < deadline:
        expanded = []
        for state in beam:
            if time.perf_counter() >= deadline:
                break
            if state["uncovered_mask"] == 0:
                completed.append(state)
                continue

            anchor_task = _choose_low_anchor_task(problem, state)
            if anchor_task is None:
                continue

            local_candidates = []
            for bundle_mask, items in task_to_bundle_candidates.get(anchor_task, []):
                if bundle_mask & ~state["uncovered_mask"]:
                    continue
                for cand in items:
                    courier_bit = 1 << cand[7]
                    if state["used_courier_mask"] & courier_bit:
                        continue
                    local_candidates.append(cand)

            local_candidates.sort(
                key=lambda cand: (
                    cand[3] / cand[8]
                    - LOW_BEAM_BETA * cand[4]
                    + LOW_BEAM_GAMMA * ((1.0 - cand[4]) ** LOW_RISK_ALPHA)
                    - LOW_BEAM_BUNDLE_BONUS * (cand[8] - 1),
                    cand[3],
                    -cand[4],
                    cand[5],
                )
            )

            for cand in local_candidates[:LOW_BEAM_PER_ANCHOR_TOPK]:
                new_state = {
                    "selected": state["selected"] + [cand],
                    "uncovered_mask": state["uncovered_mask"] & ~cand[6],
                    "used_courier_mask": state["used_courier_mask"] | (1 << cand[7]),
                    "covered": state["covered"] + cand[8],
                    "expected_accept": state["expected_accept"] + cand[4] * cand[8],
                    "total_score": state["total_score"] + cand[3],
                    "failure_risk": state["failure_risk"] + _assignment_failure_risk(cand),
                }
                expanded.append(new_state)
                stats["expanded_count"] += 1

        if completed:
            break
        if not expanded:
            break

        expanded.sort(key=_low_beam_state_value, reverse=True)
        beam = expanded[:beam_size]

    if not completed:
        completed = [state for state in beam if state["uncovered_mask"] == 0]
    stats["beam_completed_count"] = len(completed)
    _debug_scene_info("low_beam", list(stats.items()))
    if not completed:
        return None

    best_state = max(completed, key=_low_beam_state_value)
    stats["best_expected_accept"] = round(best_state["expected_accept"], 4)
    stats["best_total_score"] = round(best_state["total_score"], 4)
    stats["best_failure_risk"] = round(best_state["failure_risk"], 4)
    _debug_scene_info("low_beam_best", list(stats.items()))
    return _evaluate(best_state["selected"])


def _build_low_ilp_reduced_candidates(problem):
    keep_ids = set()

    for items in problem["bundle_frontier"].values():
        alpha_sorted = sorted(items, key=lambda c: (_alpha_rank_cost(c), c[3], -c[4], c[5]))
        for cand in alpha_sorted[:LOW_ILP_TOPK_PER_BUNDLE]:
            keep_ids.add(cand[5])
        if alpha_sorted:
            keep_ids.add(alpha_sorted[0][5])
        best_value = max(items, key=lambda c: (_low_candidate_value(c), c[4], -c[3]))
        keep_ids.add(best_value[5])
        best_w = max(items, key=lambda c: (c[4], -c[3], c[5]))
        keep_ids.add(best_w[5])

    task_count = len(problem["task_names"])
    task_to_candidates = [[] for _ in range(task_count)]
    for cand in problem["candidates"]:
        if cand[5] not in keep_ids:
            continue
        for task_idx in _mask_to_indices(cand[6]):
            task_to_candidates[task_idx].append(cand)

    refined_ids = set()
    for task_idx, items in enumerate(task_to_candidates):
        ranked = sorted(
            items,
            key=lambda c: (
                -_low_candidate_value(c) / max(1, c[8]),
                _alpha_rank_cost(c),
                c[3],
                c[5],
            ),
        )
        for cand in ranked[:LOW_ILP_TOPK_PER_TASK]:
            refined_ids.add(cand[5])

        singletons = [cand for cand in problem["bundle_frontier"].get(1 << task_idx, [])]
        singletons.sort(key=lambda c: (_alpha_rank_cost(c), c[3], -c[4], c[5]))
        for cand in singletons[:LOW_ILP_SINGLETON_GUARD_K]:
            refined_ids.add(cand[5])

    reduced = [cand for cand in problem["candidates"] if cand[5] in refined_ids]
    reduced.sort(key=lambda c: (_alpha_rank_cost(c), c[3], -c[4], c[5]))
    return reduced


def _build_low_ilp_task_index(reduced, task_count):
    task_to_ids = [[] for _ in range(task_count)]
    for local_id, cand in enumerate(reduced):
        for task_idx in _mask_to_indices(cand[6]):
            task_to_ids[task_idx].append(local_id)
    return task_to_ids


def _greedy_low_ilp_initial_solution(problem, reduced, task_to_ids):
    selected_ids = []
    used_courier_mask = 0
    uncovered_mask = (1 << len(problem["task_names"])) - 1

    while uncovered_mask:
        anchor_options = None
        anchor_list = []
        for task_idx in _mask_to_indices(uncovered_mask):
            feasible = []
            for cid in task_to_ids[task_idx]:
                cand = reduced[cid]
                courier_bit = 1 << cand[7]
                if used_courier_mask & courier_bit:
                    continue
                if cand[6] & ~uncovered_mask:
                    continue
                feasible.append(cid)
            if not feasible:
                continue
            if anchor_options is None or len(feasible) < anchor_options:
                anchor_options = len(feasible)
                anchor_list = feasible

        if anchor_options is None:
            break

        anchor_list.sort(
            key=lambda cid: (
                -_low_candidate_value(reduced[cid]),
                _alpha_rank_cost(reduced[cid]),
                reduced[cid][3],
                reduced[cid][5],
            )
        )
        chosen = reduced[anchor_list[0]]
        selected_ids.append(anchor_list[0])
        used_courier_mask |= 1 << chosen[7]
        uncovered_mask &= ~chosen[6]

    return [reduced[cid] for cid in selected_ids]


def _run_low_ilp_solver(problem, seed_solution, deadline):
    reduced = _build_low_ilp_reduced_candidates(problem)
    if not reduced:
        return seed_solution

    task_count = len(problem["task_names"])
    task_to_ids = _build_low_ilp_task_index(reduced, task_count)
    full_mask = (1 << task_count) - 1
    values = [_low_candidate_value(cand) for cand in reduced]
    unit_values = [values[idx] / max(1, reduced[idx][8]) for idx in range(len(reduced))]

    best_solution = seed_solution
    best_value = _low_solution_value(seed_solution)
    greedy_seed = _evaluate(_greedy_low_ilp_initial_solution(problem, reduced, task_to_ids))
    if _low_solution_value(greedy_seed) > best_value and greedy_seed["covered"] == best_solution["covered"]:
        best_solution = greedy_seed
        best_value = _low_solution_value(greedy_seed)

    stats = {
        "reduced_candidates": len(reduced),
        "nodes": 0,
        "pruned_bound": 0,
        "feasible_updates": 0,
    }

    def optimistic_bound(uncovered_mask, used_courier_mask, current_value):
        total = current_value
        for task_idx in _mask_to_indices(uncovered_mask):
            best_unit = None
            for cid in task_to_ids[task_idx]:
                cand = reduced[cid]
                courier_bit = 1 << cand[7]
                if used_courier_mask & courier_bit:
                    continue
                if cand[6] & ~uncovered_mask:
                    continue
                val = unit_values[cid]
                if best_unit is None or val > best_unit:
                    best_unit = val
            if best_unit is None:
                return None
            total += 100000.0 + best_unit
        return total

    def search(uncovered_mask, used_courier_mask, current_value, selected_local_ids):
        nonlocal best_solution, best_value
        if time.perf_counter() >= deadline:
            return
        stats["nodes"] += 1

        if uncovered_mask == 0:
            selected = [reduced[cid] for cid in selected_local_ids]
            proposal = _evaluate(selected)
            proposal_value = _low_solution_value(proposal)
            if proposal_value > best_value + 1e-9 or (
                abs(proposal_value - best_value) <= 1e-9 and _ranking_key(proposal) > _ranking_key(best_solution)
            ):
                best_solution = proposal
                best_value = proposal_value
                stats["feasible_updates"] += 1
            return

        ub = optimistic_bound(uncovered_mask, used_courier_mask, current_value)
        if ub is None or ub <= best_value + 1e-9:
            stats["pruned_bound"] += 1
            return

        anchor_options = None
        options = []
        for task_idx in _mask_to_indices(uncovered_mask):
            feasible = []
            for cid in task_to_ids[task_idx]:
                cand = reduced[cid]
                courier_bit = 1 << cand[7]
                if used_courier_mask & courier_bit:
                    continue
                if cand[6] & ~uncovered_mask:
                    continue
                feasible.append(cid)
            if not feasible:
                return
            if anchor_options is None or len(feasible) < anchor_options:
                anchor_options = len(feasible)
                options = feasible

        options.sort(
            key=lambda cid: (
                -values[cid],
                _alpha_rank_cost(reduced[cid]),
                reduced[cid][3],
                reduced[cid][5],
            )
        )
        for cid in options[:LOW_ILP_BRANCH_OPTION_LIMIT]:
            cand = reduced[cid]
            selected_local_ids.append(cid)
            search(
                uncovered_mask & ~cand[6],
                used_courier_mask | (1 << cand[7]),
                current_value + 100000.0 * cand[8] + values[cid],
                selected_local_ids,
            )
            selected_local_ids.pop()

    search(full_mask, 0, 0.0, [])
    _debug_scene_info("low_ilp", list(stats.items()))
    return best_solution


def _run_low_willingness_solver(problem, seed_solution, deadline, strong_low_mode=False):
    current = _refine_primary_couriers_for_low_willingness(problem, seed_solution)
    if ENABLE_LOW_PORTFOLIO_SEED:
        portfolio_seed = _run_low_willingness_portfolio(problem, deadline)
        if portfolio_seed is not None:
            if strong_low_mode:
                # 这个入口保留成可切换 ablation，方便后面确认 low portfolio 是否在破坏 base 结构。
                if (
                    _prefer_low_risk_solution_relaxed(portfolio_seed, current)
                    or _low_solution_value(portfolio_seed) > _low_solution_value(current)
                ):
                    current = portfolio_seed
            elif _prefer_low_solution(portfolio_seed, current):
                current = portfolio_seed
    if time.perf_counter() >= deadline:
        return current

    best = current
    if ENABLE_LOW_RISK_REPLACEMENT and time.perf_counter() < deadline:
        best = _run_low_risk_replacement_solver(problem, best, deadline, strong_low_mode)
    if ENABLE_LOW_ILP and time.perf_counter() < deadline:
        ilp_solution = _run_low_ilp_solver(problem, best, deadline)
        if strong_low_mode:
            if _prefer_low_risk_solution_relaxed(ilp_solution, best) or _low_solution_value(ilp_solution) > _low_solution_value(best):
                return ilp_solution
        elif _prefer_low_solution(ilp_solution, best):
            return ilp_solution
    return best


def _assignment_failure_risk(cand):
    return cand[8] * ((1.0 - cand[4]) ** LOW_RISK_ALPHA)


def _solution_failure_risk(solution):
    return sum(_assignment_failure_risk(cand) for cand in solution["selected"])


def _search_best_split_replacement(problem, base, used_couriers_without_base, budget_left):
    task_indices = _mask_to_indices(base[6])
    option_groups = []
    for task_idx in task_indices:
        options = []
        for cand in problem["single_task_options"][task_idx]:
            if cand[7] in used_couriers_without_base:
                continue
            options.append(cand)
            if len(options) >= LOW_SPLIT_SINGLE_TOPK:
                break
        if not options:
            return None
        option_groups.append(options)

    best = None
    old_risk = _assignment_failure_risk(base)

    def dfs(pos, local_used, chosen, total_score, total_risk):
        nonlocal best
        if pos == len(option_groups):
            score_delta = total_score - base[3]
            if score_delta > budget_left:
                return
            risk_gain = old_risk - total_risk
            utility = LOW_RISK_WEIGHT * risk_gain - score_delta
            if risk_gain <= 0.0 or utility <= 0.0:
                return
            record = (utility, risk_gain, score_delta, list(chosen), "split")
            if best is None or record[:3] > best[:3]:
                best = record
            return

        for cand in option_groups[pos]:
            if cand[7] in local_used:
                continue
            local_used.add(cand[7])
            chosen.append(cand)
            dfs(
                pos + 1,
                local_used,
                chosen,
                total_score + cand[3],
                total_risk + _assignment_failure_risk(cand),
            )
            chosen.pop()
            local_used.remove(cand[7])

    dfs(0, set(), [], 0.0, 0.0)
    return best


def _run_low_risk_replacement_solver(problem, solution, deadline, strong_low_mode=False):
    selected = list(solution["selected"])
    if strong_low_mode:
        budget_total = min(120.0, 0.06 * solution["total_score"])
    else:
        budget_total = min(60.0, 0.03 * solution["total_score"])
    extra_score_used = 0.0
    stats = {
        "strong_low_mode": strong_low_mode,
        "budget_total": round(budget_total, 4),
        "attempt_single": 0,
        "accept_single": 0,
        "attempt_split": 0,
        "accept_split": 0,
        "extra_score_used": 0.0,
        "risk_gain_total": 0.0,
        "score_delta_total": 0.0,
        "final_relaxed_accept": False,
        "fallback_to_original": False,
    }

    while time.perf_counter() < deadline and extra_score_used < budget_total:
        improved = False
        ranked_indices = list(range(len(selected)))
        ranked_indices.sort(
            key=lambda idx: (_assignment_failure_risk(selected[idx]), -selected[idx][8], selected[idx][3]),
            reverse=True,
        )

        for idx in ranked_indices:
            if time.perf_counter() >= deadline or extra_score_used >= budget_total:
                break

            base = selected[idx]
            budget_left = budget_total - extra_score_used
            used_without_base = set()
            for j, cand in enumerate(selected):
                if j != idx:
                    used_without_base.add(cand[7])

            best_swap = None
            if base[8] == 1 and base[4] < 0.72:
                stats["attempt_single"] += 1
                for cand in problem["single_task_options"][_mask_to_indices(base[6])[0]]:
                    if cand[7] == base[7] or cand[7] in used_without_base:
                        continue
                    willingness_gain = cand[4] - base[4]
                    score_delta = cand[3] - base[3]
                    if willingness_gain < LOW_SINGLE_W_GAIN:
                        continue
                    if score_delta > min(budget_left, LOW_SINGLE_SCORE_BASE + LOW_SINGLE_SCORE_GAIN * willingness_gain):
                        continue
                    old_risk = _assignment_failure_risk(base)
                    new_risk = _assignment_failure_risk(cand)
                    risk_gain = old_risk - new_risk
                    utility = LOW_RISK_WEIGHT * risk_gain - score_delta
                    if risk_gain <= 0.0 or utility <= 0.0:
                        continue
                    record = (utility, risk_gain, score_delta, [cand], "single")
                    if best_swap is None or record[:3] > best_swap[:3]:
                        best_swap = record
            elif base[8] > 1 and base[4] < 0.80:
                stats["attempt_split"] += 1
                best_swap = _search_best_split_replacement(problem, base, used_without_base, budget_left)

            if best_swap is None:
                continue

            _, risk_gain, score_delta, replacement, move_type = best_swap
            if base[8] == 1:
                selected[idx] = replacement[0]
            else:
                selected[idx:idx + 1] = replacement
            extra_score_used += max(0.0, score_delta)
            stats["extra_score_used"] = round(extra_score_used, 4)
            stats["risk_gain_total"] += risk_gain
            stats["score_delta_total"] += max(0.0, score_delta)
            if move_type == "single":
                stats["accept_single"] += 1
            else:
                stats["accept_split"] += 1
            improved = True
            break

        if not improved:
            break

    proposal = _evaluate(selected)
    if strong_low_mode:
        if _prefer_low_risk_solution_relaxed(proposal, solution):
            stats["final_relaxed_accept"] = True
            _debug_scene_info("low_replacement", list(stats.items()))
            return proposal
    elif _prefer_low_solution(proposal, solution):
        _debug_scene_info("low_replacement", list(stats.items()))
        return proposal
    stats["fallback_to_original"] = True
    _debug_scene_info("low_replacement", list(stats.items()))
    return solution


def _prefer_low_solution(candidate, baseline):
    if candidate["covered"] < baseline["covered"]:
        return False
    if _ranking_key(candidate) > _ranking_key(baseline):
        return True

    low_delta = _low_solution_value(candidate) - _low_solution_value(baseline)
    accept_gain = candidate["expected_accept"] - baseline["expected_accept"]
    score_delta = candidate["total_score"] - baseline["total_score"]
    risk_gain = _solution_failure_risk(baseline) - _solution_failure_risk(candidate)

    # low 场景允许“小幅加分成本换更高成功率”，
    # 但仍要求收益足够明显，避免为了 willingness 过度抬高 score。
    if low_delta <= 0.0:
        return False
    if accept_gain < 0.08 and risk_gain < 0.12:
        return False
    if score_delta > 28.0 + 72.0 * max(accept_gain, risk_gain):
        return False
    return True


def _prefer_low_risk_solution_relaxed(candidate, baseline):
    if candidate["covered"] < baseline["covered"]:
        return False
    if _ranking_key(candidate) > _ranking_key(baseline):
        return True

    accept_gain = candidate["expected_accept"] - baseline["expected_accept"]
    score_delta = candidate["total_score"] - baseline["total_score"]
    risk_gain = _solution_failure_risk(baseline) - _solution_failure_risk(candidate)

    # strong low 场景里更关心失败风险下降，不再强制要求 low_delta 为正。
    if risk_gain <= 0.0:
        return False
    if accept_gain < 0.03 and risk_gain < 0.06:
        return False
    if score_delta > 45.0 + 120.0 * max(accept_gain, risk_gain):
        return False
    return True


def _compute_courier_opportunity_cost(problem):
    courier_count = len(problem["courier_names"])
    values = [0.0 for _ in range(courier_count)]

    for cand in problem["candidates"]:
        score_per_task = cand[3] / max(1, cand[8])
        value = 8.0 * cand[8] + 20.0 * cand[4] - 0.15 * score_per_task
        if value > 0.0:
            values[cand[7]] += value

    max_value = max(values) if values else 1.0
    if max_value <= 1e-9:
        return [0.0 for _ in range(courier_count)]
    return [v / max_value for v in values]


def _choose_scarce_destroy_indices(problem, selected, rng, courier_opportunity):
    scored = []
    task_to_bundle_candidates = problem["task_to_bundle_candidates"]
    for idx, cand in enumerate(selected):
        score_per_task = cand[3] / max(1, cand[8])
        badness = score_per_task - 32.0 * cand[4] + 4.0 / max(1, cand[8])
        if cand[8] == 1:
            badness += 3.0
        badness += 8.0 * courier_opportunity[cand[7]] / max(1, cand[8])
        support = 0.0
        for task_idx in _mask_to_indices(cand[6]):
            support += len(task_to_bundle_candidates.get(task_idx, []))
        badness += min(4.0, 0.10 * support / max(1, cand[8]))
        scored.append((badness, idx))
    scored.sort(reverse=True)

    remove_count = min(len(selected), rng.choice(SCARCE_DESTROY_CHOICES))
    pool = [idx for _, idx in scored[: max(remove_count + 2, min(len(scored), 8))]]
    chosen = []
    while pool and len(chosen) < remove_count:
        pick = pool.pop(rng.randrange(len(pool)))
        if pick not in chosen:
            chosen.append(pick)
    chosen.sort()
    return chosen


def _scarce_partial_value(covered, expected_accept, total_score, selected_count):
    bundle_gain = covered - selected_count
    return (
        covered * 100000.0
        - total_score
        + expected_accept * 45.0
        - selected_count * 5.0
        + bundle_gain * 3.0
    )


def _strictly_better_scarce(proposal, base):
    if proposal["covered"] < base["covered"]:
        return False

    score_gain = base["total_score"] - proposal["total_score"]
    accept_delta = proposal["expected_accept"] - base["expected_accept"]
    selected_delta = len(base["selected"]) - len(proposal["selected"])

    if _ranking_key(proposal) <= _ranking_key(base):
        return False
    # 这里小幅放宽 score_gain 门槛，尝试释放少量原本被 strict accept 挡住的轻微优质 proposal。
    if score_gain >= 5.0 and accept_delta >= -0.05:
        return True
    if selected_delta >= 1 and score_gain >= -5.0 and accept_delta >= 0.0:
        return True
    return False


def _scarce_repair_key(cand, beta, courier_opportunity):
    score_per_task = cand[3] / max(1, cand[8])
    courier_cost = courier_opportunity[cand[7]]
    return (
        score_per_task
        - beta * cand[4]
        - 8.0 * (cand[8] - 1)
        + 5.0 * courier_cost / max(1, cand[8]),
        cand[3],
        -cand[4],
        cand[5],
    )


def _beam_repair_subset(problem, subset_task_mask, allowed_courier_mask, beta, deadline, courier_opportunity=None):
    if subset_task_mask == 0:
        return [[]]
    if subset_task_mask.bit_count() > SCARCE_MAX_REPAIR_TASK_COUNT:
        return []

    task_to_bundle_candidates = problem["task_to_bundle_candidates"]
    target_cover = subset_task_mask.bit_count()
    initial_state = {
        "selected": [],
        "uncovered": subset_task_mask,
        "used_courier_mask": 0,
        "covered": 0,
        "expected_accept": 0.0,
        "total_score": 0.0,
    }
    states = [initial_state]
    completed = []

    while states and not completed:
        if time.perf_counter() >= deadline:
            return []
        expanded = []
        for state in states:
            if time.perf_counter() >= deadline:
                return []
            uncovered = state["uncovered"]
            if uncovered == 0:
                completed.append(state)
                continue

            task_indices = _mask_to_indices(uncovered)
            anchor_task = None
            anchor_options = None
            for task_idx in task_indices:
                options = []
                for bundle_mask, items in task_to_bundle_candidates.get(task_idx, []):
                    if bundle_mask & ~uncovered:
                        continue
                    for cand in items:
                        courier_bit = 1 << cand[7]
                        if not (allowed_courier_mask & courier_bit):
                            continue
                        if state["used_courier_mask"] & courier_bit:
                            continue
                        options.append(cand)
                        break
                if anchor_options is None or len(options) < anchor_options:
                    anchor_task = task_idx
                    anchor_options = len(options)

            if anchor_task is None:
                continue

            local_candidates = []
            for bundle_mask, items in task_to_bundle_candidates.get(anchor_task, []):
                if bundle_mask & ~uncovered:
                    continue
                filtered = []
                for cand in items:
                    courier_bit = 1 << cand[7]
                    if not (allowed_courier_mask & courier_bit):
                        continue
                    if state["used_courier_mask"] & courier_bit:
                        continue
                    filtered.append(cand)
                if not filtered:
                    continue
                if courier_opportunity is None:
                    filtered.sort(key=lambda cand: _bundle_sort_key(cand, beta))
                else:
                    filtered.sort(key=lambda cand: _scarce_repair_key(cand, beta, courier_opportunity))
                local_candidates.extend(filtered[:2])

            if courier_opportunity is None:
                local_candidates.sort(key=lambda cand: _bundle_sort_key(cand, beta))
            else:
                local_candidates.sort(key=lambda cand: _scarce_repair_key(cand, beta, courier_opportunity))
            for cand in local_candidates[:SCARCE_BEAM_PER_ANCHOR]:
                courier_bit = 1 << cand[7]
                next_state = {
                    "selected": state["selected"] + [cand],
                    "uncovered": uncovered ^ cand[6],
                    "used_courier_mask": state["used_courier_mask"] | courier_bit,
                    "covered": state["covered"] + cand[8],
                    "expected_accept": state["expected_accept"] + cand[4] * cand[8],
                    "total_score": state["total_score"] + cand[3],
                }
                expanded.append(next_state)

        if completed:
            break
        if not expanded:
            return []

        expanded.sort(
            key=lambda state: _scarce_partial_value(
                state["covered"],
                state["expected_accept"],
                state["total_score"],
                len(state["selected"]),
            ),
            reverse=True,
        )
        states = expanded[:SCARCE_BEAM_SIZE]

    completed.sort(
        key=lambda state: _scarce_partial_value(
            target_cover,
            state["expected_accept"],
            state["total_score"],
            len(state["selected"]),
        ),
        reverse=True,
    )
    results = []
    for state in completed[:3]:
        if state["uncovered"] == 0:
            results.append(state["selected"])
    return results


def _build_scarce_exact_candidate_pool(problem, subset_task_mask, allowed_courier_mask, beta, courier_opportunity):
    candidate_pool = []
    for bundle_mask, items in problem["bundle_frontier"].items():
        if bundle_mask & ~subset_task_mask:
            continue

        local_items = []
        for cand in items:
            courier_bit = 1 << cand[7]
            if not (allowed_courier_mask & courier_bit):
                continue
            local_items.append(cand)

        if not local_items:
            continue

        local_items.sort(key=lambda cand: _scarce_repair_key(cand, beta, courier_opportunity))
        candidate_pool.extend(local_items[:SCARCE_BEAM_PER_ANCHOR])

    candidate_pool.sort(key=lambda cand: _scarce_repair_key(cand, beta, courier_opportunity))
    return candidate_pool[:SCARCE_EXACT_MAX_CANDIDATES]


def _scarce_exact_local_value(selected):
    covered = 0
    expected_accept = 0.0
    total_score = 0.0

    for cand in selected:
        covered += cand[8]
        expected_accept += cand[4] * cand[8]
        total_score += cand[3]

    selected_count = len(selected)
    bundle_gain = covered - selected_count
    return (
        covered * 100000.0
        - total_score
        + expected_accept * 45.0
        + SCARCE_EXACT_BUNDLE_GAIN_WEIGHT * bundle_gain
        - SCARCE_EXACT_SELECTED_PENALTY * selected_count
    )


def _exact_set_packing_repair_subset(
    problem,
    subset_task_mask,
    allowed_courier_mask,
    beta,
    deadline,
    courier_opportunity,
):
    if subset_task_mask == 0:
        return [[]]
    if subset_task_mask.bit_count() > SCARCE_EXACT_MAX_TASK_COUNT:
        return []

    candidate_pool = _build_scarce_exact_candidate_pool(
        problem,
        subset_task_mask,
        allowed_courier_mask,
        beta,
        courier_opportunity,
    )
    if not candidate_pool:
        return []

    task_to_candidates = {}
    for idx, cand in enumerate(candidate_pool):
        for task_idx in _mask_to_indices(cand[6]):
            task_to_candidates.setdefault(task_idx, []).append(idx)

    results = []
    node_count = 0

    def choose_anchor(uncovered_mask, used_courier_mask):
        best_task = None
        best_count = None
        best_options = None

        for task_idx in _mask_to_indices(uncovered_mask):
            feasible = []
            task_bit = 1 << task_idx
            for cid in task_to_candidates.get(task_idx, []):
                cand = candidate_pool[cid]
                if not (cand[6] & task_bit):
                    continue
                if cand[6] & ~uncovered_mask:
                    continue
                if used_courier_mask & (1 << cand[7]):
                    continue
                feasible.append(cid)

            count = len(feasible)
            if best_count is None or count < best_count:
                best_task = task_idx
                best_count = count
                best_options = feasible

        return best_task, best_count, best_options

    def dfs(uncovered_mask, used_courier_mask, chosen):
        nonlocal node_count
        if time.perf_counter() >= deadline:
            return
        node_count += 1
        if node_count > SCARCE_EXACT_NODE_LIMIT:
            return

        if uncovered_mask == 0:
            results.append(( _scarce_exact_local_value(chosen), list(chosen) ))
            return

        _, option_count, options = choose_anchor(uncovered_mask, used_courier_mask)
        if option_count is None or option_count == 0:
            return

        for cid in options:
            cand = candidate_pool[cid]
            courier_bit = 1 << cand[7]
            dfs(
                uncovered_mask & ~cand[6],
                used_courier_mask | courier_bit,
                chosen + [cand],
            )
            if time.perf_counter() >= deadline or node_count > SCARCE_EXACT_NODE_LIMIT:
                return

    dfs(subset_task_mask, 0, [])

    results.sort(key=lambda item: item[0], reverse=True)
    final_results = []
    for _, chosen in results[:SCARCE_EXACT_TOP_RESULTS]:
        final_results.append(chosen)
    return final_results


def _run_scarce_courier_solver(problem, seed_solution, deadline):
    rng = random.Random(20260526)
    current = _evaluate(seed_solution["selected"])
    best = current
    stale_rounds = 0
    courier_opportunity = _compute_courier_opportunity_cost(problem)
    stats = {
        "destroy_rounds": 0,
        "repair_options_count": 0,
        "proposal_best_count": 0,
        "strict_accept_count": 0,
        "current_update_count": 0,
        "best_update_count": 0,
        "repair_mode": "exact" if SCARCE_USE_EXACT_REPAIR else "beam",
    }

    while time.perf_counter() < deadline:
        selected = current["selected"]
        if len(selected) < 4:
            break

        removed_indices = _choose_scarce_destroy_indices(problem, selected, rng, courier_opportunity)
        stats["destroy_rounds"] += 1
        remaining = []
        subset_task_mask = 0
        allowed_courier_mask = 0
        used_remaining_couriers = 0

        for idx, cand in enumerate(selected):
            if idx in removed_indices:
                subset_task_mask |= cand[6]
                allowed_courier_mask |= 1 << cand[7]
            else:
                remaining.append(cand)
                used_remaining_couriers |= 1 << cand[7]

        for courier_idx in range(len(problem["courier_names"])):
            courier_bit = 1 << courier_idx
            if not (used_remaining_couriers & courier_bit):
                allowed_courier_mask |= courier_bit

        if subset_task_mask.bit_count() > SCARCE_MAX_REPAIR_TASK_COUNT:
            stale_rounds += 1
            continue

        proposal_best = None
        for beta in (35.0, 45.0, 55.0):
            if time.perf_counter() >= deadline:
                break
            if SCARCE_USE_EXACT_REPAIR:
                repair_options = _exact_set_packing_repair_subset(
                    problem,
                    subset_task_mask,
                    allowed_courier_mask,
                    beta,
                    deadline,
                    courier_opportunity,
                )
            elif ENABLE_SCARCE_BEAM_REPAIR:
                repair_options = _beam_repair_subset(
                    problem,
                    subset_task_mask,
                    allowed_courier_mask,
                    beta,
                    deadline,
                    courier_opportunity,
                )
            else:
                repair_options = []
            stats["repair_options_count"] += len(repair_options)
            for repaired in repair_options:
                proposal = _evaluate(remaining + repaired)
                if proposal_best is None:
                    proposal_best = proposal
                    continue
                if _ranking_key(proposal) > _ranking_key(proposal_best):
                    proposal_best = proposal
                    continue
                if (
                    _ranking_key(proposal) == _ranking_key(proposal_best)
                    and _scarce_solution_value(proposal) > _scarce_solution_value(proposal_best)
                ):
                    proposal_best = proposal

        if proposal_best is None:
            stale_rounds += 1
            if stale_rounds >= 3:
                current = best
                stale_rounds = 0
            continue

        stats["proposal_best_count"] += 1
        improved = False
        if _strictly_better_scarce(proposal_best, best):
            best = proposal_best
            current = proposal_best
            stale_rounds = 0
            improved = True
            stats["strict_accept_count"] += 1
            stats["best_update_count"] += 1
            stats["current_update_count"] += 1
        elif (
            proposal_best["covered"] == current["covered"]
            and _scarce_solution_value(proposal_best) > _scarce_solution_value(current) + 12.0
        ):
            current = proposal_best
            improved = True
            stats["current_update_count"] += 1

        if not improved:
            stale_rounds += 1
            if stale_rounds >= 3:
                current = best
                stale_rounds = 0

    _debug_scene_info("scarce_stats", list(stats.items()))
    return best


def _select_best_solution(candidates):
    best = None
    for _, solution in candidates:
        if solution is None:
            continue
        if best is None or _ranking_key(solution) > _ranking_key(best):
            best = solution
    return best


def _evaluate_assignments_with_backups(problem, assignments):
    candidate_lookup = problem["candidate_lookup"]
    covered = 0
    expected_accept = 0.0
    sequential_expected_score = 0.0
    total_couriers = 0

    for assignment in assignments:
        task_cnt = assignment["task_cnt"]
        fail_prob = 1.0
        bundle_accept = 0.0

        for courier_id in assignment["courier_ids"]:
            score, willingness = candidate_lookup[(assignment["task_str"], courier_id)]
            sequential_expected_score += fail_prob * willingness * score
            bundle_accept += fail_prob * willingness
            fail_prob *= (1.0 - willingness)

        covered += task_cnt
        expected_accept += task_cnt * bundle_accept
        total_couriers += len(assignment["courier_ids"])

    adjusted_cost = sequential_expected_score - PROXY_WILLINGNESS_WEIGHT * expected_accept
    return {
        "covered": covered,
        "expected_accept": expected_accept,
        "sequential_expected_score": sequential_expected_score,
        "adjusted_cost": adjusted_cost,
        "assignment_count": len(assignments),
        "total_couriers": total_couriers,
    }


def _final_output_key(final_metrics):
    return (
        final_metrics["covered"],
        -final_metrics["adjusted_cost"],
        final_metrics["expected_accept"],
        -final_metrics["sequential_expected_score"],
        -final_metrics["assignment_count"],
        -final_metrics["total_couriers"],
    )


def _select_best_output_candidate(problem, candidates):
    best_entry = None
    all_entries = []

    for name, solution in candidates:
        if solution is None:
            continue
        assignments = _augment_with_backup_couriers(problem, solution)
        final_metrics = _evaluate_assignments_with_backups(problem, assignments)
        entry = {
            "name": name,
            "solution": solution,
            "assignments": assignments,
            "final_metrics": final_metrics,
        }
        all_entries.append(entry)
        if best_entry is None or _final_output_key(final_metrics) > _final_output_key(best_entry["final_metrics"]):
            best_entry = entry

    if best_entry is None:
        return None, []

    # low_matching 已经被证明更可能带来真实收益，这里对 matching 和 beam
    # 使用略有区别的后验接受门槛，优先放大已经验证有效的 low 优势。
    for entry in all_entries:
        if not (entry["name"].startswith("low_matching") or entry["name"] == "low_beam"):
            continue
        fm = entry["final_metrics"]
        bm = best_entry["final_metrics"]
        accept_gain = fm["expected_accept"] - bm["expected_accept"]
        adjusted_gain = bm["adjusted_cost"] - fm["adjusted_cost"]
        if entry["name"].startswith("low_matching"):
            accept_threshold = 0.08
            adjusted_threshold = -8.0
        else:
            accept_threshold = 0.10
            adjusted_threshold = -8.0
        if (
            fm["covered"] == bm["covered"]
            and accept_gain >= accept_threshold
            and adjusted_gain >= adjusted_threshold
        ):
            best_entry = entry
            break

    return best_entry, all_entries


def _debug_scene_info(label, values):
    if not DEBUG_SCENE:
        return
    parts = [label]
    for key, value in values:
        parts.append(f"{key}={value}")
    print(" ".join(parts))


def _validate_and_format(assignments):
    used_task_mask = 0
    used_courier_mask = 0
    result = []

    for assignment in assignments:
        if used_task_mask & assignment["task_mask"]:
            continue

        courier_ids = []
        for courier_id, courier_idx in zip(assignment["courier_ids"], assignment["courier_indices"]):
            courier_bit = 1 << courier_idx
            if used_courier_mask & courier_bit:
                continue
            used_courier_mask |= courier_bit
            courier_ids.append(courier_id)

        if not courier_ids:
            continue

        used_task_mask |= assignment["task_mask"]
        result.append((assignment["task_str"], courier_ids))

    return result


def solve(input_text: str) -> list:
    problem = _parse_input(input_text)
    if not problem["candidates"]:
        return []

    start = time.perf_counter()
    deadline = start + TIME_LIMIT_SEC - FINAL_RESERVE_SEC
    env = _estimate_environment(problem)
    static_low_mode = (
        env["avg_best_single_w"] < LOW_STATIC_AVG_BEST_SINGLE_THRESHOLD
        or env["avg_candidate_w"] < LOW_STATIC_AVG_CANDIDATE_THRESHOLD
        or env.get("low_candidate_ratio", 0.0) >= LOW_CANDIDATE_RATIO_THRESHOLD
        or env.get("low_courier_ratio", 0.0) >= LOW_COURIER_RATIO_THRESHOLD
        or env.get("low_task_ratio", 0.0) >= LOW_TASK_RATIO_THRESHOLD
    )
    static_scarce_mode = env["courier_task_ratio"] < 1.20
    special_mode = static_low_mode or static_scarce_mode

    base_solution = _run_base_solver(problem, start, deadline, special_mode)
    candidate_solutions = [("base", base_solution)]

    env2, dynamic_low_mode, dynamic_scarce_mode, avg_primary_w = _get_scene_flags(problem, base_solution)
    force_low_debug = env["avg_best_single_w"] < 0.60
    low_mode = static_low_mode or dynamic_low_mode
    scarce_mode = static_scarce_mode or dynamic_scarce_mode
    strong_low_mode = _is_strong_low_mode(env2, avg_primary_w)

    _debug_scene_info(
        "scene",
        [
            ("avg_best_single_w", round(env["avg_best_single_w"], 4)),
            ("avg_candidate_w", round(env.get("avg_candidate_w", 0.0), 4)),
            ("low_candidate_ratio", round(env.get("low_candidate_ratio", 0.0), 4)),
            ("low_courier_ratio", round(env.get("low_courier_ratio", 0.0), 4)),
            ("low_task_ratio", round(env.get("low_task_ratio", 0.0), 4)),
            ("courier_task_ratio", round(env["courier_task_ratio"], 4)),
            ("static_low", static_low_mode),
            ("static_scarce", static_scarce_mode),
            ("base_avg_primary_w", None if avg_primary_w is None else round(avg_primary_w, 4)),
            ("dynamic_low", dynamic_low_mode),
            ("dynamic_scarce", dynamic_scarce_mode),
            ("strong_low", strong_low_mode),
            ("force_low_debug", force_low_debug),
            ("final_low", low_mode),
            ("final_scarce", scarce_mode),
        ],
    )

    scarce_solution = None
    if scarce_mode and time.perf_counter() < deadline:
        try:
            scarce_deadline = min(
                deadline,
                start + TIME_LIMIT_SEC * SCARCE_BRANCH_RATIO,
                time.perf_counter() + SCARCE_BRANCH_BUDGET_SEC,
            )
            scarce_solution = _run_scarce_courier_solver(
                problem,
                base_solution,
                scarce_deadline,
            )
            if scarce_solution is not None:
                candidate_solutions.append(("scarce", scarce_solution))
        except Exception:
            scarce_solution = None

    if (low_mode or force_low_debug) and time.perf_counter() < deadline:
        try:
            low_branch_deadline = deadline
            low_matching_solution = None
            low_beam_solution = None
            if ENABLE_LOW_MATCHING:
                matching_budget = LOW_MATCHING_TIME_BUDGET_STRONG if strong_low_mode else LOW_MATCHING_TIME_BUDGET
                matching_deadline = min(low_branch_deadline, time.perf_counter() + matching_budget)
                low_matching_candidates = _run_low_aware_matching_candidates(
                    problem,
                    matching_deadline,
                    strong_low_mode=strong_low_mode,
                )
                if low_matching_candidates:
                    low_matching_solution = low_matching_candidates[0][1]
                for name, solution in low_matching_candidates:
                    if solution is not None and solution["covered"] == len(problem["task_names"]):
                        candidate_solutions.append((name, solution))

            if ENABLE_LOW_BEAM and time.perf_counter() < deadline:
                beam_budget = LOW_BEAM_TIME_BUDGET_STRONG if strong_low_mode else LOW_BEAM_TIME_BUDGET
                beam_deadline = min(low_branch_deadline, time.perf_counter() + beam_budget)
                low_beam_solution = _run_low_failure_risk_beam_search(
                    problem,
                    beam_deadline,
                    strong_low_mode=strong_low_mode,
                )
                if low_beam_solution is not None and low_beam_solution["covered"] == len(problem["task_names"]):
                    candidate_solutions.append(("low_beam", low_beam_solution))

            _debug_scene_info(
                "low_branch_result",
                [
                    ("matching_none", low_matching_solution is None),
                    ("matching_covered", None if low_matching_solution is None else low_matching_solution["covered"]),
                    ("beam_none", low_beam_solution is None),
                    ("beam_covered", None if low_beam_solution is None else low_beam_solution["covered"]),
                    ("task_count", len(problem["task_names"])),
                ],
            )

            if (ENABLE_LOW_RISK_REPLACEMENT or ENABLE_LOW_ILP) and time.perf_counter() < deadline:
                low_solution = _run_low_willingness_solver(
                    problem,
                    base_solution,
                    deadline,
                    strong_low_mode=strong_low_mode,
                )
                if low_solution is not None:
                    candidate_solutions.append(("low_replacement", low_solution))
        except Exception:
            pass

    _debug_scene_info(
        "candidates",
        [
            (
                name,
                (
                    sol["covered"],
                    round(sol["expected_accept"], 4),
                    round(sol["total_score"], 4),
                    _ranking_key(sol),
                ),
            )
            for name, sol in candidate_solutions
            if sol is not None
        ],
    )

    best_entry, final_entries = _select_best_output_candidate(problem, candidate_solutions)
    if best_entry is None:
        return []

    _debug_scene_info(
        "final_metrics",
        [
            (
                entry["name"],
                (
                    entry["final_metrics"]["covered"],
                    round(entry["final_metrics"]["expected_accept"], 4),
                    round(entry["final_metrics"]["adjusted_cost"], 4),
                ),
            )
            for entry in final_entries
        ],
    )
    _debug_scene_info(
        "final_choice",
        [
            ("name", best_entry["name"]),
            ("covered", best_entry["final_metrics"]["covered"]),
            ("expected_accept", round(best_entry["final_metrics"]["expected_accept"], 4)),
            ("adjusted_cost", round(best_entry["final_metrics"]["adjusted_cost"], 4)),
        ],
    )
    return _validate_and_format(best_entry["assignments"])


if __name__ == "__main__":
    import sys

    print(solve(sys.stdin.read()))
