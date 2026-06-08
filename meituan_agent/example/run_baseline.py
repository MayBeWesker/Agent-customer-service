from solver import solve


def main():
    with open("large_seed301.txt", "r", encoding="utf-8") as f:
        input_text = f.read()

    result = solve(input_text)

    lookup = {}
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    for line in lines[start:]:
        task_id_list_str, courier_id, score_str, willingness_str = line.split("\t")
        lookup[(task_id_list_str, courier_id)] = (float(score_str), float(willingness_str))

    used_tasks = set()
    used_couriers = set()
    covered = 0
    expected_accept = 0.0
    total_willingness = 0.0
    avg_couriers_per_task = 0.0
    sequential_expected_score = 0.0
    valid = True

    for task_id_list_str, couriers in result:
        task_ids = [task_id for task_id in task_id_list_str.split(",") if task_id]
        if any(task_id in used_tasks for task_id in task_ids):
            valid = False
        local_seen = set()
        fail_prob = 1.0
        bundle_accept = 0.0

        for courier_id in couriers:
            if courier_id in used_couriers or courier_id in local_seen:
                valid = False
                continue
            local_seen.add(courier_id)
            used_couriers.add(courier_id)

            score, willingness = lookup[(task_id_list_str, courier_id)]
            sequential_expected_score += fail_prob * willingness * score
            bundle_accept += fail_prob * willingness
            total_willingness += willingness
            fail_prob *= (1.0 - willingness)

        used_tasks.update(task_ids)
        covered += len(task_ids)
        expected_accept += len(task_ids) * bundle_accept
        avg_couriers_per_task += len(local_seen)

    if result:
        avg_couriers_per_task /= float(len(result))

    print("valid:", valid)
    print("assignments:", len(result))
    print("covered:", covered)
    print("expected_accept:", round(expected_accept, 4))
    print("sequential_expected_score:", round(sequential_expected_score, 4))
    print("total_willingness:", round(total_willingness, 4))
    print("avg_couriers_per_task:", round(avg_couriers_per_task, 4))
    print("adjusted_cost_45:", round(sequential_expected_score - 45.0 * expected_accept, 4))
    print("前 10 条结果:")
    for item in result[:10]:
        print(item)


if __name__ == "__main__":
    main()
