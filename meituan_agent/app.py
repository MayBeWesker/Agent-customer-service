import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agent.react_agent import MeituanDispatchAgent


def main():
    parser = argparse.ArgumentParser(description="Meituan dispatch agent")
    parser.add_argument(
        "query",
        nargs="?",
        default="请使用 sa 求解 example/large_seed301.txt，并输出对应的选择",
        help="Agent query",
    )
    parser.add_argument("--case", dest="case_path", help="Case file path")
    parser.add_argument("--solver", choices=["sa", "baseline"], help="Solver name")
    parser.add_argument("--limit", type=int, default=20, help="Max displayed selections")
    args = parser.parse_args()

    agent = MeituanDispatchAgent()
    print(
        agent.execute(
            query=args.query,
            case_path=args.case_path,
            solver_name=args.solver,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
