from utils.path_tool import get_abs_path


def load_system_prompt() -> str:
    prompt_path = get_abs_path("prompts/main_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

