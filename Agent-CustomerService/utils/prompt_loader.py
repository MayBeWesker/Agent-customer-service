from logger_handler import logger

from config_handler import prompts_config
from path_tool import get_abs_path

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
        print(system_prompt_path)
    except KeyError as e:
        logger.error(f"[load_system_prompts] can not find main_prompt_path in yaml")
        raise e
    try:
        return open(system_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_system_prompts], analysing system prompts error, {str(e)}")
        raise e
    
def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[rag_prompt_path] can not find rag_prompt_path in yaml")
        raise e
    try:
        return open(rag_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[rag_prompt_path], analysing rag prompts error, {str(e)}")
        raise e
    

def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[report_prompt_path] can not find report_prompt_path in yaml")
        raise e
    try:
        return open(report_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[report_prompt_path], analysing report prompts error, {str(e)}")
        raise e
    
if __name__ == "__main__":
    print(load_report_prompts())