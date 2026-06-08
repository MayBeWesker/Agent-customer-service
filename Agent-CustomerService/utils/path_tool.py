import os

def get_project_root() -> str:
    """
    获取工程目录下面所在的根目录
    """
    current_file = os.path.abspath(__file__)

    current_dir = os.path.dirname(current_file)

    projecy_root = os.path.dirname(current_dir)
    return projecy_root

def get_abs_path(relative_path:str)->str:
    """
    传入相对路径，得到绝对路径
    """

    project_root = get_project_root()
    return os.path.join(project_root, relative_path)

if __name__ == '__main__':
    print(get_abs_path("config/config.txt"))