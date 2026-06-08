# meituan_agent

这是一个基于大模型工具调用的调度求解 Agent 项目。

它的目标是：让用户通过自然语言提问，调用内部的调度算法，对输入的 case 数据进行求解，并输出任务分配选择结果。

## 项目结构

- `app.py`
  命令行入口。适合直接在终端里调用 Agent。

- `web_app.py`
  网页入口。基于 Streamlit，适合通过聊天式界面使用 Agent。

- `agent/react_agent.py`
  Agent 主体。负责接收用户问题、调用大模型、选择工具并返回最终结果。

- `agent/tools/agent_tools.py`
  工具层。封装了 case 读取、solver 调用、结果格式化、case 信息查看等能力，供 Agent 调用。

- `model/factory.py`
  大模型初始化入口。当前使用 `ChatTongyi`，API Key 通过环境变量读取，不写入代码。

- `prompts/main_prompt.txt`
  Agent 的系统提示词，约束大模型在什么情况下调用工具，以及如何回答用户。

- `solver/solver_loader.py`
  solver 动态加载器。负责加载 `example` 目录中的算法实现。

- `example/`
  算法示例与测试样例目录。
  - `25号-15-sa.py`：主算法，模拟退火增强版 solver
  - `example_solution.py`：贪心 baseline solver
  - `large_seed301.txt`：示例输入数据

- `utils/`
  公共工具模块，例如路径处理、提示词加载等。

## 运行原理

项目整体运行流程如下：

1. 用户通过命令行或网页输入自然语言问题。
2. `react_agent.py` 创建的大模型 Agent 读取系统提示词，并理解用户意图。
3. 当用户要求“求解、输出选择结果、比较算法、分析 case”等内容时，Agent 会调用 `agent_tools.py` 中的工具。
4. 工具层会读取 case 数据，并通过 `solver_loader.py` 动态加载指定 solver。
5. solver 返回标准格式的调度结果：
   `[(task_id_list_str, [courier_id, ...]), ...]`
6. Agent 再将结果整理为用户可读的文本输出。

也就是说，这个项目的核心不是把规则写死在代码里，而是：

- 大模型负责理解问题与决定调用哪个工具
- 工具负责执行具体计算
- solver 负责给出真实调度结果

## 环境要求

建议使用你当前已有的 Conda 环境 `py10Agent`。

项目依赖至少包括：

- `streamlit`
- `langchain`
- `langchain-community`
- `dashscope`

并且需要提前在环境变量中配置好大模型相关的 API Key。

## 如何运行

### 1. 命令行运行

进入项目根目录后执行：

```bash
conda activate py10Agent
python /Volumes/Program/projects/RAG-Agent-projects/meituan_agent/app.py --solver sa --case /Volumes/Program/projects/RAG-Agent-projects/meituan_agent/example/large_seed301.txt --limit 10
```

示例输出内容包括：

- 使用的 solver
- 使用的 case 文件
- 最终 bundle 数量
- 覆盖任务数量
- 前若干条任务分配选择结果

### 2. 网页运行

执行：

```bash
conda activate py10Agent
streamlit run /Volumes/Program/projects/RAG-Agent-projects/meituan_agent/web_app.py
```

浏览器打开后，你可以：

- 直接输入自然语言问题
- 在左侧选择 `auto / sa / baseline`
- 上传自己的 `.txt` case 文件
- 设置输出结果展示条数

## 如何使用

### 常见提问方式

你可以直接输入类似下面的问题：

- `请使用 sa 求解当前 case，并输出前 10 条选择`
- `请比较 baseline 和 sa 在当前 case 上的结果差异`
- `请先分析一下当前 case 的规模，再给我求解结果`
- `请重新展示刚才的调度结果`

### 关于 case 数据

如果你没有上传文件，项目默认使用：

`example/large_seed301.txt`

如果你上传了新的 `.txt` 文件，Agent 会优先使用上传的内容进行求解。

### 关于 solver

- `sa`
  主算法，通常结果更强，适合正式求解。

- `baseline`
  贪心基线算法，适合快速测试和做结果对照。

如果你在网页中选择 `auto`，或者在自然语言里没有明确指定 solver，系统默认优先使用 `sa`。

## 当前输出形式

当前项目主要输出文本格式的调度结果，例如：

```text
solver: sa
case: example/large_seed301.txt
bundle_count: 39
covered_task_count: 40
selections:
1. T0033,T0034 -> C039, C062
2. T0000 -> C067, C044
...
```

## 后续可扩展方向

当前项目已经具备可运行的 Agent 能力。后续如果需要，还可以继续扩展：

- 将结果展示为表格
- 支持结果导出
- 增加更多 solver
- 增加更多 case 分析工具
- 增加调度结果解释能力
