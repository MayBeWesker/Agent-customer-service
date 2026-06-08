# AI用药伴侣 Demo

本目录是基于当前仓库 RAG 项目思路重构的“AI用药伴侣”演示版，面向 55-75 岁中老年人，覆盖以下核心能力：

- 拍照上传药品包装或说明书，识别药品并录入电子药箱
- 生成服药计划，并提供“已服用 / 稍后 / 漏服”状态记录
- 基于药品知识库做用药问答、副作用提示和漏服建议
- 提供家属查看的服药日志和异常摘要

## 运行方式

```bash
cd /Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent
streamlit run app.py --server.port 8510
```

运行后打开：`http://localhost:8510`

## 模块映射

- `app_file_uploader.py`
  - 对应“拍照录药 / 信息确认”
- `knowledge_base.py`
  - 对应“药品知识库构建 / 药品匹配”
- `vector_store.py`
  - 对应“知识检索”
- `rag.py`
  - 对应“RAG问答与安全提示”
- `app_qa.py`
  - 对应“用药问答界面”
- `config/workflow.json`
  - 对应“工作流定义”

## 说明

- 当前 Demo 为离线可演示版，没有强依赖外部模型 API。
- OCR 使用可编辑文本模拟识别结果，便于快速演示完整链路。
- 后续可直接替换为真实 OCR、真实向量库和大模型推理服务。
