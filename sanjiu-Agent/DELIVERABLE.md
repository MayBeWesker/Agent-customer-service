# 华润三九「AI用药伴侣」交付说明

## 1. 用户旅程地图

用户旅程图素材可直接使用你提供的成品图：

- [用户旅程地图原图](/Users/ethan/Downloads/ChatGPT%20Image%202026%E5%B9%B46%E6%9C%885%E6%97%A5%2021_52_44.png)

8 个关键事件可总结为：

1. 打开应用：进入首页，只保留“拍照录药”和“电子药箱”两个核心入口。
2. 拍照录药：上传药盒照片或说明书截图，系统识别药名、规格、用法用量。
3. 信息确认：用户手动修正识别结果，也允许家属协助确认。
4. 生成计划：设置频次和提醒时间，生成电子药箱与服药计划。
5. 定时提醒：通过大字提醒和语音式文案提示到点服药。
6. 记录状态：记录“已服用 / 稍后 / 漏服”，形成时间日志。
7. 用药问答：结合知识库和电子药箱回答副作用、漏服、联合用药问题。
8. 持续管理：家属查看记录，复诊前导出清单，识别异常情况。

## 2. Demo 地址

- 本地可体验地址：[http://localhost:8510](http://localhost:8510)
- 是否需要账号：不需要
- 建议演示路径：
  - 先在“拍照录药”里选择样例药品
  - 确认后查看“电子药箱”
  - 在“提醒记录”页点击一次“已服用”
  - 再到“用药问答”页提问“这个药有什么副作用？”

## 3. 关键代码 / 配置文件

关键代码入口：

- [app.py](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/app.py)
- [app_file_uploader.py](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/app_file_uploader.py)
- [knowledge_base.py](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/knowledge_base.py)
- [vector_store.py](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/vector_store.py)
- [rag.py](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/rag.py)
- [system_prompt.txt](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/prompts/system_prompt.txt)
- [workflow.json](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/config/workflow.json)

已生成的截图素材：

- [system-prompt.png](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/assets/system-prompt.png)
- [workflow-config.png](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/assets/workflow-config.png)
- [rag-code.png](/Volumes/Program/projects/RAG-Agent-projects/sanjiu-Agent/assets/rag-code.png)

与原始 RAG 项目的映射关系：

- `RAG-CustomerService/app_file_uploader.py` -> `sanjiu-Agent/app_file_uploader.py`
- `RAG-CustomerService/knowledge_base.py` -> `sanjiu-Agent/knowledge_base.py`
- `RAG-CustomerService/vector_store.py` -> `sanjiu-Agent/vector_store.py`
- `RAG-CustomerService/rag.py` -> `sanjiu-Agent/rag.py`
- `RAG-CustomerService/app_qa.py` -> `sanjiu-Agent/app_qa.py`

## 4. 最需要验证的假设

最需要验证的不是“AI 回答够不够聪明”，而是：

**55-75 岁中老年用户，是否愿意并且能够独立完成“拍照录药 -> 信息确认 -> 设置提醒”这条核心建档链路。**

原因：

- 后续所有价值都依赖电子药箱先建立起来。
- 如果初次录药很费劲，提醒、日志、问答和家属协同都会失效。
- 对这个年龄段用户而言，最大风险往往不是问不到答案，而是第一步就不会用、嫌麻烦、或者不敢信。

建议最小验证指标：

- 建档完成率
- 平均完成时长
- 需要家属帮助的比例
- OCR 识别后人工修改率
- 提醒设置成功率
- 用户信任评分
