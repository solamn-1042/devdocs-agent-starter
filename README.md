# DevDocs Agent

研发文档智能知识库助手：RAG检索、证据引用、LangGraph工具、FastAPI接口。

## 主要功能
- Markdown/TXT文档、Chunk ID、来源引用
- Embedding、Chroma、BM25混合检索与Rerank
- 证据不足拒答、20条人工标注样本（Gold）评测
- LangGraph search_docs/read_source工具
- 多轮追问、Trace与FastAPI

## Windows本地启动
1. 使用Conda与Python 3.12。
2. 执行 python -m pip install -r requirements.txt
3. 将.env.example复制为.env，填写自己可用的API密钥。
4. 按Day5课程重建Chroma索引。
5. 执行 python -m uvicorn api_day15:app --host 127.0.0.1 --port 8000
6. 浏览器访问 http://127.0.0.1:8000/docs

## 评测
- python eval_day8.py
- python -m unittest discover -s tests -v
- python final_audit_day16.py

## 实测指标
以下为单机实测结果，仅供参考；不同硬件、网络、模型服务及运行状态可能产生差异。

| 项目 | 结果 |
| --- | --- |
| Gold评测样本 | 20条，包含问题、预期来源和证据原文 |
| Chroma向量检索 Hit@3 | 100.0%（20/20） |
| Chroma平均检索耗时 | 37.3 ms（Embedding模型已加载，不含索引同步和模型API生成） |
| Agent混合检索＋词项重排 Hit@3 | 90.0%（18/20），不含模型生成 |
| 全量离线回归 | 94/94通过 |
| HTTP回归 | 6/6通过 |
| Day16固定检查 | 30/30通过 |

Hit@3表示前3个检索结果中包含标注来源及证据原文的样本比例。Chroma与Agent使用不同检索路径，指标分别统计；逐题Chroma结果见 [评测记录](evals/day8_results.csv)。固定20题的结果不代表对任意问题的泛化表现。

HTTP回归覆盖健康检查、正常问答、空问题、缺少问题字段、非法历史和底层服务失败，使用模拟业务调用，不代表真实模型API验证通过。

### 真实API延迟
使用 `deepseek-v4-flash` 连续请求3次，3/3成功：

- 平均完整响应耗时：**1763.61 ms**。
- 中位数：1640.18 ms；范围：1568.61～2082.03 ms。
- 测试条件：非流式短文本生成，`max_tokens=256`、`temperature=0`，客户端禁用重试。

耗时包含网络、服务端处理和生成，不含检索及本地HTTP接口；不代表首Token延迟或多次模型调用的端到端耗时。3次请求仅作连通性与延迟抽样，不代表长期性能。

## 安全与局限
本项目用于本地教学，无公网鉴权、限流和生产部署。
不得提交.env、密钥、私有知识文件与敏感日志。