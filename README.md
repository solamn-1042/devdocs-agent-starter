# DevDocs Agent 学习项目

这是从最简单的“文档检索→大模型回答”开始、逐步升级到 RAG+Agent 的个人学习项目。

教学参考：Datawhale LLM-Universe https://github.com/datawhalechina/llm-universe ；Hello-Agents https://github.com/datawhalechina/hello-agents 。本项目的初始代码为独立教学示例，不直接复制原仓库源码。

## 开始（Windows PowerShell + Conda）

```powershell
conda activate py312
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python -m unittest discover -s tests -v
python app.py
```

首先可以不填API Key测试检索；要获取真实模型回答，填入.env中的DEEPSEEK_API_KEY。

输入“Agent的工具调用由谁执行？”可查看来源、匹配分数和回答。

Day 1 为关键词检索雏形，尚不支持向量检索、文档分块、前端页面、Agent循环等，不能将其写作已完成的简历能力。
