# 🚀 RAG 年报智能问答系统

基于 **检索增强生成（RAG）** 的上市公司年报问答系统。支持 PDF 解析、向量检索、LLM 重排、多步推理，通过 Streamlit 网页界面交互。

> 本系统针对**中文年报**场景深度优化，默认使用阿里云 DashScope（Qwen-Plus）作为推理模型。

---

## ✨ 功能特性

- **📊 Streamlit 前端界面** — 一问一答，历史记录保存，参考页面溯源
- **📄 PDF 智能解析** — 基于 Docling 将年报 PDF 转为结构化 Markdown
- **🔍 向量检索 + 重排** — FAISS 向量库 + LLM 重排双重保障召回质量
- **🧠 链式推理** — LLM 分步推理，输出推理过程、摘要、引用页码
- **🏢 多公司支持** — 自动识别问题中的公司名，匹配对应年报
- **⚡ 并行处理** — PDF 解析和问题处理均支持多线程

---

## 🏗️ 系统架构

```
用户提问 → 公司名抽取 → 向量检索(FAISS) → LLM重排 → RAG上下文构建 → LLM推理 → 结构化答案
                                                          ↑
                                              PDF年报 → Markdown → 文本分块 → 向量嵌入
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Windows / Linux / macOS
- 建议使用 GPU（加速 PDF 解析）

### 安装

```bash
git clone https://github.com/你的用户名/rag-annual-report.git
cd rag-annual-report
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS
pip install -e . -r requirements.txt
```

### 配置 API Key

将项目根目录的 `env` 文件重命名为 `.env`，填入你的 API Key：

```bash
# 必须：DashScope API Key（阿里云百炼）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

# 可选：OpenAI / Gemini（如需切换模型）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
GEMINI_API_KEY=xxxxxxxxxxxxxxxx
```

> 🔑 DashScope Key 获取：https://dashscope.console.aliyun.com/apiKey  
> 需开通 **Qwen-Plus** 模型（免费额度够用）：https://dashscope.console.aliyun.com/model

### 启动 Web 界面

```bash
streamlit run app_streamlit.py
```

浏览器访问 `http://localhost:8501`，输入问题即可开始问答。

### 命令行批量处理

```bash
# 解析 PDF 年报
python main.py parse-pdfs

# 分块 + 建向量库
python main.py process-reports

# 批量处理问题
python main.py process-questions --config max
```

---

## 📁 项目结构

```
RAG-wh/
├── app_streamlit.py          # Streamlit Web 前端
├── main.py                   # CLI 命令行入口
├── src/
│   ├── pipeline.py           # 主流程编排 & 配置
│   ├── pdf_parsing.py        # PDF 解析（Docling）
│   ├── pdf_mineru.py         # PDF 转 Markdown
│   ├── text_splitter.py      # 文本分块
│   ├── ingestion.py          # 向量库构建（FAISS + BM25）
│   ├── retrieval.py          # 检索器（向量检索 / 混合检索）
│   ├── reranking.py          # LLM 重排器
│   ├── questions_processing.py  # 问题处理 & 答案生成
│   ├── api_requests.py       # LLM API 调用（DashScope / OpenAI / Gemini）
│   ├── prompts.py            # Prompt 模板
│   └── tables_serialization.py  # 表格序列化
├── data/stock_data/          # 数据目录
│   ├── pdf_reports/          # PDF 年报
│   ├── subset.csv            # 公司信息表
│   ├── questions.json        # 问题集
│   └── databases/            # 向量数据库（自动生成）
├── env                       # API Key 模板（重命名为 .env）
└── requirements.txt
```

---

## ⚙️ 配置说明

在 `src/pipeline.py` 中修改 `max_config`：

```python
max_config = RunConfig(
    parent_document_retrieval=True,   # 父文档检索
    llm_reranking=True,               # LLM 重排
    top_n_retrieval=10,               # 检索返回数量
    answering_model="qwen-plus",      # 推理模型
    api_provider="dashscope",         # API 提供商
)
```

可选模型：`qwen-plus`（推荐）、`qwen-max`、`gpt-4o-mini`、`gemini-2.0-flash`

---

## 🔧 已知问题 & 解决

| 问题 | 原因 | 解决 |
|------|------|------|
| DashScope 免费额度用完 | Qwen-Turbo 免费额度耗尽 | 切换到 `qwen-plus`（有独立额度） |
| 页码显示为 0 | Markdown 分块丢失页码 | 从行号自动估算（约 50 行/页） |
| `.pyc` 缓存导致代码未更新 | Python 字节码缓存 | `sys.dont_write_bytecode = True` |
| `NoneType is not iterable` | API 返回 `output: null` | 全面改用 `hasattr` / `.get()` 安全访问 |

---

## 📄 License

MIT
