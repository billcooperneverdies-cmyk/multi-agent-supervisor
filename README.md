# Multi-Agent Supervisor — AI 软件开发团队

一个模块化的多代理 AI 架构，由四个专业代理协作完成软件开发任务：

- **Alice** (项目经理) — 需求分析、任务分解、进度跟踪
- **Bob** (前端开发) — React/TypeScript UI 实现
- **Charlie** (后端开发) — FastAPI/SQLAlchemy API 开发
- **Diana** (QA 工程师) — 代码审查、Bug 报告、功能验收

## 架构三层

```
┌─────────────────────────────────────────────────────┐
│  LangGraph (Orchestration) — 代理协作编排层          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Alice   │──▶│  Bob/    │──▶│  Diana   │        │
│  │  (PM)    │   │  Charlie │   │  (QA)    │        │
│  │  调度中心 │◄──│ (Dev)    │◄──│  审查    │        │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘        │
│       │              │              │               │
│       └──────────────┘              │               │
│                     AgentState       │               │
└──────────────────────────────────────┼───────────────┘
                                       │ stdio
┌──────────────────────────────────────┼───────────────┐
│  MCP Server (Middleware) — 工具接口层                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ github_* │   │ slack_*  │   │ ask_human│        │
│  │ 代码管理 │   │ 团队通信 │   │ 人工介入 │        │
│  └──────────┘   └──────────┘   └──────────┘        │
│         │              │              │             │
│         └──────────────┘              │             │
│              HTTP                     │             │
└─────────────────────────────────────┼─────────────┘
                                      │
┌─────────────────────────────────────┼─────────────┐
│  External APIs — 外部服务层                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  GitHub  │   │  Slack   │   │ OpenAI   │        │
│  │  REST    │   │  REST    │   │  LLM     │        │
│  └──────────┘   └──────────┘   └──────────┘        │
└─────────────────────────────────────────────────────┘
```

## 快速启动

### 1. 克隆仓库

```bash
git clone https://github.com/billcooperneverdies-cmyk/multi-agent-supervisor.git
cd multi-agent-supervisor
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 token
```

### 3. 安装依赖

```bash
pip install -e ".[dev]"
```

### 4. 运行 MCP 服务器

```bash
python -m mcp_server.main
```

### 5. 运行四代理协作（CLI）

```bash
python -m langgraph.agent
```

### 6. 启动 FastAPI 服务

```bash
python -m langgraph.api
# 访问 http://localhost:8000/docs
```

### 7. Docker Compose（完整栈）

```bash
docker-compose up -d
```

## 项目结构

```
multi-agent-supervisor/
├── README.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Dockerfile.mcp              # MCP 服务器容器
├── Dockerfile.agent              # 主智能体容器
│
├── agents/                       # 四代理定义
│   ├── __init__.py
│   ├── alice.py                  # 项目经理：任务分解、路由调度
│   ├── bob.py                    # 前端开发：React/TypeScript/Tailwind
│   ├── charlie.py                # 后端开发：FastAPI/SQLAlchemy/PostgreSQL
│   └── diana.py                  # QA 工程师：代码审查、Bug 报告、验收
│
├── mcp_server/                   # MCP 服务器（工具层）
│   ├── __init__.py
│   ├── server.py                 # FastMCP 入口：注册所有工具
│   ├── main.py                   # stdio 传输运行器
│   └── tools/
│       ├── __init__.py
│       ├── github.py             # 6 个 GitHub 工具
│       ├── slack.py              # 3 个 Slack 工具
│       └── ask_human.py          # 人工介入工具
│
├── langgraph/                    # 编排层（LangGraph）
│   ├── __init__.py
│   ├── state_machine.py          # 四代理工作流图 + 条件路由
│   ├── agent.py                  # CLI 交互式运行器
│   └── api.py                    # FastAPI 生产服务
│
├── configs/
│   ├── litellm_proxy.yaml        # 统一 LLM 路由配置
│   └── slack_channels.yaml       # 团队通信矩阵 + 消息模板
```

## 四代理协作工作流

```
用户输入需求
    │
    ▼
┌──────────┐
│  Alice   │ 需求分析 → 创建任务板 → 分配任务
│  (PM)    │
└────┬─────┘
     │
     ├──────────────┐
     ▼              ▼
┌──────────┐  ┌──────────┐
│   Bob    │  │ Charlie  │ 并行开发
│ (前端)   │  │ (后端)   │
│frontend/ │  │backend/  │
└────┬─────┘  └────┬─────┘
     │              │
     └──────┬───────┘
            ▼
     ┌──────────┐
     │  Diana   │ 代码审查 → Bug 报告 / 验收批准
     │  (QA)    │
     └────┬─────┘
          │
          ▼
     ┌──────────┐
     │  Alice   │ 通知用户 → 关闭任务
     │  (PM)    │
     └──────────┘
```

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **编排层** | LangGraph + LangChain | 代理状态机、工作流图、条件路由 |
| **工具层** | MCP Python SDK (`FastMCP`) | 标准化工具接口、stdio 传输 |
| **模型层** | LiteLLM Proxy | OpenAI/Anthropic/本地模型统一路由 |
| **HTTP** | httpx + Pydantic | 异步 API 调用、输入验证 |
| **数据** | PostgreSQL + pgvector | 向量存储、AgentState 持久化 |
| **通信** | Slack Web API | 团队频道通信、消息模板 |
| **代码** | GitHub REST API v3 | 分支、提交、PR、Issue 管理 |
| **部署** | Docker Compose | 多服务编排、健康检查 |

## MCP 工具列表

### GitHub 工具

| 工具 | 功能 | 主要使用者 |
|------|------|----------|
| `github_create_pr` | 创建 Pull Request | Bob, Charlie |
| `github_create_branch` | 创建功能分支 | Alice, Bob, Charlie |
| `github_commit_file` | 提交代码文件 | Bob, Charlie |
| `github_read_file` | 读取代码文件 | Diana, Alice |
| `github_list_commits` | 列出最近提交 | Diana |
| `github_create_issue` | 创建 Bug Issue | Diana |

### Slack 工具

| 工具 | 功能 | 主要使用者 |
|------|------|----------|
| `slack_send_message` | 发送频道消息 | 所有代理 |
| `slack_list_channels` | 列出频道 | Alice |
| `slack_get_channel_history` | 读取频道历史 | Alice, Diana |

### 人工介入工具

| 工具 | 功能 | 触发条件 |
|------|------|----------|
| `ask_human` | 暂停工作流请求人类输入 | 需求不明确、需要决策确认 |

## 环境变量

```bash
# GitHub
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Slack
SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI / LiteLLM
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LITELLM_MASTER_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=agent_db

# Agent Service
AGENT_PORT=8000
```

## 开发路线图

- [x] MCP 服务器基础（GitHub + Slack）
- [x] LangGraph 状态机（单代理）
- [x] Docker Compose 部署
- [x] **四代理协作系统**（Alice, Bob, Charlie, Diana）
- [x] 扩展 MCP 工具集（6 GitHub + 3 Slack + 1 Human）
- [x] FastAPI 生产服务
- [ ] 工具绑定 LLM（langchain-mcp-adapters）
- [ ] 状态持久化（Redis + Postgres）
- [ ] 真实代码生成（code-act 模式）
- [ ] 监控仪表盘（任务状态、代理日志）

## License

MIT
