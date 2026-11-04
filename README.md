# Multi-Agent Supervisor — AI 软件开发团队

一个模块化的多代理 AI 架构（v0.2.0），由四个专业代理协作完成软件开发任务：

- **Alice** (项目经理) — 需求分析、任务分解、进度跟踪、路由调度
- **Bob** (前端开发) — React/TypeScript UI 实现
- **Charlie** (后端开发) — FastAPI/SQLAlchemy API 开发
- **Diana** (QA 工程师) — 代码审查、Bug 报告、功能验收

## 架构三层

```
┌─────────────────────────────────────────────────────┐
│  orchestrator/ (LangGraph 编排层)                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Alice   │──▶│  Bob/    │──▶│  Diana   │        │
│  │  (PM)    │   │  Charlie │   │  (QA)    │        │
│  │  调度中心 │◄──│ (Dev)    │◄──│  审查    │        │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘        │
│       └──────── AgentState (共享状态) ─────────┘     │
│  agents/ = 四代理 persona prompt 工厂                │
│  api.py = FastAPI 服务 · agent.py = CLI 运行器       │
└──────────────────────────────────────┼───────────────┘
                                       │ stdio (MCP_SERVER_CMD 子进程)
┌──────────────────────────────────────┼───────────────┐
│  mcp_server/ (MCP 工具层, FastMCP)                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ github_* │   │ slack_*  │   │ ask_human│        │
│  │ 代码管理 │   │ 团队通信 │   │ 人工介入 │        │
│  └──────────┘   └──────────┘   └──────────┘        │
└─────────────────────────────────────┼─────────────┘
                                      │ HTTP
┌─────────────────────────────────────┼─────────────┐
│  External APIs — 外部服务层                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  GitHub  │   │  Slack   │   │ LiteLLM  │        │
│  │  REST    │   │  REST    │   │  Proxy   │        │
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
# 编辑 .env 填入你的 token（AGENT_API_KEY / LITELLM_* / POSTGRES_PASSWORD 为必填）
```

### 3. 安装依赖

```bash
pip install -e ".[dev]"
```

### 4. 运行 MCP 服务器（stdio）

```bash
python -m mcp_server.main
```

通常不需要手动运行 —— 代理通过 `MCP_SERVER_CMD` 自动把它作为 stdio 子进程拉起（见下文「MCP 工具接线」）。

### 5. 运行四代理协作（CLI）

```bash
python -m orchestrator.agent
```

交互式运行器：输入需求描述，图运行到 `human` 节点时暂停并在终端请求输入；输入 `exit` 退出。

### 6. 启动 FastAPI 服务

```bash
python -m orchestrator.api
# 或： uvicorn orchestrator.api:app --host 127.0.0.1 --port 8000
# 访问 http://127.0.0.1:8000/docs
```

本地默认绑定 `127.0.0.1:8000`（`AGENT_HOST` / `AGENT_PORT` 可覆盖）。未设置 `AGENT_API_KEY` 时服务拒绝启动。

### 7. Docker Compose（完整栈）

```bash
docker-compose up -d
```

## API 认证

除 `GET /health` 外，所有 `/tasks/*` 端点都需要请求头：

```
X-API-Key: <AGENT_API_KEY>
```

缺失或错误返回 `401`；服务端未配置 `AGENT_API_KEY` 时返回 `503`（启动期则直接 fail-fast）。

## API 端点

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| GET | `/health` | 否 | 健康检查；返回版本、draining 与 in-flight 计数 |
| POST | `/tasks/invoke` | 是 | 启动四代理任务；body: `description`, `repo_owner`, `repo_name`, 可选 `slack_channel`, `thread_ts`, `max_iterations` |
| POST | `/tasks/human-input` | 是 | HITL 恢复；body: `task_id`, `answer` |
| GET | `/tasks/{task_id}/status` | 是 | 从检查点快照读取任务真实状态 |

## 任务状态词汇表

| 状态 | 含义 |
|------|------|
| `running` | 图仍在执行（快照存在 pending next 节点） |
| `awaiting_human` | 图在 `human` 节点 `interrupt()`，等待人工输入 |
| `completed` | 正常结束（`active_agent == "end"`） |
| `terminated_max_iterations` | 达到 `max_iterations` 被强制终止 |
| `failed` | 无 pending 工作且未正常结束 |

## HITL 人工介入恢复流程

1. 任一代理输出 `[[NEXT:human]]` → 图进入 `human` 节点并 `interrupt()`，检查点持久化暂停状态。
2. `/tasks/invoke` 或 `/tasks/{task_id}/status` 返回 `status="awaiting_human"`（invoke 响应的 `summary` 含问题摘要）。
3. 客户端调用 `POST /tasks/human-input`，body `{"task_id": "...", "answer": "..."}`。
4. 服务端以 `Command(resume=answer)` 恢复图执行，返回新的运行状态。

## 哨兵路由协议

代理节点通过输出末尾的整词哨兵 token 驱动条件路由（自由文本永远不会触发路由；终止 token 优先于代理名 token）：

| 哨兵 | 发出者 | 含义 |
|------|--------|------|
| `[[NEXT:bob]]` / `[[NEXT:charlie]]` / `[[NEXT:diana]]` / `[[NEXT:alice]]` / `[[NEXT:human]]` | 所有代理 | 将控制权移交给指定代理（或请求人工介入） |
| `[[QA:pass]]` | Diana | 验收通过 → 回到 Alice 收尾 |
| `[[QA:fail]]` | Diana | 验收失败 → 配合 `[[NEXT:bob|charlie]]` 指定修复方 |
| `[[END]]` | Alice | 任务完成，终止工作流 |

## MCP 工具接线

`MCP_SERVER_CMD`（默认 `python -m mcp_server.main`）是代理节点 spawn MCP 工具服务器的命令。首次需要工具时，编排层通过 `langchain-mcp-adapters` 以 stdio 拉起该子进程并绑定工具。未设置该变量、adapters 缺失或子进程启动失败时，图仍可运行 —— 节点退化为无工具的纯 LLM 调用（graceful degradation，日志告警）。

## 项目结构

```
multi-agent-supervisor/
├── README.md
├── LICENSE                       # MIT
├── pyproject.toml
├── .env.example                  # 全部环境变量模板（含注释）
├── docker-compose.yml
├── Dockerfile.mcp                # MCP 服务器容器
├── Dockerfile.agent              # 主智能体容器
│
├── agents/                       # 四代理 persona prompt 工厂
│   ├── alice.py                  # 项目经理：任务分解、路由调度
│   ├── bob.py                    # 前端开发：React/TypeScript/Tailwind
│   ├── charlie.py                # 后端开发：FastAPI/SQLAlchemy/PostgreSQL
│   └── diana.py                  # QA 工程师：代码审查、Bug 报告、验收
│
├── mcp_server/                   # MCP 服务器（工具层）
│   ├── server.py                 # FastMCP 入口：注册所有工具
│   ├── main.py                   # stdio 传输运行器
│   └── tools/
│       ├── github.py             # 6 个 GitHub 工具
│       ├── slack.py              # 3 个 Slack 工具
│       ├── ask_human.py          # 人工介入工具
│       └── errors.py             # ToolError + 统一错误映射
│
├── orchestrator/                 # 编排层（基于 LangGraph）
│   ├── state_machine.py          # 四代理工作流图 + 哨兵路由 + 检查点
│   ├── agent.py                  # CLI 交互式运行器
│   └── api.py                    # FastAPI 生产服务
│
├── configs/
│   └── litellm_proxy.yaml        # 统一 LLM 路由配置
│
└── tests/                        # pytest 套件（145+ 用例）
```

## 四代理协作工作流

```
用户输入需求
    │
    ▼
┌──────────┐
│  Alice   │ 需求分析 → [[NEXT:bob]] / [[NEXT:charlie]]
│  (PM)    │
└────┬─────┘
     │
     ├──────────────┐
     ▼              ▼
┌──────────┐  ┌──────────┐
│   Bob    │  │ Charlie  │ 实现 → [[NEXT:diana]]
│ (前端)   │  │ (后端)   │
│frontend/ │  │backend/  │
└────┬─────┘  └────┬─────┘
     │              │
     └──────┬───────┘
            ▼
     ┌──────────┐
     │  Diana   │ [[QA:pass]] → Alice；[[QA:fail]] + [[NEXT:bob|charlie]] → 修复
     │  (QA)    │
     └────┬─────┘
          ▼
     ┌──────────┐
     │  Alice   │ [[END]] → 任务完成
     │  (PM)    │
     └──────────┘
```

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **编排层** | LangGraph + LangChain | 代理状态机、工作流图、条件路由、检查点 |
| **工具层** | MCP Python SDK (`FastMCP`) | 标准化工具接口、stdio 传输 |
| **模型层** | LiteLLM Proxy | OpenAI/Anthropic/本地模型统一路由 |
| **HTTP** | httpx + Pydantic | 异步 API 调用、输入验证 |
| **持久化** | InMemorySaver / PostgresSaver | AgentState 检查点（`DATABASE_URL` 可选） |
| **通信** | Slack Web API | 团队频道通信 |
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

完整模板见 [.env.example](.env.example)（每个变量都有注释；**必填** = 未设置则启动失败）。

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `LITELLM_MASTER_KEY` | 是 | 无 | LiteLLM proxy 主密钥（兼 admin UI 凭据） |
| `LITELLM_API_KEY` | 是 | 无 | agent 访问 LiteLLM proxy 的密钥；代码无默认值，缺失即拒启动 |
| `POSTGRES_PASSWORD` | 是(compose) | 无 | Postgres 超级用户密码（仅内部网络可达） |
| `AGENT_API_KEY` | 是 | 无 | `/tasks/*` 的 `X-API-Key` 认证密钥；未配置则 API 拒启动 |
| `OPENAI_API_KEY` | 否 | 无 | gpt-4o 模型条目使用的 OpenAI key |
| `ANTHROPIC_API_KEY` | 否 | 无 | 仅当路由到 claude-3-5-sonnet 条目时需要 |
| `GITHUB_TOKEN` | 否 | 无 | MCP GitHub 工具使用的 PAT |
| `SLACK_BOT_TOKEN` | 否 | 无 | MCP Slack 工具使用的 xoxb- token |
| `GITHUB_API_BASE` | 否 | `https://api.github.com` | GitHub Enterprise 覆盖 |
| `SLACK_API_BASE` | 否 | `https://slack.com/api` | Slack 兼容网关覆盖 |
| `POSTGRES_USER` | 否 | `postgres` | compose 用它拼 `DATABASE_URL` |
| `POSTGRES_DB` | 否 | `agent_db` | 同上 |
| `AGENT_PORT` | 否 | `8000` | agent API 发布端口 / 容器内 uvicorn 端口 |
| `AGENT_HOST` | 否 | `127.0.0.1` | 本地运行绑定地址；compose 显式设为 `0.0.0.0` |
| `LITELLM_API_BASE` | 否 | `http://localhost:4000` | 本地运行时 proxy 地址；compose 内覆盖为 `http://litellm-proxy:4000` |
| `DATABASE_URL` | 否 | 空 | 检查点 DSN；空 = InMemorySaver，设置后用 PostgresSaver（缺包时回退内存） |
| `LLM_REQUEST_TIMEOUT` | 否 | `60` | 单次 LLM 请求超时（秒） |
| `LLM_MAX_RETRIES` | 否 | `2` | LLM 调用有界重试次数 |
| `TASK_TIMEOUT_SECONDS` | 否 | `300` | 单任务整体Deadline；超时取消并返回结构化 504 |
| `MAX_CONCURRENT_TASKS` | 否 | `4` | 并发任务上限；超限返回结构化 503 |
| `SHUTDOWN_GRACE_SECONDS` | 否 | `5` | 关停时取消在途任务的有界等待（秒） |
| `MCP_SERVER_CMD` | 否 | `python -m mcp_server.main` | 代理 spawn MCP stdio 子进程的命令；空 = 无工具运行 |
| `HISTORY_WINDOW` | 否 | `20` | 每次 LLM 调用携带的最近消息数 |
| `LITELLM_PORT` | 否 | `4000` | LiteLLM proxy 发布端口 |
| `MCP_HTTP_TIMEOUT` | 否 | `30` | MCP 工具出站 HTTP 读超时（秒） |
| `MCP_HTTP_CONNECT_TIMEOUT` | 否 | `5` | MCP 工具出站 HTTP 连接超时（秒） |

## 开发

```bash
pip install -e ".[dev]"
pytest -q          # 测试套件
ruff check .       # 代码风格 / 静态检查（0 errors）
mypy orchestrator mcp_server agents   # 严格类型检查
```

## License

MIT — 见 [LICENSE](LICENSE)。
