# MiniRAG

MiniRAG 是一个使用 Go 构建的个人知识库 RAG 简易版：上传 Markdown、TXT 或可复制文字的 PDF，使用本地 Ollama 生成 Embedding，使用阿里云百炼 Chat 模型回答问题，并返回召回片段作为引用。

## 当前能力

- 文档 hash 去重和同步索引
- 异步索引任务、任务状态查询和失败重试
- 按当前分块/Embedding 配置异步重建文档索引，保留文档 ID 和 checksum
- 60 条 Retrieval 与 31 条 RAG Benchmark 样本，覆盖中英文、改写、多文档、代码/PDF、相似干扰和无答案拒答
- Markdown 结构化分块：保留标题路径、代码块类型、语言和 PDF 页码元数据
- 同一章节内相邻短段落自动合并，并为 chunk 补充标题上下文
- 中英文分块、Unicode 安全和可配置重叠窗口
- PostgreSQL + pgvector 向量检索
- pgvector 向量召回 + PostgreSQL FTS/pg_trgm 中英文关键词召回 + RRF 融合
- 可配置候选池，引用返回两路排名、原始分数和 RRF 分数
- 本地 llama.cpp + BGE multilingual cross-encoder rerank，默认可运行并带文档多样性约束
- 阿里云百炼 OpenAI 兼容接口生成回答
- Ollama `bge-m3` 本地 Embedding
- 引用返回 chunk 元数据，网页展示章节路径、PDF 页码和代码语言
- 简单网页：上传文档、查看文档、提问、查看引用
- OpenTelemetry Trace、Prometheus Metrics 和关键依赖 readiness
- Docker Compose 启动 PostgreSQL；llama.cpp reranker 由脚本管理

## 运行前提

- Go
- Docker Compose（启动 PostgreSQL/pgvector）
- Ollama，并下载 Embedding 模型
- 阿里云百炼 API Key，且 Chat 模型有调用权限

如果系统没有 Docker，可执行：

```bash
bash scripts/install-docker.sh
```

脚本会要求输入一次 sudo 密码，并安装 Ubuntu 的 Docker 和 Compose 插件。

如果安装后执行 `bash scripts/run.sh` 出现 `docker.sock: permission denied`，说明当前终端还没有刷新用户组。重新登录终端后再运行即可；也可以临时执行：

```bash
sg docker -c 'bash scripts/run.sh'
```

安装本地模型：

```bash
ollama pull bge-m3
```

启动 Ollama 服务（保持这个终端运行）：

```bash
ollama serve
```

再打开另一个终端启动 MiniRAG。

启动数据库：

```bash
docker compose up -d postgres
```

安装并启动本地 reranker（首次下载约 438MB GGUF 模型）：

```bash
bash scripts/start-reranker.sh
```

配置环境变量：

```bash
cp .env.example .env
# 编辑 .env，填写 BAILIAN_API_KEY
set -a; source .env; set +a
```

启动服务：

```bash
bash scripts/run.sh
```

浏览器打开 http://localhost:8080。

如果当前用户没有 Docker socket 权限，但 PostgreSQL 已经由其他方式启动，可以跳过 scripts/run.sh 中的 Compose 步骤，手动启动应用：

```bash
set -a
source .env
set +a
bash scripts/start-reranker.sh
go run ./cmd/minirag
```

go run ./cmd/minirag 会以前台进程运行；关闭该终端或发送 SIGTERM 会触发 Worker 和 HTTP 服务的优雅退出。

## Observability

MiniRAG 使用 OpenTelemetry 记录请求、查询、文档索引和 Durable Queue 的执行链路，使用 Prometheus 记录延迟、错误和队列状态。观测代码通过现有 `context.Context` 贯穿调用链，不改变 Retrieval、Rerank、Queue 或 Evaluation 的业务接口。

默认只启用本地 Trace API 和 Prometheus 指标，不会向外部发送 Trace。要在本地查看完整 Trace，先启动 PostgreSQL 和观测组件：

```bash
docker compose up -d postgres
docker compose --profile observability up -d jaeger prometheus
```

然后在 `.env` 中设置 OTLP HTTP exporter（Jaeger 接收端口为 4318）：

```dotenv
METRICS_ENABLED=true
OTEL_SERVICE_NAME=minirag
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
OTEL_EXPORTER_OTLP_INSECURE=true
```

启动应用后：

- 应用指标：<http://localhost:8080/metrics>
- Jaeger Trace UI：<http://localhost:16686>
- Prometheus UI：<http://localhost:9090>
- 就绪检查：<http://localhost:8080/readyz>

一次 Query 的 Trace 会按 `HTTP Request -> Query Pipeline -> Embedding -> Vector Search -> Lexical Search -> RRF -> Rerank -> Context Build -> LLM -> Response` 展开。上传或 reindex 请求会把 W3C carrier 持久化到 PostgreSQL 的 `ingestion_jobs.trace_context`，Worker 领取后继续该 Trace，并记录 `Acquire Job`、`Durable Queue Job`、`Lease Renew`、`Retry`、`Complete` 或 `Failed`。因此服务重启后重新领取的任务仍可关联到原始请求。

### Streaming API

`POST /v1/chat/stream` 使用百炼兼容的真正 Streaming API。服务端不会先等待完整答案再包装 SSE，而是读取上游 `text/event-stream` 的每个 delta，并立即转发给客户端。请求格式与普通 Query 兼容：

```bash
curl -N -X POST http://localhost:8080/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"RAG 的基本流程是什么？","top_k":5}'
```

事件顺序如下：

```text
event: token       data: {"content":"..."}       # 可重复，实时追加
event: citations   data: {"citations":[...]}      # 模型输出完成后发送一次
event: done        data: {"total_ms":..., ...}    # 最终耗时和检索元数据
```

发生错误时发送 `event: error`。客户端断开会取消 HTTP request Context，Provider 使用同一个 Context 取消百炼请求和响应读取，停止后续 token 消费。前端网页使用 `fetch` 的 ReadableStream 解析这些 SSE 事件，因此回答会边生成边显示，引用只在 `citations` 事件到达后渲染。

Streaming Trace 在现有 Query Pipeline 下增加 `Streaming` 和 `First Token` span；`Streaming` span 记录首 token 延迟、模型生成耗时和端到端总耗时，Query Pipeline span 的结束时间代表完整请求结束。

`/readyz` 是依赖就绪检查，不是简单的进程存活检查。它会并行检查 PostgreSQL、Embedding provider、LLM provider 和启用时的 Reranker，并检查 Worker 是否正在运行；任一必需依赖未就绪时返回 `503`。Reranker 未启用时会报告 `disabled`，不会阻塞服务。

常用 PromQL：

```promql
# 按路由和状态查看请求速率
sum(rate(minirag_http_requests_total[5m])) by (route, status)

# 5xx 错误率
sum(rate(minirag_http_requests_total{status=~"5.."}[5m]))
/ sum(rate(minirag_http_requests_total[5m]))

# 按流水线阶段查看 P95 延迟
histogram_quantile(0.95,
  sum(rate(minirag_stage_duration_seconds_bucket[5m])) by (le, pipeline, stage))

# 查询总耗时 P95
histogram_quantile(0.95, sum(rate(minirag_query_duration_seconds_bucket[5m])) by (le))

# Durable Queue 当前长度、处理中数量和重试速率
minirag_queue_depth
minirag_queue_processing
sum(rate(minirag_queue_jobs_total{outcome="retried"}[5m]))
```

本地验证可以先不依赖真实模型：`curl http://localhost:8080/metrics` 应返回 Prometheus 文本格式，`curl -i http://localhost:8080/readyz` 应明确列出每个 dependency 的状态。真实 Query/Ingestion Trace 需要 PostgreSQL、Ollama、Chat provider（以及启用时的 reranker）均可访问；不要用真实 LLM 生成请求做 readiness 探针。

也可以使用 API：

```bash
curl -F 'file=@seeddata/rag-basics.md' http://localhost:8080/v1/documents
curl -X POST http://localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"RAG 的基本流程是什么？","top_k":5}'
```

上传接口返回 `202 Accepted` 和 `job.id`；通过 `GET /v1/jobs/{job_id}` 查看索引状态。

按当前分块和 Embedding 配置重建已有文档：

```bash
curl -X POST http://localhost:8080/v1/documents/{document_id}/reindex
```

reindex 返回新的异步 job，完成后保留原文档 ID、checksum 和创建时间，只替换该文档的 chunks。存在原始上传内容时，会按当前分块和 Embedding 配置完整重建；早期文档没有保留原始内容时，会使用现有 chunks 刷新 Embedding，并在 metadata 中标记 `reindex_source=stored_chunks`，但不会重新切分。

## Durable Queue 设计

索引任务使用 PostgreSQL 持久化队列。`ingestion_jobs` 是任务状态的唯一来源；Worker 的内存中不保存待执行任务 ID，因此进程重启不会丢失已经返回 `202 Accepted` 的上传或重建任务。现有上传、任务查询和 reindex API 保持不变。

选择 PostgreSQL 而不是内存 channel 或额外消息队列，是因为项目已经依赖 PostgreSQL，而且当前默认只有一个索引 Worker。这样既能获得事务、行锁和持久化能力，也不需要为 Redis、RabbitMQ 或 Kafka 增加新的部署与一致性边界。Worker 默认每 500ms 从数据库领取一个到期的 `queued` 任务。

领取任务在单个事务中执行：先按 `next_attempt_at` 和创建时间选择一行，并使用 `FOR UPDATE SKIP LOCKED` 加锁；随后将它更新为 `processing`，写入 `worker_id` 和 `lease_until` 后提交。`SKIP LOCKED` 让多个 Worker 遇到已被其他 Worker 锁定的任务时直接跳过，而不是等待或重复消费。当前只启动一个 Worker，但领取协议允许以后安全增加实例。

处理中的 Worker 会周期性续租。服务启动以及运行期间都会把 lease 已过期的 `processing` 任务恢复为 `queued`；新的 Worker 随后可以重新领取。完成、失败和续租更新都校验 `worker_id`，完成与失败还要求 lease 未过期，防止已经失去所有权的旧 Worker覆盖新 Worker 的状态。

瞬时错误按 `1s -> 5s -> 20s -> 60s` 指数退避，`next_attempt_at` 到期后才能重新领取。网络错误、超时、HTTP 408/429/5xx，以及有限的 PostgreSQL 可恢复错误会重试；无效文档、解析失败和 HTTP 4xx 等永久错误直接进入 `failed`。`attempts` 记录领取次数，`retry_count` 记录已安排的重试次数，`error_message` 保存最近一次错误摘要。

崩溃恢复流程如下：

1. Worker 领取任务并写入 lease。
2. 正常执行时持续续租，完成后在当前 Worker 所有权下写入 `completed`。
3. 进程异常退出时 lease 不再续期；lease 到期后任务恢复为 `queued`。
4. 新 Worker 重新领取并执行任务。

该语义是 At-Least-Once：崩溃发生在外部 Embedding 已调用、但任务状态尚未提交的边界时，外部调用可能重复。文档写入继续使用 checksum 去重和数据库事务保证幂等结果。收到 SIGTERM、SIGINT 或 Context Cancel 时，Worker 停止领取新任务，取消当前索引调用并把仍由自己持有的任务释放回 `queued`，HTTP 服务随后安全退出。

## 离线评测

先通过网页或 `make seed` 导入 `seeddata/` 中的示例文档，等待任务状态变成 `completed`。评测 CLI 不需要启动 MiniRAG HTTP 服务，但需要 PostgreSQL 和 Ollama；运行前加载 `.env`：

```bash
set -a
source .env
set +a
```

评测命令（`--pipeline retrieval`）不调用百炼 Chat。Vector、Lexical、Hybrid(RRF) 和 Hybrid+Rerank 使用 Ollama Embedding；Rerank 会先取 Hybrid 候选，再调用本地 llama.cpp + `bge-reranker-v2-m3-Q4_K_M.gguf` cross-encoder 打分，最后限制每个文档最多 2 个 chunk。`--mode weighted` 仍保留用于历史实验，但默认 `all` 只比较验收要求的四条链路。

默认一次比较四种模式，也可以使用 `--mode vector|lexical|weighted|hybrid|rerank` 单独运行。评测会在全部样本和模式完成后一次性输出 JSON，运行期间终端可能没有进度输出；建议先分模式验证依赖：

```bash
mkdir -p artifacts

go run ./cmd/minirag eval --dataset testdata/eval.json --mode lexical --top-k 5 --candidate-k 50 > artifacts/retrieval-lexical.json
go run ./cmd/minirag eval --dataset testdata/eval.json --mode vector --top-k 5 --candidate-k 50 > artifacts/retrieval-vector.json
go run ./cmd/minirag eval --dataset testdata/eval.json --mode hybrid --top-k 5 --candidate-k 50 > artifacts/retrieval-hybrid.json
```

需要运行完整四模式对比时，先确认 Reranker 已启动：

```bash
bash scripts/start-reranker.sh
go run ./cmd/minirag eval --dataset testdata/eval.json --mode all --top-k 5 --candidate-k 50 > artifacts/retrieval-result.json 2> artifacts/retrieval-error.log
```

Rerank 在 CPU 环境下通常是最慢阶段；`candidate-k=50` 可能耗时较长或超时。可以先用 `candidate-k=5` 做连通性验证，但该结果不应与 `candidate-k=50` 的实验直接比较。

输出结构示例（数值仅用于说明字段，不代表当前机器的最新结果）：

```json
{
  "reports": [
    {
      "mode": "hybrid_rrf",
      "top_k": 5,
      "candidate_k": 50,
      "recall_at_1": 0.9722,
      "recall_at_k": 0.9722,
      "mrr": 1,
      "ndcg_at_k": 0.9785,
      "average_latency_ms": 146.9,
      "p95_latency_ms": 171
    }
  ]
}
```

`Recall@K` 表示 Top-K 覆盖的相关文档比例，重复召回同一文档不会重复计分。`MRR` 反映第一个相关结果的排名，`nDCG@K` 衡量多个相关文档的整体排序质量。

`testdata/eval.json` 现在包含 60 条 Retrieval 样本，`testdata/rag_eval.json` 包含 31 条 RAG 样本，覆盖中英文、语义改写、多文档、代码/PDF 标签、相似干扰和无答案拒答。无答案样本会进入延迟与拒答评测，但不会作为有相关文档的 Recall 分母。

### Baseline 与 CI 回归

使用 `--baseline` 指定历史结果 JSON；`--update-baseline` 会把当前结果与历史结果合并，质量指标保留最高值，延迟和失败数保留最低值：

```bash
go run ./cmd/minirag eval --dataset testdata/eval.json --pipeline retrieval --mode hybrid --top-k 5 --candidate-k 50 --baseline artifacts/retrieval-baseline.json --update-baseline
```

CI 可以在固定数据集、`top-k` 和 `candidate-k` 下检查退化。默认相对容差为 5%，质量指标下降或延迟/失败数上升超过容差会列入 `regressions`；加 `--fail-on-regression` 时命令在输出完整 JSON 后返回非零：

```bash
go run ./cmd/minirag eval \
  --dataset testdata/eval.json \
  --pipeline retrieval \
  --mode hybrid \
  --top-k 5 \
  --candidate-k 50 \
  --baseline artifacts/retrieval-baseline.json \
  --regression-tolerance 0.05 \
  --fail-on-regression
```

Baseline 会校验数据集路径、pipeline、Top-K 和 candidate-K，避免把不同实验条件误当成回归。Baseline 只保存每种模式的聚合指标，不保存逐样本答案，适合提交为 CI artifact；原始逐样本结果仍由普通 stdout JSON 提供。

### 完整 RAG Pipeline 评测

`testdata/rag_eval.json` 在检索标注之外增加了 `reference_answer`、`required_citations` 和 `should_refuse`。运行完整评测会调用 Chat 模型，并输出 Answer Correctness、Citation Precision、Citation Recall、Faithfulness、Unsupported Claim Rate 和 Refusal Accuracy：

```bash
go run ./cmd/minirag eval --pipeline rag --dataset testdata/rag_eval.json --mode hybrid --top-k 5 --candidate-k 50
```

需要对比真正的重排链路时，先启动一个兼容 `POST /rerank` 的服务，再运行：

```bash
go run ./cmd/minirag eval --pipeline rag --dataset testdata/rag_eval.json --mode rerank --top-k 5 --candidate-k 50
```

输出仍是稳定 JSON，可直接保存为 CI artifact 或用于基线比较。默认 `RuleJudge` 不调用额外评测框架：答案使用确定性的 token F1，引用通过回答中的 `[n]` 来源标记计算，Faithfulness 和 Unsupported Claim Rate 使用引用片段的词项证据覆盖计算。`evaluation.Judge` 接口可替换为 LLM-as-a-Judge，但不改变评测报告格式。

RAG 报告同时保留检索阶段的 `latency_ms`/`p95_latency_ms`，并增加完整链路的 `average_pipeline_latency_ms`、`p95_pipeline_latency_ms`、`answer_evaluated_cases` 和 `answer_failed_cases`。因此可以区分“召回慢”与“生成慢”，也可以在 CI 中检查失败样本数。

### Prompt Optimization

Generation 使用与 Retrieval 解耦的 Context 准备步骤：保持现有召回结果不变，按 RRF/Rerank 相关性稳定排序；规范化正文后去除重复 chunk；每个文档最多保留 2 个片段；在 6000 字节预算内优先写入文档名、标题路径和页码，再写正文。最终引用编号以过滤后的 Context 顺序为准。

线上查询与 RAG Evaluation 共用同一套 Prompt。Prompt 要求完整覆盖问题关键点、优先使用 Context 中的术语、删除重复解释、每个事实就近引用，并禁止使用外部知识；Context 不足时只能回答“资料不足，我不知道。”。

在同一模型、`testdata/rag_eval.json`、Hybrid、`top-k=5`、`candidate-k=50` 条件下，本地优化对比如下：

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| Answer Correctness | 0.7109 | 0.7776 | +0.0667 |
| Citation Precision | 1.0000 | 1.0000 | 0 |
| Citation Recall | 1.0000 | 1.0000 | 0 |
| Faithfulness | 1.0000 | 1.0000 | 0 |
| Unsupported Claim Rate | 0.0000 | 0.0000 | 0 |
| Average Pipeline Latency | 5627ms | 5915ms | +288ms |
| P95 Pipeline Latency | 8689ms | 8420ms | -269ms |

下面的优化表和本地实测表是扩充 Benchmark 之前的历史快照，仅用于解释 Prompt 优化时的变化；新的 60/31 条数据请通过 `eval --baseline` 重新生成可复现基线，不应直接与旧快照混比。

### 历史 Rerank 基准

以下结果是在本机使用 llama.cpp `b10293`、`bge-reranker-v2-m3-Q4_K_M.gguf`、Ollama `bge-m3`、百炼 Chat、旧版 4 条 `testdata/rag_eval.json`、`top-k=5`、`candidate-k=50` 实际运行得到，属于历史快照；JSON 原始结果保存在评测命令 stdout 中：

| Pipeline（4 条 RAG 样本） | Recall@1 | Recall@5 | MRR | nDCG@5 | Answer Correctness | Citation P/R | Faithfulness | Unsupported | Avg Pipeline | P95 Pipeline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7974 | 1.00 / 1.00 | 1.00 | 0.00 | 5993ms | 7892ms |
| Lexical | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7897 | 1.00 / 1.00 | 1.00 | 0.00 | 5425ms | 6043ms |
| Weighted Hybrid | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.8065 | 1.00 / 1.00 | 1.00 | 0.00 | 6672ms | 9921ms |
| Hybrid + RRF | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7347 | 1.00 / 1.00 | 1.00 | 0.00 | 6162ms | 10452ms |
| Hybrid + Rerank | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7960 | 1.00 / 1.00 | 1.00 | 0.00 | 11730ms | 12825ms |

在扩充前的 18 条 Retrieval Benchmark 上，Hybrid+Rerank 曾达到 Recall@5 1.0000；该数字只作为历史记录保留。当前实验应以 `testdata/eval.json` 的 60 条样本和 baseline 文件为准。

旧版 RAG 数据集只有 4 条样本，单次模型措辞变化会明显影响平均值；扩充后的 31 条数据用于后续 CI 回归，RuleJudge 的 token-F1、引用证据覆盖等定义保持不变。

扩充前的历史本地基线：

| 模式 | Recall@1 | Recall@5 | MRR | nDCG@5 | 平均延迟 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.9722 | 1.0000 | 1.0000 | 0.9917 | 166.2ms | 233ms |
| Lexical | 0.9167 | 0.9722 | 0.9722 | 0.9580 | 1.8ms | 6ms |
| Hybrid | 0.9722 | 0.9722 | 1.0000 | 0.9785 | 146.9ms | 171ms |
| Rerank | 见下方真实基准 | - | - | - | - | - |

## 配置说明

模型、地址和切分参数全部通过环境变量配置。尤其是 `BAILIAN_CHAT_MODEL`，如果当前账号没有 `glm-5.2` 权限，改成百炼控制台中可用的 Chat 模型即可。

`RETRIEVAL_CANDIDATE_K` 默认是 `50`，表示 Vector 和 Lexical 每路进入 RRF 的最大候选数。最终返回数量仍由请求中的 `top_k` 控制；候选数越大通常召回更充分，但数据库计算量也更高。

`RERANK_ENABLED` 默认开启，`RERANK_URL` 指向 llama.cpp 的 `/v1/rerank` 接口；`RERANK_TIMEOUT_MS` 默认 30000ms。`RERANK_MAX_PER_DOC` 默认是 `2`，避免同一文档的重复 chunk 挤占最终引用。模型和二进制缓存位于 `${XDG_CACHE_HOME:-~/.cache}/minirag-reranker`。

默认 Embedding 维度是 `1024`，与 `bge-m3` 匹配。更换 Embedding 模型时，需要确认向量维度，并重新建立数据库索引；不同模型的向量不能混用。

PDF 解析依赖系统命令 `pdftotext`。Ubuntu 可以执行 `sudo apt install poppler-utils`；扫描件 PDF 暂不支持 OCR。

结构化分块从新上传文档开始生效。已有文档可以通过 reindex 切换到当前策略；没有保留原始 payload 的早期文档只能刷新已有 chunks 的 Embedding，无法重新切分。

## 开发检查

```bash
make fmt
make test
make build
```

Durable Queue 的 PostgreSQL 集成测试使用独立随机 schema，只有显式提供专用测试数据库时才运行：

```bash
MINIRAG_TEST_DATABASE_URL='postgres://minirag:minirag@localhost:5432/minirag?sslmode=disable' \
  go test ./internal/store -run '^TestQueue' -count=1
```

未设置 `MINIRAG_TEST_DATABASE_URL` 时，这些集成测试会跳过；其余单元测试照常运行。测试覆盖 `FOR UPDATE SKIP LOCKED`、多 Worker 不重复领取、延迟重试、永久失败、lease 过期恢复、服务重启重新领取和旧 Worker fencing。

## 后续路线

1. 增加文档版本管理，并在网页中补齐已有删除 API。
2. 增加元数据过滤、引用命中率评测和离线索引管理。
3. 在现有 Trace 和 Metrics 基础上增加成本统计。
4. 增加用户反馈闭环，用于构建更稳定的人工审核评测集。
