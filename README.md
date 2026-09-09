# Insurance Claims Agent

Validates an uploaded claim file against a policy knowledge base.

| Concern | Implementation |
|---|---|
| Reasoning model | Azure OpenAI `gpt-4o-mini` (structured JSON output) |
| Embeddings | Azure OpenAI `text-embedding-3-small` (1536-d) |
| Vector KB | ChromaDB (persistent, cosine) — `data/chroma/` |
| Memory, runs/steps, cost ledger, audit log, doc registry | one SQLite DB — `data/claims_agent.db` |
| Prompts | external Markdown templates in `prompts/` (content-hash versioned per run) |

## Agent flow (`POST /claims/validate`)

1. **extract** — `prompts/claim_extractor.md` → `ClaimRecord` (policy no., type, dates, items, red flags)
2. **retrieve** — several queries derived from the claim → ChromaDB top-k clauses
3. **validate** — `prompts/claim_validator.md` + `claim_validator_user.md` with clauses + session memory → `ValidationResult` (`APPROVE | REJECT | NEEDS_REVIEW`, findings citing `chunk_id`s)
4. **remember** — claim + decision appended to session memory (`session_id`)

Every LLM/embedding call is metered into `llm_calls`; every step into `agent_steps`; every HTTP request and business action into `audit_log`.

## Run locally

```bash
cp .env.example .env          # fill in endpoint + key
uv venv && uv pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Docker:

```bash
docker build -t claims-agent .
docker run -p 8000:8000 --env-file .env -v $PWD/data:/srv/data claims-agent
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/policies/ingest` | multipart `file` (+ `metadata` JSON string) → chunk, embed, upsert into ChromaDB |
| POST | `/policies/ingest-text` | same from JSON body `{filename, text, metadata}` |
| POST | `/policies/search` | `{query, top_k, where}` semantic search |
| GET | `/policies` | KB stats + document registry |
| DELETE | `/policies/{doc_id}` | remove document + its vectors |
| POST | `/claims/validate` | multipart `file` (+ `session_id`) → adjudication |
| POST | `/claims/validate-text` | `{claim_text, session_id}` |
| GET | `/claims/runs`, `/claims/runs/{run_id}` | run history incl. steps + LLM calls |
| GET/DELETE | `/memory/{session_id}` | session memory |
| GET | `/costs` | cost by model / purpose / day |
| GET | `/audit` | audit log (`?action=`, `?limit=`) |
| GET | `/prompts` | prompt templates + versions |
| GET | `/health` | liveness + KB stats |

Send `x-actor: <user>` to attribute actions in the audit log; `x-request-id` is honoured or generated.

## Quick test

```bash
curl -X POST localhost:8000/policies/ingest -H 'x-actor: me' \
  -F file=@sample_data/policies/motor_comprehensive_policy.md -F 'metadata={"product":"motor"}'

curl -X POST localhost:8000/claims/validate -H 'x-actor: me' \
  -F file=@sample_data/claims/claim_motor_valid.txt -F session_id=priya-MC-2041877 | jq .validation

curl localhost:8000/costs | jq .total
```

## Tests

```bash
pytest --cov=app        # offline: fake LLM, real SQLite + ChromaDB in tmp dir
```

## Configuration (`.env`)

`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `CHAT_DEPLOYMENT`, `EMBEDDING_DEPLOYMENT`,
optional `SQLITE_PATH`, `CHROMA_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, `MEMORY_WINDOW`, `PRICING_JSON`.
