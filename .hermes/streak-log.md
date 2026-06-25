# NGS RAG — Quality Streak Log

Automated daily evaluation. Each run generates questions, scores answers, and applies Tier 1 fixes automatically.
Tier 2 fixes create Kanban review cards. Tier 3 issues are reported without code changes.

| Run ID | Date | Passed | Streak | Changes | Decision | Avg Score |
|------|------|--------|--------|---------|----------|------------|
| (baseline) | — | — | 0 | Initial setup | — | — |

---

## Tier System

| Tier | Scope | Action |
|------|-------|--------|
| **Tier 1** | Prompt wording, thresholds, config, question wording | Auto-merge to main |
| **Tier 2** | Chunking logic, retrieval strategy, new code | Branch → Kanban review card → user merges/discards |
| **Tier 3** | Architecture, new deps, DB schema | Halt — report only, no code written |

---

## Quick Reference

- **Eval harness:** `python scripts/streak_eval.py`
- **Cron:** Daily 03:00 Berlin (`0 3 * * *`)
- **Model:** `llama3.1:8b` (Ollama local)
- **Embedding:** `haybu/mxbai-embed-large-latest:latest`
- **Pass threshold:** ≥60/100 per question
