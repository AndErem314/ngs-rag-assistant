# NGS RAG — Quality Streak Log

Automated daily evaluation. Each run generates questions, scores answers, and applies Tier 1 fixes automatically.
Tier 2 fixes create Kanban review cards. Tier 3 issues are reported without code changes.

| Run ID | Date | Passed | Streak | Changes | Decision | Avg Score |
|------|------|--------|--------|---------|----------|------------|
| 20260701-0304 | 01 Jul 2026 | 12/20 | 0 | In the Nextera XT protocol, what bead-to-supernatant ratio i..., What is the incubation temperature and time for the tagmenta..., What volume of Neutralize Tagment Buffer (NT) is added after... | — | 67.9% |

| 20260630-0304 | 30 Jun 2026 | 11/20 | 0 | In the Nextera XT protocol, what bead-to-supernatant ratio i..., What is the incubation temperature and time for the tagmenta..., What volume of Neutralize Tagment Buffer (NT) is added after... | — | 63.4% |

| 20260629-0304 | 29 Jun 2026 | 11/20 | 0 | In the Nextera XT protocol, what bead-to-supernatant ratio i..., How should Illumina Purification Beads (IPB) be resuspended ..., What volume of Neutralize Tagment Buffer (NT) is added after... | — | 61.4% |

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
