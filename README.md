# GA5 Incident Agent (FastAPI on Render)

Implements the `ga5-incident-agent/v2` contract:

- `POST /v2/incidents` starts a run and returns the diagnosis plus the first diagnostic dispatches.
- `POST /v2/incidents/{runId}/receipts` processes outcomes (200, 503 retry, timeout) and approvals.
- `GET /v2/incidents/{runId}` returns the current stored state.
- `GET /healthz` is the Render health check.

## Layout

```
app/main.py      HTTP routes, size limit, error mapping (no request content echoed)
app/engine.py    validation, plan checks, receipt state machine, actionLog/receiptLog, OTLP builder
app/planner.py   prompt construction (sensitive object excluded) and OpenAI-compatible model call
app/store.py     SQLite run store (derived state only, never the sensitive object)
tests/           local flow tests with a fake planner
```

Planning, state changes, and trace construction are separate. Replays and receipts
read stored state and never call the model.

## Deploy on Render

1. Push this folder to a GitHub repo.
2. In Render, choose **New > Blueprint** and select the repo. Render reads `render.yaml`.
   (Or **New > Web Service**, runtime Python, build `pip install -r requirements.txt`,
   start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, and add a disk at `/var/data`.)
3. In the service's **Environment** tab, set `LLM_API_KEY` to your provider key.
   Any OpenAI-compatible provider works: change `LLM_BASE_URL` and `LLM_MODEL` to match.
4. Deploy. Your base URL is `https://<service>.onrender.com`. Submit that to the grader.

Render's free plan has no persistent disk. The `render.yaml` uses `starter` so that
runs survive restarts. Without a disk, state is lost on restart, and replays of earlier
runs would no longer match.

## Test locally

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Smoke test after deploy

```bash
curl -s https://<service>.onrender.com/healthz
```

## Notes and things to tune

- **Model budget.** Each new `runId` makes one planning call (with one retry if the output
  fails validation). The default model is `llama-3.1-8b-instant`, and the timeout is
  `LLM_TIMEOUT_SECONDS=8`. The grader allows 18 seconds per request, so keep the timeout
  under about 8 seconds.
- **Limit detection.** A diagnostic `resultClass` that matches `limit|ceiling|at_max|maxed`
  is treated as "already at the scaling limit". The effect is then replaced by the
  planner's `pageEffect`, and the scale effect is listed in `suppressed`. Adjust
  `LIMIT_RE` in `app/engine.py` if the grader uses different wording.
- **OTLP encoding.** Attributes use standard OTLP JSON (`stringValue`, `intValue`,
  `boolValue`). `ga5.attempt` and `http.response.status_code` are emitted as integers.
  Span IDs and trace IDs are lowercase hex.
- **Concurrency.** A single lock serializes state changes, so concurrent receipts for the
  same run cannot interleave.
- **Verification status.** The code compiles. The bundled tests exercise replay, conflicts,
  retry, timeout suppression, approval, and trace correlation. Run them locally before
  deploying, since they were not executed in the build environment.
