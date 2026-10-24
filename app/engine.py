"""Incident engine.

Three separate concerns:
  1. planning   - model call and plan validation (only in create_run, once per runId)
  2. state      - receipt-driven state machine (retries, approvals, effects)
  3. telemetry  - actionLog, receiptLog and OTLP built from stored state only

Replays and receipts never call the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
from functools import lru_cache

from .planner import PlannerError, parse_json_object

log = logging.getLogger("ga5")

PROFILE = "ga5-incident-agent/v2"
AGENT_NAME = "incident-response"
EVIDENCE_RE = re.compile(r"^\[([^\]\s]+)\]")
LIMIT_RE = re.compile(r"limit|ceiling|at_max|maxed", re.IGNORECASE)
TRACE_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")

UNSET, OK, ERROR = 0, 1, 2
KIND_INTERNAL, KIND_SERVER, KIND_CLIENT = 1, 2, 3


class ApiError(Exception):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status = status
        self.code = code


# ---------------------------------------------------------------- helpers

def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_ns() -> int:
    return time.time_ns()


def rand_hex(nbytes: int) -> str:
    while True:
        value = secrets.token_hex(nbytes)
        if int(value, 16) != 0:
            return value


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(8)}"


def _nonempty(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _kv(key: str, value) -> dict:
    if isinstance(value, bool):
        any_value = {"boolValue": value}
    elif isinstance(value, int):
        any_value = {"intValue": value}
    else:
        any_value = {"stringValue": str(value)}
    return {"key": key, "value": any_value}


@lru_cache(maxsize=32)
def evidence_index(transcript: str) -> frozenset:
    """Preprocessing is deterministic, so cache it by transcript."""
    ids = set()
    for line in transcript.splitlines():
        match = EVIDENCE_RE.match(line.strip())
        if match:
            ids.add(match.group(1))
    return frozenset(ids)


# ------------------------------------------------------------- validation

def validate_request(body) -> None:
    if not isinstance(body, dict):
        raise ApiError(422, "body_not_object")
    if body.get("profile") != PROFILE:
        raise ApiError(400, "unsupported_profile")
    if not _nonempty(body.get("runId")):
        raise ApiError(422, "invalid_run_id")
    if not isinstance(body.get("publicMarker"), str):
        raise ApiError(422, "invalid_public_marker")

    incident = body.get("incident")
    if not isinstance(incident, dict):
        raise ApiError(422, "invalid_incident")
    for key in ("incidentId", "title", "service", "severity", "transcript"):
        if not isinstance(incident.get(key), str):
            raise ApiError(422, "invalid_incident")
    roots = incident.get("allowedRootCauses")
    if not isinstance(roots, list) or not roots or not all(_nonempty(r) for r in roots):
        raise ApiError(422, "invalid_root_causes")

    catalog = body.get("toolCatalog")
    if not isinstance(catalog, list) or not catalog:
        raise ApiError(422, "invalid_tool_catalog")
    for tool in catalog:
        if not isinstance(tool, dict) or not _nonempty(tool.get("name")):
            raise ApiError(422, "invalid_tool_catalog")
        if "inputSchema" in tool and not isinstance(tool["inputSchema"], dict):
            raise ApiError(422, "invalid_tool_catalog")

    policy = body.get("policy")
    if not isinstance(policy, dict):
        raise ApiError(422, "invalid_policy")
    max_diag = policy.get("maximumDiagnostics")
    if not isinstance(max_diag, int) or isinstance(max_diag, bool) or not 1 <= max_diag <= 3:
        raise ApiError(422, "invalid_policy")
    for key in ("effectTools", "approvalRequiredFor", "doNotExport"):
        value = policy.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ApiError(422, "invalid_policy")


def _check_evidence(value, allowed, low: int, high: int) -> list:
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise PlannerError("bad_evidence_count")
    if len(set(value)) != len(value):
        raise PlannerError("duplicate_evidence")
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            raise PlannerError("unknown_evidence")
    return list(value)


def _check_required(tool: dict, args: dict) -> None:
    schema = tool.get("inputSchema") or {}
    required = schema.get("required", [])
    if isinstance(required, list):
        for key in required:
            if key not in args:
                raise PlannerError("missing_argument")


def validate_plan(raw, incident: dict, catalog: list, policy: dict) -> dict:
    if not isinstance(raw, dict):
        raise PlannerError("plan_not_object")
    root = raw.get("rootCause")
    if root not in incident["allowedRootCauses"]:
        raise PlannerError("bad_root_cause")

    valid_ids = evidence_index(incident["transcript"])
    dx_evidence = _check_evidence(raw.get("evidence"), valid_ids, 2, 4)

    tools = {t["name"]: t for t in catalog}
    effect_names = set(policy["effectTools"])
    diagnostics = raw.get("diagnostics")
    if not isinstance(diagnostics, list) or not 1 <= len(diagnostics) <= policy["maximumDiagnostics"]:
        raise PlannerError("bad_diagnostic_count")

    diag_names = set()
    clean_diags = []
    for item in diagnostics:
        if not isinstance(item, dict):
            raise PlannerError("bad_diagnostic")
        name = item.get("toolName")
        if name not in tools or name in effect_names or name in diag_names:
            raise PlannerError("bad_diagnostic_tool")
        args = item.get("arguments")
        if not isinstance(args, dict):
            raise PlannerError("bad_arguments")
        _check_required(tools[name], args)
        evidence = _check_evidence(item.get("evidence"), set(dx_evidence), 1, 4)
        diag_names.add(name)
        clean_diags.append({"toolName": name, "arguments": args, "evidence": evidence})

    return {
        "rootCause": root,
        "evidence": dx_evidence,
        "diagnostics": clean_diags,
        "effect": _check_effect(raw.get("effect"), tools, diag_names, dx_evidence),
        "pageEffect": _check_effect(raw.get("pageEffect"), tools, diag_names, dx_evidence),
    }


def _check_effect(value, tools, diag_names, evidence):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlannerError("bad_effect")
    name = value.get("toolName")
    if name not in tools or name in diag_names:
        raise PlannerError("bad_effect_tool")
    args = value.get("arguments")
    if not isinstance(args, dict):
        raise PlannerError("bad_arguments")
    _check_required(tools[name], args)
    return {"toolName": name, "arguments": args, "evidence": list(evidence)}


def _run_planner(planner, incident, catalog, policy) -> dict:
    """One model call per new run, with one retry if the output is unusable."""
    error = None
    start = end = now_ns()
    for _ in range(2):
        start = now_ns()
        try:
            text = planner.complete(incident, catalog, policy)
            end = now_ns()
            plan = validate_plan(parse_json_object(text), incident, catalog, policy)
            return {"plan": plan, "model": planner.model, "startNs": start, "endNs": end, "error": None}
        except Exception as exc:  # planner transport, parsing, or validation failure
            end = now_ns()
            error = "planner_failed"
            log.error("planner attempt failed: %s: %s", type(exc).__name__, exc)
    return {"plan": None, "model": planner.model, "startNs": start, "endNs": end, "error": error}


# ----------------------------------------------------------- state machine

def _add_action(state, phase, tool, args, evidence, action_id=None, extra=None):
    action_id = action_id or new_id("act")
    action = {
        "actionId": action_id,
        "callId": new_id("call"),
        "phase": phase,
        "toolName": tool,
        "arguments": args,
        "evidence": list(evidence),
        "spanId": rand_hex(8),
        "attempts": [],
        "state": "pending",
        "resultClass": None,
        "errorType": None,
        "startNs": None,
        "endNs": None,
    }
    state["actions"][action_id] = action
    state["actionOrder"].append(action_id)
    return _issue(state, action, 1, extra)


def _issue(state, action, attempt_no: int, extra=None) -> dict:
    """Issue one physical attempt. Each attempt gets its own CLIENT span ID."""
    span_id = rand_hex(8)
    issued = now_ns()
    dispatch = {
        "actionId": action["actionId"],
        "callId": action["callId"],
        "phase": action["phase"],
        "toolName": action["toolName"],
        "arguments": action["arguments"],
        "evidence": action["evidence"],
        "attempt": attempt_no,
        "traceparent": f"00-{state['traceId']}-{span_id}-01",
    }
    if state.get("tracestate"):
        dispatch["tracestate"] = state["tracestate"]
    if extra:
        dispatch.update(extra)
    action["attempts"].append({
        "attempt": attempt_no,
        "spanId": span_id,
        "issuedNs": issued,
        "receipt": None,
        "receivedNs": None,
    })
    if action["startNs"] is None:
        action["startNs"] = issued
    state["dispatchLog"].append(dispatch)
    return dispatch


def _finish(state, status: str) -> None:
    state["status"] = status
    state["endNs"] = now_ns()


def _suppress_planned_effects(state) -> None:
    plan = state["plan"]
    for key in ("effect", "pageEffect"):
        if plan and plan.get(key):
            name = plan[key]["toolName"]
            if name not in state["suppressed"]:
                state["suppressed"].append(name)


def _check_outcome(state, outcome, seen: set):
    if not isinstance(outcome, dict):
        raise ApiError(422, "invalid_outcome")
    action_id = outcome.get("actionId")
    call_id = outcome.get("callId")
    attempt_no = outcome.get("attempt")
    status = outcome.get("status")
    nonce = outcome.get("nonce")
    result_class = outcome.get("resultClass", "")
    error_type = outcome.get("errorType")
    if not (_nonempty(action_id) and _nonempty(call_id) and _nonempty(nonce)):
        raise ApiError(422, "invalid_outcome")
    if not isinstance(attempt_no, int) or isinstance(attempt_no, bool):
        raise ApiError(422, "invalid_outcome")
    if not isinstance(status, int) or isinstance(status, bool):
        raise ApiError(422, "invalid_outcome")
    if not isinstance(result_class, str):
        raise ApiError(422, "invalid_outcome")
    if error_type is not None and not isinstance(error_type, str):
        raise ApiError(422, "invalid_outcome")

    action = state["actions"].get(action_id)
    if action is None or action["callId"] != call_id or action["state"] != "pending":
        raise ApiError(422, "unknown_or_settled_action")
    last = action["attempts"][-1]
    if attempt_no != last["attempt"] or last["receipt"] is not None:
        raise ApiError(422, "attempt_not_pending")
    key = ("outcome", action_id, attempt_no)
    if key in seen:
        raise ApiError(422, "duplicate_outcome")
    seen.add(key)
    return outcome, action, last


def _check_approval(state, approval, seen: set):
    if not isinstance(approval, dict):
        raise ApiError(422, "invalid_approval")
    approval_id = approval.get("approvalId")
    decision = approval.get("decision")
    nonce = approval.get("nonce")
    if not (_nonempty(approval_id) and _nonempty(nonce)):
        raise ApiError(422, "invalid_approval")
    if decision not in ("approved", "rejected"):
        raise ApiError(422, "invalid_approval")
    record = state["approvals"].get(approval_id)
    if record is None or record["state"] != "pending":
        raise ApiError(422, "unknown_or_settled_approval")
    key = ("approval", approval_id)
    if key in seen:
        raise ApiError(422, "duplicate_approval")
    seen.add(key)
    return approval, record


def _apply_outcome(state, receipt_id, outcome, action, attempt, dispatches) -> None:
    now = now_ns()
    status = outcome["status"]
    result_class = outcome.get("resultClass", "")
    error_type = outcome.get("errorType")
    attempt["receipt"] = {
        "receiptId": receipt_id,
        "status": status,
        "resultClass": result_class,
        "nonce": outcome["nonce"],
        "errorType": error_type,
    }
    attempt["receivedNs"] = now
    state["receiptLog"].append({
        "receiptId": receipt_id,
        "actionId": action["actionId"],
        "callId": action["callId"],
        "attempt": attempt["attempt"],
        "status": status,
        "resultClass": result_class,
        "nonce": outcome["nonce"],
    })

    if status == 200:
        action["state"] = "succeeded"
        action["resultClass"] = result_class
        action["endNs"] = now
    elif status == 503 and attempt["attempt"] == 1:
        # Exactly one retry for 503. The retry carries a new attempt and span.
        dispatches.append(_issue(state, action, 2))
    else:
        action["state"] = "failed"
        action["errorType"] = error_type or str(status)
        action["endNs"] = now


def _apply_approval(state, receipt_id, approval, record) -> None:
    now = now_ns()
    record["state"] = approval["decision"]
    record["nonce"] = approval["nonce"]
    record["receiptId"] = receipt_id
    record["receivedNs"] = now
    state["receiptLog"].append({
        "receiptId": receipt_id,
        "approvalId": record["approvalId"],
        "decision": approval["decision"],
        "nonce": approval["nonce"],
    })


def _dispatch_effect(state, dispatches) -> None:
    es = state["effect"]
    extra = None
    if es["approvalId"]:
        record = state["approvals"][es["approvalId"]]
        extra = {"approvalId": record["approvalId"], "approvalNonce": record["nonce"]}
    dispatch = _add_action(
        state, "effect", es["tool"], es["args"], es["evidence"],
        action_id=es["reservedActionId"], extra=extra,
    )
    es["actionId"] = dispatch["actionId"]
    dispatches.append(dispatch)


def _advance(state, dispatches, requests) -> None:
    """Move the state machine forward. Appends new dispatches and approval requests."""
    actions = [state["actions"][aid] for aid in state["actionOrder"]]
    diags = [a for a in actions if a["phase"] == "diagnostic"]

    if any(a["state"] == "pending" for a in diags):
        return
    if any(a["state"] == "failed" for a in diags):
        # A timed-out or failed prerequisite suppresses the dependent effect.
        _suppress_planned_effects(state)
        _finish(state, "failed")
        return

    es = state["effect"]
    plan = state["plan"]

    if not es["decided"]:
        es["decided"] = True
        at_limit = any(LIMIT_RE.search(a["resultClass"] or "") for a in diags)
        if at_limit:
            # At the permitted scaling limit: page the owner, never scale.
            if plan["effect"] and plan["effect"]["toolName"] not in state["suppressed"]:
                state["suppressed"].append(plan["effect"]["toolName"])
            chosen = plan["pageEffect"]
        else:
            chosen = plan["effect"]
        if chosen is None:
            _finish(state, "completed")
            return
        es["tool"] = chosen["toolName"]
        es["args"] = chosen["arguments"]
        es["evidence"] = chosen["evidence"]
        if es["tool"] in state["approvalRequiredFor"]:
            approval_id = new_id("appr")
            reserved = new_id("act")
            request = {
                "approvalId": approval_id,
                "actionId": reserved,
                "toolName": es["tool"],
                "argumentsDigest": sha256_hex(canon(es["args"])),
            }
            state["approvals"][approval_id] = {
                **request, "state": "pending", "nonce": None,
                "spanId": rand_hex(8), "requestedNs": now_ns(), "receivedNs": None,
            }
            es["approvalId"] = approval_id
            es["reservedActionId"] = reserved
            requests.append(request)
            return
        es["reservedActionId"] = None
        _dispatch_effect(state, dispatches)
        return

    if es["approvalId"] and es["actionId"] is None:
        record = state["approvals"][es["approvalId"]]
        if record["state"] == "pending":
            return
        if record["state"] == "rejected":
            if es["tool"] not in state["suppressed"]:
                state["suppressed"].append(es["tool"])
            _finish(state, "completed")
            return
        _dispatch_effect(state, dispatches)
        return

    if es["actionId"] is None:
        return
    act = state["actions"][es["actionId"]]
    if act["state"] == "pending":
        return
    if act["state"] == "succeeded":
        state["chosenEffect"] = act["toolName"]
        _finish(state, "completed")
    else:
        _finish(state, "failed")


def _build_response(state, dispatches, requests) -> dict:
    if state["status"] == "waiting":
        return {
            "runId": state["runId"],
            "status": "waiting",
            "diagnosis": state["diagnosis"],
            "dispatches": dispatches,
            "approvals": requests,
        }
    return final_body(state)


def final_body(state) -> dict:
    return {
        "runId": state["runId"],
        "status": state["status"],
        "diagnosis": state["diagnosis"],
        "chosenEffect": state["chosenEffect"],
        "suppressed": state["suppressed"],
        "actionLog": state["dispatchLog"],
        "receiptLog": state["receiptLog"],
        "otlp": build_otlp(state),
    }


# --------------------------------------------------------------- entry points

def create_run(store, planner, body, traceparent=None, tracestate=None) -> dict:
    validate_request(body)
    run_id = body["runId"]
    request_hash = sha256_hex(canon(body))

    existing = store.get(run_id)
    if existing is not None:
        if existing["requestHash"] != request_hash:
            raise ApiError(409, "run_conflict")
        return existing["createResponse"]

    incident = body["incident"]
    catalog = body["toolCatalog"]
    policy = body["policy"]

    match = TRACE_RE.match(traceparent or "")
    if match and int(match.group(1), 16) and int(match.group(2), 16):
        trace_id, parent_span, trace_state = match.group(1), match.group(2), tracestate
    else:
        trace_id, parent_span, trace_state = rand_hex(16), None, None

    started = now_ns()
    planned = _run_planner(planner, incident, catalog, policy)

    state = {
        "runId": run_id,
        "requestHash": request_hash,
        "publicMarker": body["publicMarker"],
        "traceId": trace_id,
        "tracestate": trace_state,
        "parentSpanId": parent_span,
        "serverSpanId": rand_hex(8),
        "agentSpanId": rand_hex(8),
        "joinSpanId": rand_hex(8),
        "serverStartNs": started,
        "endNs": None,
        "model": {
            "name": planned["model"],
            "spanId": rand_hex(8),
            "startNs": planned["startNs"],
            "endNs": planned["endNs"],
            "error": planned["error"],
        },
        "status": "waiting",
        "diagnosis": None,
        "plan": planned["plan"],
        "approvalRequiredFor": list(policy["approvalRequiredFor"]),
        "effect": {
            "decided": False, "tool": None, "args": None, "evidence": None,
            "reservedActionId": None, "actionId": None, "approvalId": None,
        },
        "actions": {},
        "actionOrder": [],
        "dispatchLog": [],
        "receiptLog": [],
        "approvals": {},
        "chosenEffect": None,
        "suppressed": [],
        "receipts": {},
        "current": None,
        "createResponse": None,
    }

    if planned["plan"] is None:
        state["endNs"] = now_ns()
        state["status"] = "failed"
        out = final_body(state)
    else:
        plan = planned["plan"]
        state["diagnosis"] = {"rootCause": plan["rootCause"], "evidence": plan["evidence"]}
        dispatches = [
            _add_action(state, "diagnostic", d["toolName"], d["arguments"], d["evidence"])
            for d in plan["diagnostics"]
        ]
        out = {
            "runId": run_id,
            "status": "waiting",
            "diagnosis": state["diagnosis"],
            "dispatches": dispatches,
            "approvals": [],
        }

    state["createResponse"] = out
    state["current"] = out
    store.put(run_id, state)
    return out


def apply_receipt(store, run_id: str, body) -> dict:
    state = store.get(run_id)
    if state is None:
        raise ApiError(404, "run_not_found")
    if not isinstance(body, dict):
        raise ApiError(422, "body_not_object")

    receipt_id = body.get("receiptId")
    if not _nonempty(receipt_id):
        raise ApiError(422, "invalid_receipt_id")
    outcomes = body.get("outcomes", [])
    approvals = body.get("approvals", [])
    if not isinstance(outcomes, list) or not isinstance(approvals, list):
        raise ApiError(422, "invalid_receipt")
    if not outcomes and not approvals:
        raise ApiError(422, "empty_receipt")

    digest = sha256_hex(canon(body))
    seen_receipt = state["receipts"].get(receipt_id)
    if seen_receipt is not None:
        if seen_receipt["hash"] != digest:
            raise ApiError(409, "receipt_conflict")
        return seen_receipt["response"]

    if state["status"] != "waiting":
        raise ApiError(422, "run_not_waiting")

    # Validate everything before changing anything.
    seen = set()
    checked_outcomes = [_check_outcome(state, o, seen) for o in outcomes]
    checked_approvals = [_check_approval(state, a, seen) for a in approvals]

    dispatches, requests = [], []
    for outcome, action, attempt in checked_outcomes:
        _apply_outcome(state, receipt_id, outcome, action, attempt, dispatches)
    for approval, record in checked_approvals:
        _apply_approval(state, receipt_id, approval, record)
    _advance(state, dispatches, requests)

    response = _build_response(state, dispatches, requests)
    state["receipts"][receipt_id] = {"hash": digest, "response": response}
    state["current"] = response
    store.put(run_id, state)
    return response


def get_current(store, run_id: str) -> dict:
    state = store.get(run_id)
    if state is None:
        raise ApiError(404, "run_not_found")
    return state["current"]


# ------------------------------------------------------------------ OTLP

def build_otlp(state) -> dict:
    trace_id = state["traceId"]
    end = state["endNs"] or now_ns()
    common = [_kv("ga5.run.id", state["runId"]), _kv("ga5.public.marker", state["publicMarker"])]
    spans = []

    def add(span_id, parent_id, name, kind, start, stop, attrs, code=UNSET, links=None):
        span = {
            "traceId": trace_id,
            "spanId": span_id,
            "name": name,
            "kind": kind,
            "startTimeUnixNano": str(int(start)),
            "endTimeUnixNano": str(int(stop)),
            "attributes": attrs + common,
            "status": {"code": code},
        }
        if parent_id:
            span["parentSpanId"] = parent_id
        if links:
            span["links"] = links
        spans.append(span)

    server_start = state["serverStartNs"]
    add(state["serverSpanId"], state["parentSpanId"], "POST /v2/incidents",
        KIND_SERVER, server_start, end, [])

    agent = state["agentSpanId"]
    add(agent, state["serverSpanId"], f"invoke_agent {AGENT_NAME}", KIND_INTERNAL,
        server_start, end, [
            _kv("gen_ai.operation.name", "invoke_agent"),
            _kv("gen_ai.agent.name", AGENT_NAME),
        ])

    model = state["model"]
    model_attrs = [
        _kv("gen_ai.operation.name", "chat"),
        _kv("gen_ai.request.model", model["name"] or "unknown"),
    ]
    model_code = UNSET
    if model["error"]:
        model_attrs.append(_kv("error.type", model["error"]))
        model_code = ERROR
    add(model["spanId"], agent, "chat incident-plan", KIND_CLIENT,
        model["startNs"], model["endNs"], model_attrs, model_code)

    for action_id in state["actionOrder"]:
        act = state["actions"][action_id]
        tool = act["toolName"]
        act_code = ERROR if act["state"] == "failed" else UNSET
        add(act["spanId"], agent, f"execute_tool {tool}", KIND_INTERNAL,
            act["startNs"], act["endNs"] or end, [
                _kv("ga5.action.id", action_id),
                _kv("gen_ai.tool.name", tool),
                _kv("gen_ai.tool.call.id", act["callId"]),
                _kv("gen_ai.operation.name", "execute_tool"),
            ], act_code)

        for att in act["attempts"]:
            receipt = att["receipt"]
            attrs = [
                _kv("ga5.action.id", action_id),
                _kv("ga5.attempt", att["attempt"]),
                _kv("http.request.method", "POST"),
                _kv("http.request.resend_count", att["attempt"] - 1),
            ]
            code = UNSET
            stop = att["receivedNs"] or end
            if receipt:
                attrs.append(_kv("ga5.receipt.id", receipt["receiptId"]))
                attrs.append(_kv("ga5.receipt.nonce", receipt["nonce"]))
                status = receipt["status"]
                if status > 0:
                    attrs.append(_kv("http.response.status_code", status))
                if status == 200:
                    code = OK
                else:
                    code = ERROR
                    attrs.append(_kv("error.type", receipt.get("errorType") or str(status)))
            add(att["spanId"], act["spanId"], f"POST tool/{tool}", KIND_CLIENT,
                att["issuedNs"], stop, attrs, code)

    diags = [state["actions"][aid] for aid in state["actionOrder"]
             if state["actions"][aid]["phase"] == "diagnostic"]
    if len(diags) > 1:
        links = [{"traceId": trace_id, "spanId": d["spanId"]} for d in diags]
        add(state["joinSpanId"], agent, "incident.join", KIND_INTERNAL,
            min(d["startNs"] for d in diags),
            max((d["endNs"] or end) for d in diags), [], links=links)

    for record in state["approvals"].values():
        attrs = [
            _kv("ga5.approval.id", record["approvalId"]),
            _kv("ga5.action.id", record["actionId"]),
            _kv("ga5.approval.decision", record["state"]),
        ]
        if record["nonce"]:
            attrs.append(_kv("ga5.approval.receipt.nonce", record["nonce"]))
        add(record["spanId"], agent, "approval_gate", KIND_INTERNAL,
            record["requestedNs"], record["receivedNs"] or end, attrs)

    return {
        "resourceSpans": [{
            "resource": {"attributes": [_kv("service.name", "ga5-incident-agent")]},
            "scopeSpans": [{"scope": {"name": "ga5-incident-agent"}, "spans": spans}],
        }]
    }
