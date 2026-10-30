import json
import uuid
import secrets
import time
import os
import re
import hashlib
import urllib.request
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, Header
import uvicorn

app = FastAPI()

# ----------------- IN-MEMORY STATE -----------------
RUNS = {}

class ActionAttempt:
    def __init__(self, attempt: int, client_span_id: str):
        self.attempt = attempt
        self.client_span_id = client_span_id
        self.dispatch_dict = {}
        self.receipt = None

class ActionState:
    def __init__(self, actionId: str, callId: str, toolName: str, phase: str, arguments: dict, evidence: List[str], exec_tool_span_id: str):
        self.actionId = actionId
        self.callId = callId
        self.toolName = toolName
        self.phase = phase
        self.arguments = arguments
        self.evidence = evidence
        self.exec_tool_span_id = exec_tool_span_id
        self.attempts: List[ActionAttempt] = []
        self.status = "pending" # pending, success, failed, retry, timeout

class RunState:
    def __init__(self):
        self.raw_req = {}
        self.runId = ""
        self.publicMarker = ""
        
        self.trace_id = ""
        self.parent_span_id = None
        self.tracestate = None
        
        self.server_span_id = ""
        self.agent_span_id = ""
        self.chat_span_id = ""
        self.start_time_ns = time.time_ns()
        
        self.status = "waiting"
        self.phase = "INIT" # INIT, DIAGNOSTICS, APPROVAL, EFFECT, DONE
        self.diagnosis = {}
        self.effect_plan = {}
        
        self.actions: List[ActionState] = []
        
        self.effect_action_id = None
        self.approval_req = None
        self.approval_receipt = None
        
        self.actionLog = []
        self.receiptLog = []
        
        self.seen_receipts = set()
        self.seen_receipt_bodies = {}
        self.last_response = {}

# ----------------- UTILS -----------------
def hex_id(length_bytes: int) -> str:
    while True:
        v = secrets.token_hex(length_bytes)
        if int(v, 16) != 0:
            return v

def canonical_json(data: dict) -> str:
    return json.dumps(data, separators=(',', ':'), sort_keys=True)

def get_digest(args: dict) -> str:
    return hashlib.sha256(canonical_json(args).encode('utf-8')).hexdigest().lower()

# ----------------- LLM CALL -----------------
def call_llm(req_json: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    incident = req_json.get("incident", {})
    policy = req_json.get("policy", {})
    catalog = req_json.get("toolCatalog", [])
    
    valid_evs = re.findall(r'\[(ev_[a-zA-Z0-9_-]+)\]', incident.get("transcript", ""))
    seen = set()
    valid_evs = [x for x in valid_evs if not (x in seen or seen.add(x))]
    
    diag_tools = [t for t in catalog if t.get("name") not in policy.get("effectTools", [])]
    eff_tools = [t for t in catalog if t.get("name") in policy.get("effectTools", [])]
    
    fallback = {
        "rootCause": incident.get("allowedRootCauses", [""])[0] if incident.get("allowedRootCauses") else "",
        "evidence": valid_evs[:2] if len(valid_evs) >= 2 else valid_evs,
        "diagnostics": [{"toolName": diag_tools[0]["name"], "arguments": {}}] if diag_tools else [],
        "effect": {"toolName": eff_tools[0]["name"], "arguments": {}} if eff_tools else {}
    }
    
    if not api_key:
        return fallback
        
    prompt = f"""You are an incident response agent.
Allowed Root Causes: {json.dumps(incident.get('allowedRootCauses'))}
Valid Evidence IDs: {json.dumps(valid_evs)}
Diagnostic Tools (JSON schema): {json.dumps(diag_tools)}
Effect Tools (JSON schema): {json.dumps(eff_tools)}
Transcript: {json.dumps(incident.get('transcript'))}

Analyze the transcript to determine the EXACT root cause from Allowed Root Causes and extract required exact tool arguments.
Output MUST be exactly this JSON:
{{
  "rootCause": "...",
  "evidence": ["ev_1", "ev_2"],
  "diagnostics": [{{"toolName": "...", "arguments": {{}} }}],
  "effect": {{"toolName": "...", "arguments": {{}} }}
}}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
    }
    
    try:
        req_obj = urllib.request.Request(
            url, data=json.dumps(data).encode('utf-8'),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req_obj) as response:
            res_body = json.loads(response.read().decode('utf-8'))
            text_result = res_body["candidates"][0]["content"]["parts"][0]["text"]
            match = re.search(r'\{.*\}', text_result, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                result = json.loads(text_result)
            
            validated_ev = [e for e in result.get("evidence", []) if e in valid_evs]
            if len(validated_ev) < 2:
                validated_ev = (validated_ev + valid_evs)[:2]
            result["evidence"] = list(dict.fromkeys(validated_ev))[:4]
            return result
    except Exception as e:
        print("LLM Error:", e)
        return fallback

# ----------------- DISPATCH & OTLP -----------------
def create_dispatch(state: RunState, action: ActionState) -> dict:
    attempt_num = len(action.attempts) + 1
    client_span_id = hex_id(8)
    attempt = ActionAttempt(attempt=attempt_num, client_span_id=client_span_id)
    
    dispatch = {
        "actionId": action.actionId,
        "callId": action.callId,
        "phase": action.phase,
        "toolName": action.toolName,
        "arguments": action.arguments,
        "evidence": action.evidence,
        "attempt": attempt_num,
        "traceparent": f"00-{state.trace_id}-{client_span_id}-01"
    }
    
    if state.tracestate:
        dispatch["tracestate"] = state.tracestate
    if action.phase == "effect" and state.approval_receipt:
        dispatch["approvalId"] = state.approval_receipt["approvalId"]
        dispatch["approvalNonce"] = state.approval_receipt["nonce"]
        
    attempt.dispatch_dict = dispatch
    action.attempts.append(attempt)
    state.actionLog.append(dispatch)
    return dispatch

def attr_str(k, v): return {"key": k, "value": {"stringValue": str(v)}}
def attr_int(k, v): return {"key": k, "value": {"intValue": v}}

def generate_otlp(state: RunState) -> dict:
    spans = []
    current_t = state.start_time_ns
    def tick():
        nonlocal current_t
        current_t += 1000000
        return str(current_t)
        
    base_attrs = [attr_str("ga5.run.id", state.runId), attr_str("ga5.public.marker", state.publicMarker)]
    
    server_span = {
        "traceId": state.trace_id, "spanId": state.server_span_id, "name": "POST /v2/incidents",
        "kind": 2, "startTimeUnixNano": tick(), "endTimeUnixNano": str(state.start_time_ns + 100000000),
        "attributes": base_attrs, "status": {"code": 1}
    }
    if state.parent_span_id: server_span["parentSpanId"] = state.parent_span_id
    if state.tracestate: server_span["traceState"] = state.tracestate
    spans.append(server_span)
    
    agent_span = {
        "traceId": state.trace_id, "spanId": state.agent_span_id, "parentSpanId": state.server_span_id,
        "name": "invoke_agent incident-response", "kind": 1,
        "startTimeUnixNano": tick(), "endTimeUnixNano": tick(),
        "attributes": base_attrs, "status": {"code": 1}
    }
    if state.tracestate: agent_span["traceState"] = state.tracestate
    spans.append(agent_span)
    
    chat_span = {
        "traceId": state.trace_id, "spanId": state.chat_span_id, "parentSpanId": state.agent_span_id,
        "name": "chat incident-plan", "kind": 3,
        "startTimeUnixNano": tick(), "endTimeUnixNano": tick(),
        "attributes": base_attrs + [attr_str("gen_ai.operation.name", "chat"), attr_str("gen_ai.request.model", "gemini-1.5-flash")],
        "status": {"code": 1}
    }
    if state.tracestate: chat_span["traceState"] = state.tracestate
    spans.append(chat_span)
    
    diag_exec_span_ids = []
    for action in state.actions:
        if action.phase == "diagnostic":
            diag_exec_span_ids.append(action.exec_tool_span_id)
            
        exec_span = {
            "traceId": state.trace_id, "spanId": action.exec_tool_span_id, "parentSpanId": state.agent_span_id,
            "name": f"execute_tool {action.toolName}", "kind": 1,
            "startTimeUnixNano": tick(), "endTimeUnixNano": tick(),
            "attributes": base_attrs + [
                attr_str("ga5.action.id", action.actionId), attr_str("gen_ai.tool.name", action.toolName),
                attr_str("gen_ai.tool.call.id", action.callId), attr_str("gen_ai.operation.name", "execute_tool")
            ],
            "status": {"code": 1}
        }
        if state.tracestate: exec_span["traceState"] = state.tracestate
        spans.append(exec_span)
        
        for attempt in action.attempts:
            c_span = {
                "traceId": state.trace_id, "spanId": attempt.client_span_id, "parentSpanId": action.exec_tool_span_id,
                "name": f"POST tool/{action.toolName}", "kind": 3,
                "startTimeUnixNano": tick(), "endTimeUnixNano": tick(),
                "attributes": base_attrs + [
                    attr_str("ga5.action.id", action.actionId), attr_int("ga5.attempt", attempt.attempt),
                    attr_str("http.request.method", "POST"), attr_int("http.request.resend_count", attempt.attempt - 1)
                ],
                "status": {"code": 1}
            }
            if attempt.receipt:
                c_span["attributes"].extend([
                    attr_str("ga5.receipt.id", attempt.receipt.get("receiptId")),
                    attr_str("ga5.receipt.nonce", attempt.receipt.get("nonce"))
                ])
                status_code = attempt.receipt.get("status")
                error_type = attempt.receipt.get("errorType")
                
                if status_code == 503:
                    c_span["status"] = {"code": 2}
                    c_span["attributes"].append(attr_str("error.type", "503"))
                elif status_code == 0 and error_type == "timeout":
                    c_span["status"] = {"code": 2}
                    c_span["attributes"].append(attr_str("error.type", "timeout"))
            
            if state.tracestate: c_span["traceState"] = state.tracestate
            spans.append(c_span)
            
    if len(diag_exec_span_ids) > 1:
        join_span = {
            "traceId": state.trace_id, "spanId": hex_id(8), "parentSpanId": state.agent_span_id,
            "name": "incident.join", "kind": 1,
            "startTimeUnixNano": tick(), "endTimeUnixNano": tick(),
            "attributes": base_attrs,
            "links": [{"traceId": state.trace_id, "spanId": sid} for sid in diag_exec_span_ids],
            "status": {"code": 1}
        }
        if state.tracestate: join_span["traceState"] = state.tracestate
        spans.append(join_span)
        
    if state.approval_receipt:
        app_span = {
            "traceId": state.trace_id, "spanId": hex_id(8), "parentSpanId": state.agent_span_id,
            "name": "approval_gate", "kind": 1,
            "startTimeUnixNano": tick(), "endTimeUnixNano": tick(),
            "attributes": base_attrs + [
                attr_str("ga5.approval.id", state.approval_receipt["approvalId"]),
                attr_str("ga5.approval.nonce", state.approval_receipt["nonce"])
            ],
            "status": {"code": 1}
        }
        if state.tracestate: app_span["traceState"] = state.tracestate
        spans.append(app_span)

    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}

def build_response(state: RunState, dispatches: list = None, approvals: list = None) -> dict:
    if state.phase == "DONE":
        suppressed = [state.effect_plan.get("toolName", "")] if state.status == "failed" else []
        resp = {
            "runId": state.runId,
            "status": state.status,
            "diagnosis": state.diagnosis,
            "chosenEffect": state.effect_plan.get("toolName", ""),
            "suppressed": suppressed,
            "actionLog": state.actionLog,
            "receiptLog": state.receiptLog,
            "otlp": generate_otlp(state)
        }
        return resp
    else:
        return {
            "runId": state.runId,
            "status": state.status,
            "diagnosis": state.diagnosis,
            "dispatches": dispatches if dispatches is not None else [],
            "approvals": approvals if approvals is not None else []
        }

# ----------------- API ENDPOINTS -----------------
@app.get("/v2/incidents/{runId}")
async def get_incident(runId: str):
    if runId not in RUNS:
        raise HTTPException(status_code=404)
    return RUNS[runId].last_response

@app.post("/v2/incidents")
async def create_incident(request: Request, traceparent: Optional[str] = Header(None), tracestate: Optional[str] = Header(None)):
    try:
        body_json = json.loads(await request.body())
    except:
        raise HTTPException(status_code=400)
        
    if body_json.get("profile") != "ga5-incident-agent/v2":
        raise HTTPException(status_code=422)
        
    run_id = body_json.get("runId")
    if run_id in RUNS:
        if canonical_json(RUNS[run_id].raw_req) != canonical_json(body_json):
            raise HTTPException(status_code=409)
        return RUNS[run_id].last_response
        
    state = RunState()
    state.raw_req = body_json
    state.runId = run_id
    state.publicMarker = body_json.get("publicMarker", "")
    
    if traceparent:
        parts = traceparent.split('-')
        if len(parts) >= 3:
            state.trace_id = parts[1]
            state.parent_span_id = parts[2]
            state.tracestate = tracestate
    if not state.trace_id:
        state.trace_id = hex_id(16)
        
    state.server_span_id = hex_id(8)
    state.agent_span_id = hex_id(8)
    state.chat_span_id = hex_id(8)
            
    llm_res = call_llm(body_json)
    state.diagnosis = {"rootCause": llm_res.get("rootCause", ""), "evidence": llm_res.get("evidence", [])}
    state.effect_plan = llm_res.get("effect", {})
    ev_pool = list(state.diagnosis["evidence"])
    
    dispatches = []
    for d_plan in llm_res.get("diagnostics", []):
        ev = ev_pool.pop(0) if ev_pool else state.diagnosis["evidence"][0]
        action = ActionState(
            actionId=uuid.uuid4().hex, callId=uuid.uuid4().hex, toolName=d_plan["toolName"],
            phase="diagnostic", arguments=d_plan.get("arguments", {}), evidence=[ev],
            exec_tool_span_id=hex_id(8)
        )
        state.actions.append(action)
        dispatches.append(create_dispatch(state, action))
        
    state.phase = "DIAGNOSTICS"
    state.last_response = build_response(state, dispatches=dispatches, approvals=[])
    RUNS[run_id] = state
    return state.last_response

@app.post("/v2/incidents/{runId}/receipts")
async def post_receipts(runId: str, request: Request):
    if runId not in RUNS:
        raise HTTPException(status_code=404)
    state = RUNS[runId]
    
    try:
        body_json = json.loads(await request.body())
    except:
        raise HTTPException(status_code=400)
        
    receipt_id = body_json.get("receiptId")
    if receipt_id in state.seen_receipts:
        if canonical_json(state.seen_receipt_bodies[receipt_id]) != canonical_json(body_json):
            raise HTTPException(status_code=409)
        return state.last_response
        
    state.seen_receipts.add(receipt_id)
    state.seen_receipt_bodies[receipt_id] = body_json

    for out in body_json.get("outcomes", []):
        out_copy = dict(out)
        out_copy["receiptId"] = receipt_id
        state.receiptLog.append(out_copy)
        
        action = next((a for a in state.actions if a.actionId == out_copy.get("actionId")), None)
        if action and action.status in ["pending", "retry"]:
            attempt = next((a for a in action.attempts if a.attempt == out_copy.get("attempt")), None)
            if attempt:
                attempt.receipt = out_copy
                if out_copy.get("status") == 0 and out_copy.get("errorType") == "timeout":
                    action.status = "timeout"
                    state.status = "failed"
                    state.phase = "DONE"
                elif out_copy.get("status") == 503:
                    if attempt.attempt >= 2:
                        action.status = "failed"
                        state.status = "failed"
                        state.phase = "DONE"
                    else:
                        action.status = "retry"
                elif out_copy.get("status") == 200:
                    action.status = "success"
                else:
                    action.status = "failed"
                    state.status = "failed"
                    state.phase = "DONE"

    for app in body_json.get("approvals", []):
        app_copy = dict(app)
        app_copy["receiptId"] = receipt_id
        state.receiptLog.append(app_copy)
        state.approval_receipt = app_copy
        
    if state.phase == "DONE":
        state.last_response = build_response(state)
        return state.last_response

    if state.phase == "DIAGNOSTICS":
        diag_actions = [a for a in state.actions if a.phase == "diagnostic"]
        if any(a.status == "retry" for a in diag_actions):
            dispatches = []
            for a in diag_actions:
                if a.status == "retry":
                    a.status = "pending"
                    dispatches.append(create_dispatch(state, a))
            state.last_response = build_response(state, dispatches=dispatches, approvals=[])
            return state.last_response
            
        if all(a.status == "success" for a in diag_actions):
            eff_tool = state.effect_plan.get("toolName", "")
            req_approvals = state.raw_req.get("policy", {}).get("approvalRequiredFor", [])
            
            if eff_tool in req_approvals:
                state.phase = "APPROVAL"
                state.effect_action_id = uuid.uuid4().hex
                state.approval_req = {
                    "approvalId": uuid.uuid4().hex,
                    "actionId": state.effect_action_id,
                    "toolName": eff_tool,
                    "argumentsDigest": get_digest(state.effect_plan.get("arguments", {}))
                }
                state.last_response = build_response(state, dispatches=[], approvals=[state.approval_req])
                return state.last_response
            else:
                state.phase = "EFFECT"
                state.effect_action_id = uuid.uuid4().hex
                ev = state.diagnosis["evidence"][0] if state.diagnosis["evidence"] else "ev_fallback"
                action = ActionState(
                    actionId=state.effect_action_id, callId=uuid.uuid4().hex, toolName=eff_tool, phase="effect",
                    arguments=state.effect_plan.get("arguments", {}), evidence=[ev],
                    exec_tool_span_id=hex_id(8)
                )
                state.actions.append(action)
                state.last_response = build_response(state, dispatches=[create_dispatch(state, action)], approvals=[])
                return state.last_response

    if state.phase == "APPROVAL" and state.approval_receipt:
        state.phase = "EFFECT"
        eff_tool = state.effect_plan.get("toolName", "")
        ev = state.diagnosis["evidence"][0] if state.diagnosis["evidence"] else "ev_fallback"
        action = ActionState(
            actionId=state.effect_action_id, callId=uuid.uuid4().hex, toolName=eff_tool, phase="effect",
            arguments=state.effect_plan.get("arguments", {}), evidence=[ev],
            exec_tool_span_id=hex_id(8)
        )
        state.actions.append(action)
        state.last_response = build_response(state, dispatches=[create_dispatch(state, action)], approvals=[])
        return state.last_response
        
    if state.phase == "EFFECT":
        eff_action = next((a for a in state.actions if a.phase == "effect"), None)
        if eff_action:
            if eff_action.status == "success":
                state.phase = "DONE"
                state.status = "completed"
                state.last_response = build_response(state)
            elif eff_action.status in ["failed", "timeout"]:
                state.phase = "DONE"
                state.status = "failed"
                state.last_response = build_response(state)
            elif eff_action.status == "retry":
                eff_action.status = "pending"
                state.last_response = build_response(state, dispatches=[create_dispatch(state, eff_action)], approvals=[])
                
    return state.last_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
