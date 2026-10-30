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
    def __init__(self, actionId, callId, toolName, phase, arguments, evidence, exec_tool_span_id):
        self.actionId = actionId
        self.callId = callId
        self.toolName = toolName
        self.phase = phase
        self.arguments = arguments
        self.evidence = evidence
        self.exec_tool_span_id = exec_tool_span_id
        self.attempts: List[ActionAttempt] = []
        self.status = "pending" # pending, success, failed, retry

class RunState:
    def __init__(self):
        self.raw_req = {}
        self.runId = ""
        self.publicMarker = ""
        self.trace_id = secrets.token_hex(16)
        self.server_span_id = secrets.token_hex(8)
        self.agent_span_id = secrets.token_hex(8)
        self.chat_span_id = secrets.token_hex(8)
        self.parent_span_id = None
        self.tracestate = None
        self.start_time_ns = time.time_ns()
        
        self.status = "waiting"
        self.phase = "INIT"
        self.diagnosis = {}
        self.effect_plan = {}
        self.ev_pool = []
        
        self.actions: List[ActionState] = []
        self.approval_req = None
        self.approval_receipt = None
        
        self.actionLog = []
        self.receiptLog = []
        
        self.seen_receipts = set()
        self.seen_receipt_bodies = {}
        self.last_response = {}

# ----------------- UTILS & LLM (GEMINI 1.5 FLASH) -----------------
def canonical_json(data: dict) -> str:
    return json.dumps(data, separators=(',', ':'), sort_keys=True)

def get_digest(args: dict) -> str:
    return hashlib.sha256(canonical_json(args).encode('utf-8')).hexdigest().lower()

def call_llm(req_json: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    incident = req_json.get("incident", {})
    policy = req_json.get("policy", {})
    catalog = req_json.get("toolCatalog", [])
    
    valid_evs = re.findall(r'\[(ev_[a-zA-Z0-9_-]+)\]', incident.get("transcript", ""))
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
        
    prompt = f"""You are an expert incident response AI. Find the root cause and required tools.
Incident: {incident.get('title')}
Severity: {incident.get('severity')}
Transcript: {incident.get('transcript')}

Allowed Root Causes: {json.dumps(incident.get('allowedRootCauses'))}
Valid Evidence IDs from transcript: {json.dumps(valid_evs)}

Diagnostic Tools Available (JSON schemas):
{json.dumps(diag_tools, indent=2)}

Effect Tools Available (JSON schemas):
{json.dumps(eff_tools, indent=2)}

Maximum Diagnostics Allowed: {policy.get('maximumDiagnostics', 3)}

Task:
1. Determine the exact root cause from the Allowed Root Causes.
2. Choose 2 to 4 Evidence IDs from the valid list that support this cause.
3. Select 1 to {policy.get('maximumDiagnostics', 3)} Diagnostic Tools to verify the issue. Provide exact arguments based on the tool schemas.
4. Select exactly 1 Effect Tool to fix the issue. Provide exact arguments based on the tool schemas.

Output strictly as a JSON object matching this exact schema:
{{
  "rootCause": "chosen cause string",
  "evidence": ["ev_...", "ev_..."],
  "diagnostics": [{{"toolName": "...", "arguments": {{}} }}],
  "effect": {{"toolName": "...", "arguments": {{}} }}
}}"""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
    }
    
    req_obj = urllib.request.Request(
        url, data=json.dumps(data).encode('utf-8'),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    
    try:
        with urllib.request.urlopen(req_obj) as response:
            res_body = json.loads(response.read().decode('utf-8'))
            text_result = res_body["candidates"][0]["content"]["parts"][0]["text"]
            # Safely extract JSON in case Gemini includes markdown backticks despite mimeType
            match = re.search(r'\{.*\}', text_result, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                result = json.loads(text_result)
            
            validated_ev = [e for e in result.get("evidence", []) if e in valid_evs]
            if len(validated_ev) < 2:
                validated_ev = (validated_ev + valid_evs)[:2]
            result["evidence"] = list(set(validated_ev))[:4]
            return result
    except Exception as e:
        print("LLM Error:", e)
        return fallback

def create_dispatch(state: RunState, action: ActionState) -> dict:
    attempt_obj = ActionAttempt(attempt=len(action.attempts) + 1, client_span_id=secrets.token_hex(8))
    dispatch = {
        "actionId": action.actionId,
        "callId": action.callId,
        "phase": action.phase,
        "toolName": action.toolName,
        "arguments": action.arguments,
        "evidence": action.evidence,
        "attempt": attempt_obj.attempt,
        "traceparent": f"00-{state.trace_id}-{attempt_obj.client_span_id}-01"
    }
    if state.tracestate:
        dispatch["tracestate"] = state.tracestate
    if action.phase == "effect" and state.approval_receipt:
        dispatch["approvalId"] = state.approval_receipt["approvalId"]
        dispatch["approvalNonce"] = state.approval_receipt["nonce"]
        
    attempt_obj.dispatch_dict = dispatch
    action.attempts.append(attempt_obj)
    state.actionLog.append(dispatch)
    return dispatch

# ----------------- OTLP GENERATION -----------------
def attr_str(k, v): return {"key": k, "value": {"stringValue": str(v)}}
def attr_int(k, v): return {"key": k, "value": {"intValue": v}}

def generate_otlp(state: RunState) -> dict:
    spans = []
    current_time = state.start_time_ns
    
    def tick():
        nonlocal current_time
        current_time += 1000000
        return str(current_time)
        
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
            "traceId": state.trace_id, "spanId": secrets.token_hex(8), "parentSpanId": state.agent_span_id,
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
            "traceId": state.trace_id, "spanId": secrets.token_hex(8), "parentSpanId": state.agent_span_id,
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

def build_final_response(state: RunState) -> dict:
    suppressed = [state.effect_plan["toolName"]] if state.status == "failed" else []
    return {
        "runId": state.runId,
        "status": state.status,
        "diagnosis": state.diagnosis,
        "chosenEffect": state.effect_plan.get("toolName", ""),
        "suppressed": suppressed,
        "actionLog": state.actionLog,
        "receiptLog": state.receiptLog,
        "otlp": generate_otlp(state),
        "dispatches": [],
        "approvals": []
    }

# ----------------- API ENDPOINTS -----------------
@app.get("/v2/incidents/{runId}")
async def get_incident(runId: str):
    if runId not in RUNS:
        raise HTTPException(404)
    return RUNS[runId].last_response

@app.post("/v2/incidents")
async def create_incident(request: Request, traceparent: Optional[str] = Header(None), tracestate: Optional[str] = Header(None)):
    try:
        body_json = json.loads(await request.body())
    except:
        raise HTTPException(400)
        
    if body_json.get("profile") != "ga5-incident-agent/v2":
        raise HTTPException(422)
        
    run_id = body_json.get("runId")
    if run_id in RUNS:
        if canonical_json(RUNS[run_id].raw_req) != canonical_json(body_json):
            raise HTTPException(409)
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
            
    llm_res = call_llm(body_json)
    state.diagnosis = {"rootCause": llm_res.get("rootCause"), "evidence": llm_res.get("evidence", [])}
    state.effect_plan = llm_res.get("effect", {})
    state.ev_pool = list(state.diagnosis["evidence"])
    
    dispatches = []
    for d_plan in llm_res.get("diagnostics", []):
        action = ActionState(
            actionId=str(uuid.uuid4()), callId=str(uuid.uuid4()), toolName=d_plan["toolName"],
            phase="diagnostic", arguments=d_plan.get("arguments", {}),
            evidence=[state.ev_pool.pop(0)] if state.ev_pool else [state.diagnosis["evidence"][0]],
            exec_tool_span_id=secrets.token_hex(8)
        )
        state.actions.append(action)
        dispatches.append(create_dispatch(state, action))
        
    state.phase = "DIAGNOSTICS"
    resp = {
        "runId": state.runId, "status": "waiting",
        "diagnosis": state.diagnosis, "dispatches": dispatches, "approvals": []
    }
    state.last_response = resp
    RUNS[run_id] = state
    return resp

@app.post("/v2/incidents/{runId}/receipts")
async def post_receipts(runId: str, request: Request):
    if runId not in RUNS:
        raise HTTPException(404)
    state = RUNS[runId]
    try:
        body_json = json.loads(await request.body())
    except:
        raise HTTPException(400)
        
    receipt_id = body_json.get("receiptId")
    if receipt_id in state.seen_receipts:
        if canonical_json(state.seen_receipt_bodies[receipt_id]) != canonical_json(body_json):
            raise HTTPException(409)
        return state.last_response
        
    state.seen_receipts.add(receipt_id)
    state.seen_receipt_bodies[receipt_id] = body_json

    for out in body_json.get("outcomes", []):
        out_copy = dict(out)
        out_copy["receiptId"] = receipt_id
        state.receiptLog.append(out_copy)
        
        action = next((a for a in state.actions if a.actionId == out_copy.get("actionId")), None)
        if action:
            attempt = next((a for a in action.attempts if a.attempt == out_copy.get("attempt")), None)
            if attempt:
                attempt.receipt = out_copy
                if out_copy.get("status") == 0 and out_copy.get("errorType") == "timeout":
                    action.status = "failed"
                    state.status = "failed"
                    state.phase = "DONE"
                elif out_copy.get("status") == 503:
                    if attempt.attempt >= 2: # Max 1 retry rule
                        action.status = "failed"
                        state.status = "failed"
                        state.phase = "DONE"
                    else:
                        action.status = "retry"
                elif out_copy.get("status") == 200:
                    action.status = "success"

    for app in body_json.get("approvals", []):
        app_copy = dict(app)
        app_copy["receiptId"] = receipt_id
        state.receiptLog.append(app_copy)
        state.approval_receipt = app_copy
        
    if state.status == "failed":
        state.last_response = build_final_response(state)
        return state.last_response

    if state.phase == "DIAGNOSTICS":
        diag_actions = [a for a in state.actions if a.phase == "diagnostic"]
        if any(a.status == "retry" for a in diag_actions):
            dispatches = []
            for a in diag_actions:
                if a.status == "retry":
                    a.status = "pending"
                    dispatches.append(create_dispatch(state, a))
            state.last_response = {"runId": state.runId, "status": "waiting", "diagnosis": state.diagnosis, "dispatches": dispatches, "approvals": []}
            return state.last_response
            
        if all(a.status == "success" for a in diag_actions):
            eff_tool = state.effect_plan.get("toolName")
            req_approvals = state.raw_req.get("policy", {}).get("approvalRequiredFor", [])
            
            if eff_tool in req_approvals:
                state.phase = "APPROVAL"
                state.approval_req = {
                    "approvalId": str(uuid.uuid4()), "actionId": str(uuid.uuid4()),
                    "toolName": eff_tool, "argumentsDigest": get_digest(state.effect_plan.get("arguments", {}))
                }
                state.last_response = {"status": "waiting", "dispatches": [], "approvals": [state.approval_req]}
                return state.last_response
            else:
                state.phase = "EFFECT"
                action = ActionState(
                    actionId=str(uuid.uuid4()), callId=str(uuid.uuid4()), toolName=eff_tool, phase="effect",
                    arguments=state.effect_plan.get("arguments", {}),
                    evidence=[state.ev_pool.pop(0)] if state.ev_pool else [state.diagnosis["evidence"][0]],
                    exec_tool_span_id=secrets.token_hex(8)
                )
                state.actions.append(action)
                state.last_response = {"status": "waiting", "dispatches": [create_dispatch(state, action)], "approvals": []}
                return state.last_response

    if state.phase == "APPROVAL" and state.approval_receipt:
        state.phase = "EFFECT"
        action = ActionState(
            actionId=state.approval_req["actionId"], callId=str(uuid.uuid4()), toolName=state.effect_plan.get("toolName"), phase="effect",
            arguments=state.effect_plan.get("arguments", {}),
            evidence=[state.ev_pool.pop(0)] if state.ev_pool else [state.diagnosis["evidence"][0]],
            exec_tool_span_id=secrets.token_hex(8)
        )
        state.actions.append(action)
        state.last_response = {"status": "waiting", "dispatches": [create_dispatch(state, action)], "approvals": []}
        return state.last_response
        
    if state.phase == "EFFECT":
        eff_action = next((a for a in state.actions if a.phase == "effect"), None)
        if eff_action:
            if eff_action.status == "success" or eff_action.status == "failed":
                state.phase = "DONE"
                state.status = "completed" if eff_action.status == "success" else "failed"
                state.last_response = build_final_response(state)
            elif eff_action.status == "retry":
                eff_action.status = "pending"
                state.last_response = {"runId": state.runId, "status": "waiting", "dispatches": [create_dispatch(state, eff_action)], "approvals": []}
                
    return state.last_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
