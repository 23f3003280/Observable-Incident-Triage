import os
import json
import uuid
import hashlib
import time
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI()

# In-memory storage for state and idempotency
DB: Dict[str, Dict[str, Any]] = {}

# Make sure to set your OpenAI (or compatible) API key
# Example: export OPENAI_API_KEY="sk-..."
LLM_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# --- Helper Functions ---

def gen_hex(length: int) -> str:
    return os.urandom(length // 2).hex()

def hash_dict(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def extract_trace_context(traceparent: str):
    # format: 00-traceid-spanid-01
    if traceparent and len(traceparent) == 55 and traceparent.startswith("00-"):
        parts = traceparent.split("-")
        return parts[1], parts[2]
    return gen_hex(32), gen_hex(16)

def ask_llm(incident: dict, allowed_causes: list, catalog: list) -> dict:
    prompt = f"""
You are an incident responder triage agent. 
Incident Transcript:
{incident.get('transcript', '')}

Allowed Root Causes: {allowed_causes}
Tool Catalog: {json.dumps(catalog)}

Task:
1. Select 1 rootCause from Allowed Root Causes.
2. Select 2-4 evidence IDs (e.g. [ev_xyz]) from the transcript that support this.
3. Select 1-3 diagnostic tools from the catalog to confirm this (provide arguments).
4. Select 1 effect tool from the catalog to fix this (provide arguments).

Return ONLY valid JSON matching this schema:
{{
  "rootCause": "...",
  "evidence": ["ev_1", "ev_2"],
  "diagnostics": [{{"toolName": "...", "arguments": {{}} }}],
  "effect": {{"toolName": "...", "arguments": {{}} }}
}}
"""
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    
    # Use synchronous HTTPX for simplicity in this single-file async script wrapper
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(LLM_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])

def generate_otlp_trace(run_state: dict) -> dict:
    run_id = run_state["runId"]
    pub_marker = run_state["publicMarker"]
    trace_id = run_state["trace_id"]
    
    def make_attr(k, v):
        if isinstance(v, int): return {"key": k, "value": {"intValue": str(v)}}
        return {"key": k, "value": {"stringValue": str(v)}}

    base_attrs = [make_attr("ga5.run.id", run_id), make_attr("ga5.public.marker", pub_marker)]
    
    spans = []
    
    # 1. SERVER span
    server_span_id = run_state["server_span_id"]
    spans.append({
        "traceId": trace_id, "spanId": server_span_id,
        "name": "POST /v2/incidents", "kind": 2, "attributes": base_attrs
    })
    
    # 2. INTERNAL invoke_agent
    agent_span_id = gen_hex(16)
    spans.append({
        "traceId": trace_id, "spanId": agent_span_id, "parentSpanId": server_span_id,
        "name": "invoke_agent", "kind": 1, "attributes": base_attrs
    })
    
    # 3. CLIENT chat incident-plan
    spans.append({
        "traceId": trace_id, "spanId": gen_hex(16), "parentSpanId": agent_span_id,
        "name": "chat", "kind": 3, 
        "attributes": base_attrs + [
            make_attr("gen_ai.operation.name", "chat"),
            make_attr("gen_ai.request.model", "gpt-4o-mini")
        ]
    })
    
    diagnostic_exec_span_ids = []
    
    # Iterate through action log to build tool spans
    action_map = {} # actionId -> execute_tool spanId
    
    for action in run_state["actionLog"]:
        action_id = action["actionId"]
        if action_id not in action_map:
            exec_span_id = gen_hex(16)
            action_map[action_id] = exec_span_id
            if action["phase"] == "diagnostic":
                diagnostic_exec_span_ids.append(exec_span_id)
            spans.append({
                "traceId": trace_id, "spanId": exec_span_id, "parentSpanId": agent_span_id,
                "name": "execute_tool", "kind": 1,
                "attributes": base_attrs + [
                    make_attr("ga5.action.id", action_id),
                    make_attr("gen_ai.tool.name", action["toolName"]),
                    make_attr("gen_ai.tool.call.id", action["callId"]),
                    make_attr("gen_ai.operation.name", "execute_tool")
                ]
            })
            
        client_span_id = action["traceparent"].split("-")[2]
        client_attrs = base_attrs + [
            make_attr("ga5.action.id", action_id),
            make_attr("ga5.attempt", action["attempt"]),
            make_attr("http.request.method", "POST"),
            make_attr("http.request.resend_count", action["attempt"] - 1)
        ]
        
        # Attach receipt data if we have it
        receipt = next((r for r in run_state["receiptLog"] if r.get("actionId") == action_id and r.get("attempt") == action["attempt"]), None)
        status_info = {}
        if receipt:
            client_attrs.append(make_attr("ga5.receipt.id", receipt["receiptId"]))
            client_attrs.append(make_attr("ga5.receipt.nonce", receipt["nonce"]))
            if receipt["status"] == 503:
                status_info = {"code": 2} # Error
                client_attrs.append(make_attr("error.type", "503"))
            elif receipt["status"] == 0 or receipt.get("errorType") == "timeout":
                status_info = {"code": 2}
                client_attrs.append(make_attr("error.type", "timeout"))
        
        spans.append({
            "traceId": trace_id, "spanId": client_span_id, "parentSpanId": action_map[action_id],
            "name": f"POST tool/{action['toolName']}", "kind": 3,
            "attributes": client_attrs,
            "status": status_info
        })
        
    # incident.join span
    if diagnostic_exec_span_ids:
        spans.append({
            "traceId": trace_id, "spanId": gen_hex(16), "parentSpanId": agent_span_id,
            "name": "incident.join", "kind": 1,
            "attributes": base_attrs,
            "links": [{"traceId": trace_id, "spanId": ds} for ds in diagnostic_exec_span_ids]
        })
        
    # approval_gate span
    approval_receipt = next((r for r in run_state["receiptLog"] if "approvalId" in r), None)
    if approval_receipt or run_state.get("approval_requested"):
        gate_attrs = base_attrs.copy()
        if approval_receipt:
            gate_attrs.append(make_attr("ga5.approval.id", approval_receipt["approvalId"]))
            gate_attrs.append(make_attr("ga5.approval.nonce", approval_receipt["nonce"]))
            
        spans.append({
            "traceId": trace_id, "spanId": gen_hex(16), "parentSpanId": agent_span_id,
            "name": "approval_gate", "kind": 1, "attributes": gate_attrs
        })

    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}

# --- Endpoints ---

@app.post("/v2/incidents")
async def create_incident(request: Request):
    body = await request.json()
    run_id = body.get("runId")
    
    # Idempotency and conflict check
    req_hash = hash_dict(body)
    if run_id in DB:
        if DB[run_id]["req_hash"] != req_hash:
            raise HTTPException(status_code=409, detail="Changed content")
        return DB[run_id]["current_response"]

    # Parse trace context
    trace_id, server_span_id = extract_trace_context(request.headers.get("traceparent", ""))
    
    # Use LLM to diagnose
    try:
        llm_res = ask_llm(body["incident"], body["incident"]["allowedRootCauses"], body["toolCatalog"])
    except Exception as e:
        raise HTTPException(status_code=500, detail="LLM failure")

    # Construct dispatches
    dispatches = []
    for idx, diag in enumerate(llm_res.get("diagnostics", [])):
        action_id = f"act_diag_{gen_hex(8)}"
        call_id = f"call_{gen_hex(8)}"
        client_span_id = gen_hex(16)
        
        # Ensure we cite at least one evidence per requirement
        ev = [llm_res["evidence"][0]] if llm_res.get("evidence") else []
        
        dispatches.append({
            "actionId": action_id, "callId": call_id,
            "phase": "diagnostic", "toolName": diag["toolName"],
            "arguments": diag.get("arguments", {}), "evidence": ev,
            "attempt": 1,
            "traceparent": f"00-{trace_id}-{client_span_id}-01"
        })

    # Prepare State
    state = {
        "req_hash": req_hash,
        "runId": run_id,
        "publicMarker": body.get("publicMarker"),
        "trace_id": trace_id,
        "server_span_id": server_span_id,
        "policy": body.get("policy", {}),
        "diagnosis": {
            "rootCause": llm_res.get("rootCause"),
            "evidence": llm_res.get("evidence", [])[:4]
        },
        "planned_effect": llm_res.get("effect", {}),
        "status": "waiting",
        "actionLog": list(dispatches),
        "receiptLog": [],
        "pending_diagnostics": {d["actionId"]: d for d in dispatches},
        "approval_requested": False,
        "completed": False,
        "suppressed": []
    }
    
    response_payload = {
        "runId": run_id,
        "status": "waiting",
        "diagnosis": state["diagnosis"],
        "dispatches": dispatches,
        "approvals": []
    }
    
    state["current_response"] = response_payload
    DB[run_id] = state
    return response_payload

@app.post("/v2/incidents/{runId}/receipts")
async def process_receipts(runId: str, request: Request):
    if runId not in DB:
        raise HTTPException(status_code=404, detail="Run not found")
        
    body = await request.json()
    receipt_hash = hash_dict(body)
    state = DB[runId]
    
    # Check if receipt is already processed identically (Idempotency)
    if "receipts_seen" not in state: state["receipts_seen"] = {}
    if body.get("receiptId") in state["receipts_seen"]:
        if state["receipts_seen"][body["receiptId"]] != receipt_hash:
            raise HTTPException(status_code=409, detail="Changed content")
        return state["current_response"]
        
    state["receipts_seen"][body["receiptId"]] = receipt_hash

    new_dispatches = []
    new_approvals = []
    
    # Handle Outcomes (Diagnostics / Effects)
    if "outcomes" in body:
        for out in body["outcomes"]:
            out["receiptId"] = body["receiptId"]
            state["receiptLog"].append(out)
            act_id = out["actionId"]
            
            # Diagnostic Retry Logic
            if act_id in state["pending_diagnostics"]:
                if out.get("status") == 503 and out.get("attempt", 1) == 1:
                    orig_d = state["pending_diagnostics"][act_id]
                    retry_d = dict(orig_d)
                    retry_d["attempt"] = 2
                    retry_span = gen_hex(16)
                    retry_d["traceparent"] = f"00-{state['trace_id']}-{retry_span}-01"
                    state["actionLog"].append(retry_d)
                    new_dispatches.append(retry_d)
                elif out.get("status") == 0 or out.get("errorType") == "timeout":
                    # Fails the diagnostic, suppress effects
                    del state["pending_diagnostics"][act_id]
                    state["completed"] = True
                    state["status"] = "failed"
                else:
                    # Success
                    del state["pending_diagnostics"][act_id]

            # Effect completion
            if "pending_effect" in state and act_id == state["pending_effect"]["actionId"]:
                state["completed"] = True
                state["status"] = "completed"

    # Handle Approvals
    if "approvals" in body:
        for app in body["approvals"]:
            app["receiptId"] = body["receiptId"]
            state["receiptLog"].append(app)
            if app.get("decision") == "approved":
                # Dispatch the effect
                eff = state["planned_effect"]
                eff_act_id = state.get("reserved_effect_actionId", f"act_eff_{gen_hex(8)}")
                eff_dispatch = {
                    "actionId": eff_act_id,
                    "callId": f"call_{gen_hex(8)}",
                    "phase": "effect",
                    "toolName": eff["toolName"],
                    "arguments": eff.get("arguments", {}),
                    "evidence": state["diagnosis"]["evidence"][:1],
                    "attempt": 1,
                    "traceparent": f"00-{state['trace_id']}-{gen_hex(16)}-01"
                }
                state["actionLog"].append(eff_dispatch)
                new_dispatches.append(eff_dispatch)
                state["pending_effect"] = eff_dispatch

    # State transition logic if we aren't completed and diagnostics finished
    if not state["completed"] and len(state["pending_diagnostics"]) == 0 and not new_dispatches and not state.get("pending_effect"):
        eff = state["planned_effect"]
        policy = state["policy"]
        req_approval = eff["toolName"] in policy.get("approvalRequiredFor", [])
        
        eff_act_id = f"act_eff_{gen_hex(8)}"
        state["reserved_effect_actionId"] = eff_act_id
        
        if req_approval and not state["approval_requested"]:
            # Need approval
            arg_digest = hashlib.sha256(json.dumps(eff.get("arguments", {}), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            app_req = {
                "approvalId": f"app_{gen_hex(8)}",
                "actionId": eff_act_id,
                "toolName": eff["toolName"],
                "argumentsDigest": arg_digest
            }
            new_approvals.append(app_req)
            state["approval_requested"] = True
        elif not req_approval:
            # Dispatch effect immediately
            eff_dispatch = {
                "actionId": eff_act_id,
                "callId": f"call_{gen_hex(8)}",
                "phase": "effect",
                "toolName": eff["toolName"],
                "arguments": eff.get("arguments", {}),
                "evidence": state["diagnosis"]["evidence"][:1],
                "attempt": 1,
                "traceparent": f"00-{state['trace_id']}-{gen_hex(16)}-01"
            }
            state["actionLog"].append(eff_dispatch)
            new_dispatches.append(eff_dispatch)
            state["pending_effect"] = eff_dispatch

    # Build Response
    if state["completed"]:
        response_payload = {
            "runId": runId,
            "status": state["status"],
            "diagnosis": state["diagnosis"],
            "chosenEffect": state["planned_effect"].get("toolName"),
            "suppressed": state["suppressed"],
            "actionLog": state["actionLog"],
            "receiptLog": state["receiptLog"],
            "otlp": generate_otlp_trace(state)
        }
    else:
        response_payload = {
            "runId": runId,
            "status": "waiting",
            "diagnosis": state["diagnosis"],
            "dispatches": new_dispatches,
            "approvals": new_approvals
        }
        
    state["current_response"] = response_payload
    return response_payload

@app.get("/v2/incidents/{runId}")
async def get_incident(runId: str):
    if runId not in DB:
        raise HTTPException(status_code=404, detail="Run not found")
    return DB[runId]["current_response"]
