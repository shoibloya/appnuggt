import os
import re
import io
import time
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

import streamlit as st

# LangChain / OpenAI & Tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate, PromptTemplate,
    SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
)
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults

# -------------------------- App Config --------------------------
st.set_page_config(page_title="BUTA Beachhead Finder (Parallel Mode)", layout="wide")

# Secrets/env
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["apiKey"]
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets["tapiKey"]

# -------------------------- LLM & Tools --------------------------
LLM = ChatOpenAI(model="gpt-4.1", temperature=0)

def make_search_executor() -> AgentExecutor:
    """Create a fresh agent executor (thread-safe usage)."""
    tavily_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=True,
        search_depth="advanced",
    )
    tools = [tavily_tool]
    search_system = """
You are a precise market research analyst. For each search query:
- Use the Tavily tool to gather recent, credible facts.
- Return 3–6 short bullets.
- Each bullet MUST end with a source link in parentheses.
- Prefer primary data, reputable news, analyst/industry reports, government/NGO stats.
- If evidence is mixed or uncertain, say so explicitly and still cite a source.
- No fluff. Facts only.
"""
    search_prompt = ChatPromptTemplate(
        input_variables=["agent_scratchpad","input"],
        messages=[
            SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=search_system)),
            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="{input}")),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ],
    )
    agent = create_openai_tools_agent(LLM, tools, search_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

# -------------------------- Core Prompts (UNCHANGED) --------------------------
PLANNER_SCHEMA = JsonOutputParser()
PLANNER_PROMPT = """
You are an iterative beachhead-finder using the BUTA framework:

B = Budget (ability/willingness to pay),
U = Urgency (near-term reason to adopt),
T = Top-3 Fit (solution plausibly among top options),
A = Access (clear way to reach/convert the segment).

TASK:
Given the idea below and any previous attempts & findings, propose ONE best next candidate segment to investigate now, and a tight Tavily search plan (2 specific queries for each of B, U, T, A). The candidate must be concrete (e.g., “Dental clinics with 2–5 chairs in California using Open Dental”).

Stop conditions:
- If a prior candidate already met strong BUTA evidence (≥2 dimensions High and none Low), you may return {"decision":"stop","reason":"good_fit_identified"}.
- If after multiple attempts it is unlikely to find a good fit, return {"decision":"stop","reason":"exhausted"} and suggest the best available candidate to present anyway.

Return STRICT JSON ONLY:

{
  "decision": "continue | stop",
  "reason": "short reason",
  "candidate": "Precise segment name",
  "why_this": "Brief rationale",
  "plan": {
    "Budget": ["query1", "query2"],
    "Urgency": ["query1", "query2"],
    "Top-3 Fit": ["query1", "query2"],
    "Access": ["query1", "query2"]
  }
}
"""

EVALUATOR_SCHEMA = JsonOutputParser()
EVALUATOR_PROMPT = """
Evaluate the candidate with the research notes provided. Score each BUTA dimension:

- rating: "High" | "Medium" | "Low"
- evidence: 2–5 short justifications derived from the research bullets (quote/summarize)
- sources: 2–6 URLs that appear in the notes (no invented links)

Then decide:
- "good_fit" if evidence indicates at least two "High" ratings and none "Low"
- else "not_yet"

Return STRICT JSON ONLY:

{
  "BUTA": {
    "budget": {"rating":"High|Medium|Low","evidence":["..."],"sources":["..."]},
    "urgency":{"rating":"High|Medium|Low","evidence":["..."],"sources":["..."]},
    "top_fit":{"rating":"High|Medium|Low","evidence":["..."],"sources":["..."]},
    "access":{"rating":"High|Medium|Low","evidence":["..."],"sources":["..."]}
  },
  "decision":"good_fit | not_yet",
  "notes":"short commentary"
}
"""

FINAL_WRITER_PROMPT = """
Write a single, clean BUTA report that includes:
1) A concise narrative summary of the best beachhead(s) with ratings.
2) Evidence bullets (2–4 per dimension) with source links inline.
3) Go-to-market hypothesis, Key risks, Immediate validation steps.

IMPORTANT: After the narrative, append a complete “Research Appendix” that includes EVERY Tavily query and ALL bullets exactly as gathered (do not omit anything).

Idea overview:
{idea_overview}

Recommended beachheads with evaluations:
{evaluations}

All research notes (full text):
{full_research_notes}

Produce one readable report (no code or JSON).
"""

# -------------------------- New Helper Prompts (non-core) --------------------------
VALIDATOR_SCHEMA = JsonOutputParser()
VALIDATOR_PROMPT = """
You validate whether a proposed beachhead segment is concrete and specific.

A good beachhead includes:
- WHO (industry/role/company type/size),
- WHERE (geo, if applicable),
- CONTEXT/STACK (tooling/platform/regulatory or other qualifying attributes).

Return STRICT JSON ONLY:
{
  "valid": true|false,
  "reasons": ["short reason 1", "short reason 2"],
  "improve": "rewrite as a crisper, concrete segment"
}
"""

FIXED_PLANNER_SCHEMA = JsonOutputParser()
FIXED_PLANNER_PROMPT = """
Given the overall idea and a FIXED candidate segment, produce a Tavily query plan with 2 queries per BUTA dimension.
Return STRICT JSON ONLY:

{
  "candidate": "exact candidate echoed",
  "plan": {
    "Budget": ["q1","q2"],
    "Urgency": ["q1","q2"],
    "Top-3 Fit": ["q1","q2"],
    "Access": ["q1","q2"]
  }
}
"""

# -------------------------- Sidebar Inputs --------------------------
st.sidebar.header("Your Idea")

problem = st.sidebar.text_area("Problem / Job-to-be-Done (required)")
solution = st.sidebar.text_area("Your Solution & Differentiators (required)")

industry = st.sidebar.text_input("Industry / Category (optional)")
target_geos = st.sidebar.text_input("Target Geography (optional)")
buyer_roles = st.sidebar.text_input("Buyer / User Roles (optional)")
price_point = st.sidebar.text_input("Expected Price Point (optional)")
evidence = st.sidebar.text_area("Evidence/Traction (optional)")
competitors = st.sidebar.text_area("Known Alternatives/Competitors (optional)")
constraints = st.sidebar.text_area("Constraints (optional)")

st.sidebar.markdown("---")
# Multiline beachhead input (per request)
student_beachhead = st.sidebar.text_area(
    "Your Proposed Beachhead (optional)",
    placeholder="e.g., Dental clinics with 2–5 chairs in California using Open Dental",
    height=90,
)
# Alternative is always ON (no checkbox)

start = st.sidebar.button("Run BUTA", type="primary")

# -------------------------- Helpers --------------------------
def normalize_urls(text: str) -> List[str]:
    return list(set(re.findall(r'https?://[\w\.-/%\?\=&\+#]+', text)))

def run_tavily_query(executor: AgentExecutor, q: str) -> str:
    try:
        out = executor.invoke({"input": q})["output"]
    except Exception as e:
        out = f"- Search error for '{q}': {e}"
    return out

def score_eval(ev: Dict[str,Any]) -> int:
    rank = {"High":2,"Medium":1,"Low":0}
    return sum(rank[ev["BUTA"][k]["rating"]] for k in ["budget","urgency","top_fit","access"])

def research_dimension(segment: str, dim: str, queries: List[str],
                       emit, counters: Dict[str,int], total_steps: int,
                       executor: AgentExecutor) -> str:
    notes = f"\n=== {segment} :: {dim} ===\n"
    for q in queries[:2]:
        counters["done"] += 1
        emit({"type": "step", "done": counters["done"], "total": total_steps,
              "text": f"[{segment}] Searching: '{q}'"})
        bullets = run_tavily_query(executor, q)
        counters["done"] += 1
        emit({"type": "step", "done": counters["done"], "total": total_steps,
              "text": f"[{segment}] Finished: '{q}'"})
        notes += f"Query: {q}\n"
        notes += bullets + "\n"
    return notes

# PDF builder (ReportLab if available; fallback to Markdown)
def build_pdf_from_text(text: str) -> Optional[bytes]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.units import mm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
        styles = getSampleStyleSheet()
        normal = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            alignment=TA_LEFT,
            fontName='Helvetica',
            fontSize=10,
            leading=14,
        )
        flow = []
        for line in text.split("\n"):
            safe = (line
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
            flow.append(Paragraph(safe, normal))
            flow.append(Spacer(1, 3))
        doc.build(flow)
        pdf_bytes = buf.getvalue()
        buf.close()
        return pdf_bytes
    except Exception:
        return None

# -------------------------- UI: Title & Pre-flight --------------------------
st.title("BUTA Beachhead Finder — Parallel")

if not start:
    st.info("Fill in Problem and Solution, optionally add your proposed beachhead, then click **Run BUTA**.")
    st.stop()

if not problem.strip() or not solution.strip():
    st.error("Please fill in both required fields: Problem and Solution.")
    st.stop()

# -------------------------- Idea Overview --------------------------
idea_overview_lines = [
    f"Problem: {problem}",
    f"Solution: {solution}",
]
if industry: idea_overview_lines.append(f"Industry: {industry}")
if target_geos: idea_overview_lines.append(f"Geography: {target_geos}")
if buyer_roles: idea_overview_lines.append(f"Buyer/User Roles: {buyer_roles}")
if price_point: idea_overview_lines.append(f"Price Point: {price_point}")
if evidence: idea_overview_lines.append(f"Evidence/Traction: {evidence}")
if competitors: idea_overview_lines.append(f"Competitors: {competitors}")
if constraints: idea_overview_lines.append(f"Constraints: {constraints}")

# -------------------------- Layout: two columns for parallel tracks --------------------------
left_col, right_col = st.columns(2)

# Build progress UIs (main thread only)
student_box = left_col.container()
student_prog = student_box.progress(0.0, text="Your Beachhead — Initializing…")
student_msg = student_box.empty()

alt_box = right_col.container()
alt_prog = alt_box.progress(0.0, text="Suggested Alternative — Initializing…")
alt_msg = alt_box.empty()

# -------------------------- Validation (student beachhead) --------------------------
validated_student = None
if student_beachhead.strip():
    validator_raw = LLM.invoke([
        SystemMessage(content=VALIDATOR_PROMPT),
        HumanMessage(content=f"Proposed beachhead: {student_beachhead}\n\nIdea context:\n" + "\n".join(idea_overview_lines))
    ])
    validator = VALIDATOR_SCHEMA.invoke(validator_raw)
    if validator.get("valid", False):
        validated_student = student_beachhead.strip()
        left_col.success("✅ Beachhead looks specific enough.")
    else:
        improved = validator.get("improve", "").strip()
        reasons = validator.get("reasons", [])
        msg = "⚠️ Your beachhead could be more concrete."
        if reasons:
            msg += " Reasons: " + "; ".join(reasons)
        if improved:
            msg += f"\nSuggested rewrite: **{improved}**"
        left_col.info(msg)
        validated_student = improved or student_beachhead.strip()
else:
    left_col.warning("No student-proposed beachhead provided. The left track will be skipped.")

# -------------------------- Parallel Workers (NO Streamlit calls inside) --------------------------
# Totals (keep consistent with event emissions)
TOTAL_STEPS_STUDENT = 18  # 1 plan + (4 dims * 2 queries * 2 steps) + 1 eval
MAX_ROUNDS = 3
TOTAL_STEPS_ALT = MAX_ROUNDS * (1 + (4*2*2) + 1) + 1  # == 55 (incl. finalizing pad)

def student_worker(q: Queue) -> Tuple[Optional[Dict[str,Any]], str, str]:
    """Evaluate student-proposed beachhead with BUTA. Returns (evaluation_json, candidate_name, notes)."""
    def emit(ev: Dict[str, Any]):
        q.put(ev)

    if not validated_student:
        emit({"type": "step", "done": TOTAL_STEPS_STUDENT, "total": TOTAL_STEPS_STUDENT,
              "text": "Skipped (no student beachhead provided)."})
        return None, "", ""

    executor = make_search_executor()
    counters = {"done": 0}

    # Plan queries for FIXED candidate
    counters["done"] += 1
    emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_STUDENT,
          "text": "Planning Tavily queries for your beachhead…"})
    fixed_plan_raw = LLM.invoke([
        SystemMessage(content=FIXED_PLANNER_PROMPT),
        HumanMessage(content=("Overall idea:\n" + "\n".join(idea_overview_lines) +
                              f"\n\nFixed candidate segment:\n{validated_student}"))
    ])
    fixed_plan = FIXED_PLANNER_SCHEMA.invoke(fixed_plan_raw)
    plan_queries = fixed_plan.get("plan", {})

    # Research
    all_notes = ""
    for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
        all_notes += research_dimension(validated_student, dim, plan_queries.get(dim, []),
                                        emit, counters, TOTAL_STEPS_STUDENT, executor)

    # Evaluate
    counters["done"] += 1
    emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_STUDENT,
          "text": "Evaluating BUTA fit for your beachhead…"})
    eval_input = f"Candidate: {validated_student}\n\nResearch notes:\n{all_notes}"
    eval_raw = LLM.invoke([SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=eval_input)])
    evaluation = EVALUATOR_SCHEMA.invoke(eval_raw)

    # Pad to 100% if any steps remain
    while counters["done"] < TOTAL_STEPS_STUDENT:
        counters["done"] += 1
        emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_STUDENT,
              "text": "Finalizing…"})

    emit({"type": "done"})
    return evaluation, validated_student, all_notes

def alternative_worker(q: Queue, student_segment_for_context: Optional[str]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], str]:
    """Run the original iterative loop to find another beachhead, avoiding duplication with student segment."""
    def emit(ev: Dict[str, Any]):
        q.put(ev)

    executor = make_search_executor()

    attempts: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []

    previous_attempts_text = ""
    if student_segment_for_context:
        previous_attempts_text += (
            f"Attempt on '{student_segment_for_context}': "
            f"B:Unknown U:Unknown T:Unknown A:Unknown. "
            f"This was student-proposed; avoid proposing the same or closely overlapping segment.\n"
        )

    counters = {"done": 0}

    for round_idx in range(1, MAX_ROUNDS + 1):
        # Plan
        counters["done"] += 1
        emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_ALT,
              "text": f"Round {round_idx}: Planning the next segment to investigate…"})
        plan_input = f"""
Idea:
Problem: {problem}
Solution: {solution}
Industry: {industry}
Geos: {target_geos}
Buyer roles: {buyer_roles}
Price point: {price_point}
Evidence: {evidence}
Competitors: {competitors}
Constraints: {constraints}

Previous attempts & findings:
{previous_attempts_text}
"""
        plan_raw = LLM.invoke([SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=plan_input)])
        plan = PLANNER_SCHEMA.invoke(plan_raw)

        if plan.get("decision") == "stop":
            counters["done"] += 1
            emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_ALT,
                  "text": f"Stopped planning: {plan.get('reason','')}"})
            if plan.get("candidate"):
                recommendations.append({"segment": plan["candidate"], "eval": None})
            break

        candidate = plan["candidate"]
        if student_segment_for_context and candidate.strip().lower() == student_segment_for_context.strip().lower():
            previous_attempts_text += f"\nSkipped duplicate candidate '{candidate}'.\n"
            continue

        plan_queries = plan["plan"]

        # Research
        all_notes_for_round = ""
        for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
            all_notes_for_round += research_dimension(candidate, dim, plan_queries.get(dim, []),
                                                      emit, counters, TOTAL_STEPS_ALT, executor)

        # Evaluate
        counters["done"] += 1
        emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_ALT,
              "text": f"Evaluating BUTA fit for '{candidate}'…"})
        eval_input = f"Candidate: {candidate}\n\nResearch notes:\n{all_notes_for_round}"
        eval_raw = LLM.invoke([SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=eval_input)])
        evaluation = EVALUATOR_SCHEMA.invoke(eval_raw)

        attempts.append({"segment": candidate, "eval": evaluation, "notes": all_notes_for_round})
        recommendations.append({"segment": candidate, "eval": evaluation})

        if evaluation["decision"] == "good_fit":
            counters["done"] += 1
            emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_ALT,
                  "text": f"Strong fit found for '{candidate}'."})
            break

        previous_attempts_text += (
            f"\nAttempt on '{candidate}': "
            f"B:{evaluation['BUTA']['budget']['rating']} "
            f"U:{evaluation['BUTA']['urgency']['rating']} "
            f"T:{evaluation['BUTA']['top_fit']['rating']} "
            f"A:{evaluation['BUTA']['access']['rating']}. "
            f"{evaluation.get('notes','')}\n"
        )

    # Pad to 100%
    while counters["done"] < TOTAL_STEPS_ALT:
        counters["done"] += 1
        emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_ALT,
              "text": "Finalizing…"})

    emit({"type": "done"})
    return attempts, recommendations, previous_attempts_text

# -------------------------- Launch background tasks --------------------------
student_q: Queue = Queue()
alt_q: Queue = Queue()

futures = {}
with ThreadPoolExecutor(max_workers=2) as pool:
    if validated_student:
        futures["student"] = pool.submit(student_worker, student_q)
    futures["alt"] = pool.submit(alternative_worker, alt_q, validated_student)

    # Main thread: poll queues and update UI safely
    finished = set()
    while len(finished) < len(futures):
        # drain student queue
        try:
            while True:
                ev = student_q.get_nowait()
                if ev.get("type") == "step":
                    done = ev.get("done", 0)
                    total = max(ev.get("total", 1), 1)
                    pct = min(done / total, 1.0)
                    student_msg.info(ev.get("text", "Working…"))
                    student_prog.progress(pct, text=ev.get("text", "Working…"))
                elif ev.get("type") == "done":
                    finished.add("student")
                student_q.task_done()
        except Empty:
            pass

        # drain alt queue
        try:
            while True:
                ev = alt_q.get_nowait()
                if ev.get("type") == "step":
                    done = ev.get("done", 0)
                    total = max(ev.get("total", 1), 1)
                    pct = min(done / total, 1.0)
                    alt_msg.info(ev.get("text", "Working…"))
                    alt_prog.progress(pct, text=ev.get("text", "Working…"))
                elif ev.get("type") == "done":
                    finished.add("alt")
                alt_q.task_done()
        except Empty:
            pass

        # check futures that might be done without sending 'done' (edge cases)
        for key, fut in futures.items():
            if key in finished:
                continue
            if fut.done():
                finished.add(key)

        time.sleep(0.1)

# Gather results from futures
student_result: Tuple[Optional[Dict[str,Any]], str, str] = (None, "", "")
alt_attempts: List[Dict[str,Any]] = []
alt_recs: List[Dict[str,Any]] = []
_ = ""

if "student" in futures:
    student_result = futures["student"].result()
if "alt" in futures:
    alt_attempts, alt_recs, _ = futures["alt"].result()

# -------------------------- Combine results & compile final report --------------------------
attempts_all: List[Dict[str, Any]] = []
recommendations: List[Dict[str, Any]] = []

# Student track
student_eval_json, student_segment_name, student_notes = student_result
if student_eval_json and student_segment_name:
    attempts_all.append({"segment": student_segment_name, "eval": student_eval_json, "notes": student_notes})
    recommendations.append({"segment": student_segment_name, "eval": student_eval_json})

# Alternative track
recommendations.extend(alt_recs)
attempts_all.extend(alt_attempts)

def _scored(rec):
    return score_eval(rec["eval"]) if rec["eval"] else -1

scored_recs = [r for r in recommendations if r["eval"] is not None]
scored_recs = sorted(scored_recs, key=_scored, reverse=True)
top_recs: List[Dict[str,Any]] = []

# Ensure student beachhead stays if present
if student_eval_json and student_segment_name:
    top_recs.append({"segment": student_segment_name, "eval": student_eval_json})
for r in scored_recs:
    if len(top_recs) >= 2:
        break
    if not student_segment_name or r["segment"].strip().lower() != student_segment_name.strip().lower():
        top_recs.append(r)

if not top_recs and recommendations:
    top_recs = recommendations[:1]

# Build evaluations text for final writer
evaluations_text = ""
for rec in top_recs:
    seg = rec["segment"]
    evaluations_text += f"\nSegment: {seg}\n"
    if rec["eval"] is not None:
        ev = rec["eval"]["BUTA"]
        for k in ["budget","urgency","top_fit","access"]:
            evaluations_text += f"- {k.title()} rating: {ev[k]['rating']}\n"
            for e in ev[k]["evidence"][:4]:
                evaluations_text += f"  • {e}\n"
            if ev[k]["sources"]:
                evaluations_text += "  sources: " + ", ".join(ev[k]["sources"]) + "\n"

# Full research notes (EVERYTHING from Tavily kept verbatim)
full_research_notes = "\n".join([a["notes"] for a in attempts_all])

st.toast("Compiling report…", icon="🧩")
final_raw = LLM.invoke([
    SystemMessage(content=FINAL_WRITER_PROMPT.format(
        idea_overview="\n".join(idea_overview_lines),
        evaluations=evaluations_text,
        full_research_notes=full_research_notes,
    ))
])
final_report_text = final_raw.content

st.success("BUTA analysis complete. See the combined report below.", icon="✅")
st.subheader("Complete BUTA Report")
st.write(final_report_text)

# -------------------------- Downloads --------------------------
st.markdown("### Download")
pdf_bytes = build_pdf_from_text(final_report_text)
col_a, col_b = st.columns(2)

with col_a:
    if pdf_bytes:
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="buta_report.pdf",
            mime="application/pdf",
            type="primary",
        )
    else:
        st.warning("PDF generation library not available. Please use the Markdown download instead.")

with col_b:
    st.download_button(
        label="Download Markdown",
        data=final_report_text.encode("utf-8"),
        file_name="buta_report.md",
        mime="text/markdown",
    )
