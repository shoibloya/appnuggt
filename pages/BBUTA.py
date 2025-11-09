import os
import re
import io
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# 1) Student beachhead validator (quick, structured feedback)
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

# 2) Fixed-candidate query planner: produce BUTA queries for a given student candidate
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
student_beachhead = st.sidebar.text_input(
    "Your Proposed Beachhead (optional)",
    placeholder="e.g., Dental clinics with 2–5 chairs in California using Open Dental"
)
also_suggest_alt = st.sidebar.checkbox("Also suggest and evaluate an alternative beachhead (in parallel)", value=True)
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

def make_step_ui(container, label: str):
    box = container.container()
    prog = box.progress(0.0, text=f"{label} — Initializing…")
    msg = box.empty()
    def step(msg_text: str, done: int, total: int):
        msg.info(msg_text)
        prog.progress(min(done/total, 1.0), text=msg_text)
    return step

def score_eval(ev: Dict[str,Any]) -> int:
    rank = {"High":2,"Medium":1,"Low":0}
    return sum(rank[ev["BUTA"][k]["rating"]] for k in ["budget","urgency","top_fit","access"])

def research_dimension(segment: str, dim: str, queries: List[str], step_cb, counters: Dict[str,int], total_steps: int, executor: AgentExecutor) -> str:
    notes = f"\n=== {segment} :: {dim} ===\n"
    for q in queries[:2]:
        counters["done"] += 1
        step_cb(f"[{segment}] Searching: '{q}'", counters["done"], total_steps)
        bullets = run_tavily_query(executor, q)
        counters["done"] += 1
        step_cb(f"[{segment}] Finished: '{q}'", counters["done"], total_steps)
        notes += f"Query: {q}\n"
        notes += bullets + "\n"
    return notes

# PDF builder (with ReportLab if available; fallback to Markdown download)
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

# -------------------------- UI --------------------------
st.title("BUTA Beachhead Finder — Parallel")

if not start:
    st.info("Fill in Problem and Solution, optionally add your proposed beachhead, select whether to suggest an alternative, then click **Run BUTA**.")
    st.stop()

if not problem.strip() or not solution.strip():
    st.error("Please fill in both required fields: Problem and Solution.")
    st.stop()

# -------------------------- Idea Overview (unchanged block) --------------------------
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

# Left = Student-proposed evaluation; Right = Suggested alternative (if enabled)
student_step = make_step_ui(left_col, "Your Beachhead (BUTA)")
alt_step = make_step_ui(right_col, "Suggested Alternative (BUTA)") if also_suggest_alt else None

# -------------------------- Validation (student beachhead) --------------------------
validated_student = None
validation_feedback = ""

if student_beachhead.strip():
    validator_raw = LLM.invoke([
        SystemMessage(content=VALIDATOR_PROMPT),
        HumanMessage(content=f"Proposed beachhead: {student_beachhead}\n\nIdea context:\n" + "\n".join(idea_overview_lines))
    ])
    validator = VALIDATOR_SCHEMA.invoke(validator_raw)
    if validator.get("valid", False):
        validated_student = student_beachhead.strip()
        validation_feedback = "✅ Looks specific enough."
    else:
        improved = validator.get("improve", "").strip()
        reasons = validator.get("reasons", [])
        validation_feedback = "⚠️ Your beachhead could be more concrete.\n"
        if reasons:
            validation_feedback += "Reasons: " + "; ".join(reasons) + "\n"
        if improved:
            validation_feedback += f"Suggested rewrite: **{improved}**"
        # Still proceed using improved if available; else use original
        validated_student = improved or student_beachhead.strip()
    left_col.info(validation_feedback)
else:
    left_col.warning("No student-proposed beachhead provided. The left track will be skipped.")

# -------------------------- Parallel Workers --------------------------
# Counters and totals
# Student track: 1 (plan queries) + 16 (search start/finish) + 1 (evaluate) = 18
TOTAL_STEPS_STUDENT = 18
student_counters = {"done": 0}

# Alternative track: keep previous estimate: per round 18; 3 rounds + 1 compile = 55
MAX_ROUNDS = 3
TOTAL_STEPS_ALT = MAX_ROUNDS * (1 + (4*2*2) + 1) + 1  # == 55
alt_counters = {"done": 0}

def student_worker() -> Tuple[Optional[Dict[str,Any]], str, str]:
    """Evaluate student-proposed beachhead with BUTA. Returns (evaluation_json, candidate_name, notes)."""
    if not validated_student:
        return None, "", ""
    executor = make_search_executor()

    # Plan queries for the FIXED candidate (non-core prompt)
    student_counters["done"] += 1
    student_step("Planning Tavily queries for your beachhead…", student_counters["done"], TOTAL_STEPS_STUDENT)
    fixed_plan_raw = LLM.invoke([
        SystemMessage(content=FIXED_PLANNER_PROMPT),
        HumanMessage(content=(
            "Overall idea:\n" + "\n".join(idea_overview_lines) +
            f"\n\nFixed candidate segment:\n{validated_student}"
        ))
    ])
    fixed_plan = FIXED_PLANNER_SCHEMA.invoke(fixed_plan_raw)
    plan_queries = fixed_plan.get("plan", {})

    # Research
    all_notes = ""
    for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
        all_notes += research_dimension(validated_student, dim, plan_queries.get(dim, []),
                                        student_step, student_counters, TOTAL_STEPS_STUDENT, executor)

    # Evaluate
    student_counters["done"] += 1
    student_step("Evaluating BUTA fit for your beachhead…", student_counters["done"], TOTAL_STEPS_STUDENT)
    eval_input = f"Candidate: {validated_student}\n\nResearch notes:\n{all_notes}"
    eval_raw = LLM.invoke([SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=eval_input)])
    evaluation = EVALUATOR_SCHEMA.invoke(eval_raw)

    # Finalize meter if we finished early
    while student_counters["done"] < TOTAL_STEPS_STUDENT:
        student_counters["done"] += 1
        student_step("Finalizing…", student_counters["done"], TOTAL_STEPS_STUDENT)

    return evaluation, validated_student, all_notes

def alternative_worker(student_segment_for_context: Optional[str]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], str]:
    """Run the original iterative loop to find another beachhead, avoiding duplication with student segment."""
    if not also_suggest_alt:
        return [], [], ""
    executor = make_search_executor()

    attempts: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []

    previous_attempts_text = ""
    if student_segment_for_context:
        # Seed prior attempts to avoid duplication
        previous_attempts_text += (
            f"Attempt on '{student_segment_for_context}': "
            f"B:Unknown U:Unknown T:Unknown A:Unknown. "
            f"This was student-proposed; avoid proposing the same or closely overlapping segment.\n"
        )

    for round_idx in range(1, MAX_ROUNDS + 1):
        # Plan
        alt_counters["done"] += 1
        alt_step(f"Round {round_idx}: Planning the next segment to investigate…", alt_counters["done"], TOTAL_STEPS_ALT)

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
            alt_counters["done"] += 1
            alt_step(f"Stopped planning: {plan.get('reason','')}", alt_counters["done"], TOTAL_STEPS_ALT)
            if plan.get("candidate"):
                recommendations.append({"segment": plan["candidate"], "eval": None})
            break

        candidate = plan["candidate"]
        if student_segment_for_context and candidate.strip().lower() == student_segment_for_context.strip().lower():
            # Skip exact duplicate
            previous_attempts_text += f"\nSkipped duplicate candidate '{candidate}'.\n"
            continue

        plan_queries = plan["plan"]

        # Research
        all_notes_for_round = ""
        for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
            alt_counters["done"] += 0  # (no update here; done inside research_dimension)
            all_notes_for_round += research_dimension(candidate, dim, plan_queries.get(dim, []),
                                                      alt_step, alt_counters, TOTAL_STEPS_ALT, executor)

        # Evaluate
        alt_counters["done"] += 1
        alt_step(f"Evaluating BUTA fit for '{candidate}'…", alt_counters["done"], TOTAL_STEPS_ALT)

        eval_input = f"Candidate: {candidate}\n\nResearch notes:\n{all_notes_for_round}"
        eval_raw = LLM.invoke([SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=eval_input)])
        evaluation = EVALUATOR_SCHEMA.invoke(eval_raw)

        attempts.append({"segment": candidate, "eval": evaluation, "notes": all_notes_for_round})
        recommendations.append({"segment": candidate, "eval": evaluation})

        if evaluation["decision"] == "good_fit":
            alt_counters["done"] += 1
            alt_step(f"Strong fit found for '{candidate}'.", alt_counters["done"], TOTAL_STEPS_ALT)
            break

        # Prepare context
        previous_attempts_text += (
            f"\nAttempt on '{candidate}': "
            f"B:{evaluation['BUTA']['budget']['rating']} "
            f"U:{evaluation['BUTA']['urgency']['rating']} "
            f"T:{evaluation['BUTA']['top_fit']['rating']} "
            f"A:{evaluation['BUTA']['access']['rating']}. "
            f"{evaluation.get('notes','')}\n"
        )

    # Ensure meter completes
    while alt_counters["done"] < TOTAL_STEPS_ALT:
        alt_counters["done"] += 1
        alt_step("Finalizing…", alt_counters["done"], TOTAL_STEPS_ALT)

    return attempts, recommendations, previous_attempts_text

# -------------------------- Run parallel --------------------------
student_result = (None, "", "")
alt_attempts: List[Dict[str,Any]] = []
alt_recs: List[Dict[str,Any]] = []
_ = ""

with ThreadPoolExecutor(max_workers=2 if also_suggest_alt else 1) as pool:
    futures = []
    if validated_student:
        futures.append(pool.submit(student_worker))
    if also_suggest_alt:
        futures.append(pool.submit(alternative_worker, validated_student))
    for f in as_completed(futures):
        res = f.result()
        # Identify by arity
        if isinstance(res, tuple) and len(res) == 3 and isinstance(res[1], str) and isinstance(res[2], str) and (res[0] is None or isinstance(res[0], dict)):
            student_result = res  # (evaluation_json, candidate_name, notes)
        else:
            alt_attempts, alt_recs, _ = res  # (attempts, recommendations, previous_attempts_text)

# -------------------------- Gather results --------------------------
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

# Keep best 1–2 by score (prefer including the student’s if available)
def _scored(rec):
    return score_eval(rec["eval"]) if rec["eval"] else -1

scored_recs = [r for r in recommendations if r["eval"] is not None]
scored_recs = sorted(scored_recs, key=_scored, reverse=True)
top_recs: List[Dict[str,Any]] = []

# Ensure student beachhead stays if present
if student_eval_json and student_segment_name:
    top_recs.append({"segment": student_segment_name, "eval": student_eval_json})
# Fill remaining slots with best alternatives not equal to student segment
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

# -------------------------- Compile final report (UNCHANGED FINAL WRITER PROMPT) --------------------------
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
