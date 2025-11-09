# =============================================================================
# BUTA Beachhead Finder — Parallel (Thread-Safe UI, PDF export)
# =============================================================================

import os
import re
import io
import time
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
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

# --- LangChain Agents API compatibility (handles older/newer versions) ---
try:
    # Newer common path
    from langchain.agents import AgentExecutor, create_openai_tools_agent
except ImportError:
    # Fallbacks for envs where symbols moved
    try:
        from langchain.agents.agent import AgentExecutor  # old path
    except Exception as _:
        AgentExecutor = None  # will assert below

    try:
        # In some versions create_openai_tools_agent is named create_tool_calling_agent
        from langchain.agents import create_tool_calling_agent as create_openai_tools_agent
    except Exception as _:
        raise ImportError(
            "LangChain agents API not found. Please upgrade/downgrade langchain to a version with "
            "`AgentExecutor` and `create_openai_tools_agent` (or `create_tool_calling_agent`)."
        )

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
    assert AgentExecutor is not None, "AgentExecutor class unavailable in this LangChain version."
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

# -------------------------- CORE PROMPTS (DO NOT CHANGE) --------------------------
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

# -------------------------- Helper Prompts (non-core; safe to edit) --------------------------
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

# -------------------------- Sidebar Inputs (ALL REQUIRED) --------------------------
st.sidebar.header("Fill All Fields")

problem = st.sidebar.text_area("Problem / Job-to-be-Done", height=80)
solution = st.sidebar.text_area("Your Solution & Differentiators", height=100)
industry = st.sidebar.text_input("Industry / Category")
target_geos = st.sidebar.text_input("Target Geography")
buyer_roles = st.sidebar.text_input("Buyer / User Roles")
price_point = st.sidebar.text_input("Expected Price Point")
competitors = st.sidebar.text_area("Known Alternatives / Competitors", height=60)

st.sidebar.markdown("---")
student_beachhead = st.sidebar.text_area(
    "Your Proposed Beachhead (Segment)",
    placeholder="e.g., Mid-sized DTC fashion stores (50–200 SKUs) in the US on Shopify Plus, using GA4 and Klaviyo.",
    height=90,
)

start = st.sidebar.button("Run BUTA", type="primary")

# -------------------------- Small helpers --------------------------
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

# PDF builder: ReportLab first, fpdf2 fallback
def build_pdf_from_text(text: str) -> Optional[bytes]:
    # Try ReportLab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.units import mm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=18*mm, bottomMargin=18*mm)
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
            safe = (line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
            flow.append(Paragraph(safe, normal))
            flow.append(Spacer(1, 3))
        doc.build(flow)
        pdf_bytes = buf.getvalue()
        buf.close()
        return pdf_bytes
    except Exception:
        pass

    # Fallback to fpdf2 (simple layout)
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("helvetica", size=11)
        content = text.replace("\r\n", "\n")
        for line in content.split("\n"):
            if not line.strip():
                pdf.ln(3)
            else:
                pdf.multi_cell(0, 5, line)
        return pdf.output(dest="S").encode("latin-1", "ignore")
    except Exception:
        return None

# -------------------------- Main UI --------------------------
st.title("BUTA Beachhead Finder — Parallel")

# Detailed instructions (MAIN AREA). They disappear once Run is clicked.
if not start:
    st.markdown(
        """
### How this app works

You’ll provide your startup idea and a **specific** proposed beachhead segment.  
When you click **Run BUTA**, the app:
1. Runs a **BUTA Analysis on your beachhead** (Budget, Urgency, Top-3 Fit, Access).
2. In parallel, researches a **Suggested Alternative beachhead** (avoids duplicating your segment).
3. Compiles a single report with a narrative summary, inline evidence with links, and a complete Research Appendix.
4. Lets you **download the full report as a PDF**.

#### What to enter (be precise)

- **Problem / Job-to-be-Done**  
  Example: *“E-commerce brands struggle to convert product page visits into purchases.”*  
  Not: *“Low conversions.”* (too vague)

- **Solution & Differentiators**  
  Example: *“AI CRO assistant that analyzes on-page behavior, generates A/B tests, 1-click deploy, integrates with Shopify + GA4.”*

- **Industry / Category**  
  Example: *“E-commerce conversion optimization.”*

- **Target Geography**  
  Example: *“United States.”* (or specify country/region/state)

- **Buyer / User Roles**  
  Example: *“Head of Growth; E-commerce Manager.”*

- **Expected Price Point**  
  Example: *“$500–$1,500 per month.”*

- **Known Alternatives / Competitors**  
  Example: *“Optimizely, VWO, Convert.com.”*

- **Your Proposed Beachhead (Segment)**  
  **Good:** *“Mid-sized DTC fashion stores (50–200 SKUs) in the US on Shopify Plus, using GA4 and Klaviyo.”*  
  **Bad:** *“Shopify stores.”* (too broad)

When ready, fill all fields in the left panel and click **Run BUTA**.
        """.strip()
    )
    st.stop()

# Validate that all fields are filled
required_fields = {
    "Problem / Job-to-be-Done": problem,
    "Solution & Differentiators": solution,
    "Industry / Category": industry,
    "Target Geography": target_geos,
    "Buyer / User Roles": buyer_roles,
    "Expected Price Point": price_point,
    "Known Alternatives / Competitors": competitors,
    "Your Proposed Beachhead (Segment)": student_beachhead,
}
missing = [k for k, v in required_fields.items() if not v or not v.strip()]
if missing:
    st.error("Please fill in all required fields: " + ", ".join(missing))
    st.stop()

# -------------------------- Idea Overview --------------------------
idea_overview_lines = [
    f"Problem: {problem}",
    f"Solution: {solution}",
    f"Industry: {industry}",
    f"Geography: {target_geos}",
    f"Buyer/User Roles: {buyer_roles}",
    f"Price Point: {price_point}",
    f"Competitors: {competitors}",
]

# -------------------------- Vertical loaders (one below the other) --------------------------
st.subheader("BUTA Analysis — **User's Beachhead**")
student_box = st.container()
student_prog = student_box.progress(0.0, text="User's Beachhead — Initializing…")
student_msg = student_box.empty()

st.subheader("BUTA Analysis — **Suggested Alternative**")
alt_box = st.container()
alt_prog = alt_box.progress(0.0, text="Suggested Alternative — Initializing…")
alt_msg = alt_box.empty()

# -------------------------- Validation (silent; no banners) --------------------------
validated_student = student_beachhead.strip()
if validated_student:
    validator_raw = LLM.invoke([
        SystemMessage(content=VALIDATOR_PROMPT),
        HumanMessage(content=f"Proposed beachhead: {validated_student}\n\nIdea context:\n" + "\n".join(idea_overview_lines))
    ])
    validator = VALIDATOR_SCHEMA.invoke(validator_raw)
    if not validator.get("valid", False):
        improved = validator.get("improve", "").strip()
        validated_student = improved or validated_student

# -------------------------- Workers (NO Streamlit calls inside) --------------------------
TOTAL_STEPS_STUDENT = 18  # 1 plan + (4 dims * 2 queries * 2 steps) + 1 eval
MAX_ROUNDS = 3
TOTAL_STEPS_ALT = MAX_ROUNDS * (1 + (4*2*2) + 1) + 1  # == 55

def student_worker(q: Queue) -> Tuple[Optional[Dict[str,Any]], str, str]:
    """Evaluate user's beachhead with BUTA. Returns (evaluation_json, candidate_name, notes)."""
    def emit(ev: Dict[str, Any]):
        q.put(ev)

    executor = make_search_executor()
    counters = {"done": 0}

    counters["done"] += 1
    emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_STUDENT,
          "text": "Planning Tavily queries for the user's beachhead…"})

    fixed_plan_raw = LLM.invoke([
        SystemMessage(content=FIXED_PLANNER_PROMPT),
        HumanMessage(content=("Overall idea:\n" + "\n".join(idea_overview_lines) +
                              f"\n\nFixed candidate segment:\n{validated_student}"))
    ])
    fixed_plan = FIXED_PLANNER_SCHEMA.invoke(fixed_plan_raw)
    plan_queries = fixed_plan.get("plan", {})

    all_notes = ""
    for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
        all_notes += research_dimension(validated_student, dim, plan_queries.get(dim, []),
                                        emit, counters, TOTAL_STEPS_STUDENT, executor)

    counters["done"] += 1
    emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_STUDENT,
          "text": "Evaluating BUTA fit for the user's beachhead…"})

    eval_input = f"Candidate: {validated_student}\n\nResearch notes:\n{all_notes}"
    eval_raw = LLM.invoke([SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=eval_input)])
    evaluation = EVALUATOR_SCHEMA.invoke(eval_raw)

    while counters["done"] < TOTAL_STEPS_STUDENT:
        counters["done"] += 1
        emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_STUDENT,
              "text": "Finalizing…"})

    emit({"type": "done"})
    return evaluation, validated_student, all_notes

def alternative_worker(q: Queue, student_segment_for_context: Optional[str]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], str]:
    """Iterative loop to find another beachhead, avoiding duplication with user's segment."""
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
            f"This was user-proposed; avoid proposing the same or closely overlapping segment.\n"
        )

    counters = {"done": 0}

    for round_idx in range(1, MAX_ROUNDS + 1):
        counters["done"] += 1
        emit({"type": "step", "done": counters["done"], "total": TOTAL_STEPS_ALT,
              "text": f"Round {round_idx}: Planning the next alternative segment…"})

        plan_input = f"""
Idea:
Problem: {problem}
Solution: {solution}
Industry: {industry}
Geos: {target_geos}
Buyer roles: {buyer_roles}
Price point: {price_point}
Competitors: {competitors}

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

        all_notes_for_round = ""
        for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
            all_notes_for_round += research_dimension(candidate, dim, plan_queries.get(dim, []),
                                                      emit, counters, TOTAL_STEPS_ALT, executor)

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

        time.sleep(0.1)

# -------------------------- Gather results --------------------------
student_result: Tuple[Optional[Dict[str,Any]], str, str] = (None, "", "")
alt_attempts: List[Dict[str,Any]] = []
alt_recs: List[Dict[str,Any]] = []
_ = ""

student_result = futures["student"].result()
alt_attempts, alt_recs, _ = futures["alt"].result()

# Combine
attempts_all: List[Dict[str, Any]] = []
recommendations: List[Dict[str, Any]] = []

# User (student) track
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

# Always include user's beachhead if available
if student_eval_json and student_segment_name:
    top_recs.append({"segment": student_segment_name, "eval": student_eval_json})
for r in scored_recs:
    if len(top_recs) >= 2:
        break
    if not student_segment_name or r["segment"].strip().lower() != student_segment_name.strip().lower():
        top_recs.append(r)

if not top_recs and recommendations:
    top_recs = recommendations[:1]

# Build evaluations text for final writer (NOT displayed; only for PDF)
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

# Final writer — single report (NOT displayed; used for PDF)
final_raw = LLM.invoke([
    SystemMessage(content=FINAL_WRITER_PROMPT.format(
        idea_overview="\n".join(idea_overview_lines),
        evaluations=evaluations_text,
        full_research_notes=full_research_notes,
    ))
])
final_report_text = final_raw.content

# -------------------------- Download (PDF ONLY) --------------------------
pdf_bytes = build_pdf_from_text(final_report_text)
st.markdown("---")
if pdf_bytes:
    st.download_button(
        label="Download BUTA Report (PDF)",
        data=pdf_bytes,
        file_name="buta_report.pdf",
        mime="application/pdf",
        type="primary",
    )
else:
    st.error("PDF generation failed.")
