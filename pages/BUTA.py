import os
import re
from typing import Dict, List, Any

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
st.set_page_config(page_title="BUTA Beachhead Finder (Simple)", layout="wide")

# Secrets/env
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["apiKey"]
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets["tapiKey"]

# -------------------------- LLM & Tools --------------------------
LLM = ChatOpenAI(model="gpt-4.1", temperature=0)

TAVILY_TOOL = TavilySearchResults(
    max_results=3,
    include_answer=True,
    include_raw_content=True,
    search_depth="advanced",
)
TOOLS = [TAVILY_TOOL]

SEARCH_AGENT_SYSTEM = """
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
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=SEARCH_AGENT_SYSTEM)),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="{input}")),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)
SEARCH_AGENT = create_openai_tools_agent(LLM, TOOLS, search_prompt)
EXECUTOR = AgentExecutor(agent=SEARCH_AGENT, tools=TOOLS, verbose=False)

# -------------------------- Prompts --------------------------
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

start = st.sidebar.button("Run BUTA", type="primary")

# -------------------------- Helpers --------------------------
def normalize_urls(text: str) -> List[str]:
    return list(set(re.findall(r'https?://[\w\.-/%\?\=&\+#]+', text)))

def run_tavily_query(q: str) -> str:
    try:
        out = EXECUTOR.invoke({"input": q})["output"]
    except Exception as e:
        out = f"- Search error for '{q}': {e}"
    return out

def research_dimension(segment: str, dim: str, queries: List[str]) -> str:
    # Detailed snippet updates for exact queries
    global steps_done, TOTAL_STEPS
    notes = f"\n=== {segment} :: {dim} ===\n"
    for q in queries:
        steps_done += 1
        step(f"I am searching for '{q}'.", steps_done, TOTAL_STEPS)
        bullets = run_tavily_query(q)
        steps_done += 1
        step(f"I finished searching for '{q}'.", steps_done, TOTAL_STEPS)
        notes += f"Query: {q}\n"
        notes += bullets + "\n"
    return notes

def score_eval(ev: Dict[str,Any]) -> int:
    rank = {"High":2,"Medium":1,"Low":0}
    return sum(rank[ev["BUTA"][k]["rating"]] for k in ["budget","urgency","top_fit","access"])

# -------------------------- UI --------------------------
st.title("BUTA Beachhead Finder — Simple")

if not start:
    st.info("Fill in Problem and Solution, then click **Run BUTA**.")
    st.stop()

if not problem.strip() or not solution.strip():
    st.error("Please fill in both required fields: Problem and Solution.")
    st.stop()

# -------------------------- Progress Bar + Snippets --------------------------
snippet = st.empty()
progress = st.progress(0.0, text="Initializing…")

def step(msg: str, done: int, total: int):
    snippet.info(msg)
    progress.progress(min(done/total, 1.0), text=msg)

# Updated total to account for per-query start/finish messages:
# per round: 1 (plan) + 4 dims * 2 queries * 2 steps (start+finish) = 16 + 1 evaluate = 18
# 3 rounds => 54 + 1 final compile => 55
MAX_ROUNDS = 3
TOTAL_STEPS = MAX_ROUNDS * (1 + (4*2*2) + 1) + 1
steps_done = 0

# -------------------------- Overview (kept simple) --------------------------
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

# -------------------------- Iterative Loop (think → search → adapt) --------------------------
attempts: List[Dict[str, Any]] = []
recommendations: List[Dict[str, Any]] = []

previous_attempts_text = ""

for round_idx in range(1, MAX_ROUNDS + 1):
    # Plan
    steps_done += 1
    step(f"Round {round_idx}: I am planning the next best segment to investigate.", steps_done, TOTAL_STEPS)

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
        steps_done += 1
        step(f"I decided to stop planning because: {plan.get('reason','')}.", steps_done, TOTAL_STEPS)
        if plan.get("candidate"):
            recommendations.append({"segment": plan["candidate"], "eval": None})  # placeholder
        break

    candidate = plan["candidate"]
    plan_queries = plan["plan"]

    # Research across BUTA (now shows exact search messages)
    all_notes_for_round = ""
    for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
        # Actually run queries and capture full text (no omissions)
        all_notes_for_round += research_dimension(candidate, dim, plan_queries.get(dim, [])[:2])

    # Evaluate
    steps_done += 1
    step(f"I am evaluating the BUTA fit for '{candidate}' using the collected research.", steps_done, TOTAL_STEPS)

    eval_input = f"Candidate: {candidate}\n\nResearch notes:\n{all_notes_for_round}"
    eval_raw = LLM.invoke([SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=eval_input)])
    evaluation = EVALUATOR_SCHEMA.invoke(eval_raw)

    attempts.append({"segment": candidate, "eval": evaluation, "notes": all_notes_for_round})
    recommendations.append({"segment": candidate, "eval": evaluation})

    if evaluation["decision"] == "good_fit":
        steps_done += 1
        step(f"I found a strong fit for '{candidate}'. I will stop iterating and compile the report.", steps_done, TOTAL_STEPS)
        break

    # Prepare context for next round
    previous_attempts_text += (
        f"\nAttempt on '{candidate}': "
        f"B:{evaluation['BUTA']['budget']['rating']} "
        f"U:{evaluation['BUTA']['urgency']['rating']} "
        f"T:{evaluation['BUTA']['top_fit']['rating']} "
        f"A:{evaluation['BUTA']['access']['rating']}. "
        f"{evaluation.get('notes','')}\n"
    )

# Keep the best 1–2 candidates by score
scored_recs = [r for r in recommendations if r["eval"] is not None]
scored_recs = sorted(scored_recs, key=lambda r: score_eval(r["eval"]), reverse=True)[:2]
if not scored_recs and recommendations:
    # In case planner stopped before any research
    scored_recs = recommendations[:1]

# Build evaluations text for final writer (still no JSON shown to user)
evaluations_text = ""
for rec in scored_recs:
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
full_research_notes = "\n".join([a["notes"] for a in attempts])

# Final writer — single report with narrative + complete research appendix
steps_done += 1
step("I am compiling the single report with the narrative summary and the complete research appendix.", steps_done, TOTAL_STEPS)

final_raw = LLM.invoke([
    SystemMessage(content=FINAL_WRITER_PROMPT.format(
        idea_overview="\n".join(idea_overview_lines),
        evaluations=evaluations_text,
        full_research_notes=full_research_notes,
    ))
])
final_report_text = final_raw.content

# Done — fill progress if ended early
while steps_done < TOTAL_STEPS:
    steps_done += 1
    step("Finalizing…", steps_done, TOTAL_STEPS)
st.toast("BUTA analysis complete.", icon="✅")

# -------------------------- Single Report Output --------------------------
st.subheader("Complete BUTA Report")
# This single report includes BOTH the narrative summary and the full research appendix.
st.write(final_report_text)

