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
st.set_page_config(page_title="BUTA Beachhead Finder", layout="wide")

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
Write a clear BUTA summary for the recommended beachhead(s).
- Use headings and bullets (no code, no JSON).
- Include ratings and 2–4 key evidence bullets per dimension with source links inline.
- Conclude with: Go-to-market hypothesis, Key risks, Immediate validation steps.

Idea overview:
{idea_overview}

Recommended beachheads with evaluations:
{evaluations}

Produce a clean, readable report.
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

start = st.sidebar.button("Find My Beachhead", type="primary")

# -------------------------- Helpers --------------------------
def normalize_urls(text: str) -> List[str]:
    return list(set(re.findall(r'https?://[\w\.-/%\?\=&\+#]+', text)))

def run_tavily_query(q: str) -> str:
    try:
        out = EXECUTOR.invoke({"input": q})["output"]
    except Exception as e:
        out = f"- Search error for '{q}': {e}"
    return out

def research_dimension(segment: str, dim: str, queries: List[str], log_list: List[str]) -> str:
    notes = f"\n=== {segment} :: {dim} ===\n"
    for q in queries:
        log_list.append(f"Searching ({dim}): {q}")
        bullets = run_tavily_query(q)
        log_list.append(bullets)
        notes += f"Query: {q}\n{bullets}\n"
    return notes

def score_eval(rec: Dict[str,Any]) -> int:
    rank = {"High":2,"Medium":1,"Low":0}
    return sum(rank[rec["eval"]["BUTA"][k]["rating"]] for k in ["budget","urgency","top_fit","access"])

# -------------------------- UI Layout --------------------------
st.title("BUTA Beachhead Finder — Iterative & Transparent")
st.caption("Think → Search → Adapt, with a hard stop at 3 rounds. Live progress and sources included.")

tabs = st.tabs(["Dashboard", "Live Feed", "Final Report", "Reference"])

# Reference tab (no HTML/CSS; uses built-in image component)
with tabs[3]:
    st.subheader("BUTA Reference")
    st.write("B = Budget, U = Urgency, T = Top-3 Fit, A = Access")
    for p in [
        "/mnt/data/83246a71-57ac-41b1-b7b2-a189c1564b83.png",
        "/mnt/data/f4e0e3f7-fb29-40d3-9a15-e6ac9fd54a26.png",
        "/mnt/data/e6ed0f72-3758-4ab2-9232-2f9bddb29922.png",
        "/mnt/data/c419f54e-6bd4-4da6-9c8f-839754b71f78.png",
        "/mnt/data/81b8e7b3-5e9a-4ae1-ba5f-876ebdebd4a0.png",
    ]:
        try:
            st.image(p, use_container_width=True)
        except Exception:
            pass

# Guard rails
if not start:
    with tabs[0]:
        st.subheader("How this works")
        st.write(
            "1) Enter **Problem** and **Solution** (only required fields). "
            "2) The agent proposes a specific candidate beachhead and a focused web research plan. "
            "3) It researches across **B**, **U**, **T**, **A** using Tavily (with sources). "
            "4) It evaluates and adapts, up to 3 rounds, stopping early on a strong fit. "
            "5) A compact final report is generated."
        )
    st.stop()

if not problem.strip() or not solution.strip():
    with tabs[0]:
        st.error("Please fill in both required fields: Problem and Solution.")
    st.stop()

# -------------------------- Overview (Dashboard) --------------------------
with tabs[0]:
    st.subheader("Idea Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Problem:** {problem}")
        if industry: st.write(f"**Industry:** {industry}")
        if buyer_roles: st.write(f"**Buyer/User Roles:** {buyer_roles}")
        if evidence: st.write(f"**Evidence/Traction:** {evidence}")
    with col2:
        st.write(f"**Solution:** {solution}")
        if target_geos: st.write(f"**Target Geography:** {target_geos}")
        if price_point: st.write(f"**Price Point:** {price_point}")
        if competitors: st.write(f"**Known Competitors:** {competitors}")
    if constraints: st.write(f"**Constraints:** {constraints}")
    st.divider()

# -------------------------- Iterative Loop --------------------------
max_rounds = 3
attempts: List[Dict[str, Any]] = []
recommendations: List[Dict[str, Any]] = []
feed: List[str] = []

progress_bar = st.progress(0.0, text="Starting…")
status_area = st.empty()

previous_attempts_text = ""
best_so_far_text = st.empty()
best_metrics_placeholder = st.empty()
alt_placeholder = st.empty()

def update_best_dashboard():
    if len(recommendations) == 0:
        return
    # Best is first by score
    best = sorted(recommendations, key=score_eval, reverse=True)[0]
    seg = best["segment"]
    ev = best["eval"]["BUTA"]
    with tabs[0]:
        best_so_far_text.subheader(f"Current Best Beachhead: {seg}")
        cols = st.columns(4)
        map_ = {"High":"🟢 High", "Medium":"🟡 Medium", "Low":"🔵 Low"}
        cols[0].metric("Budget", map_.get(ev["budget"]["rating"], ev["budget"]["rating"]))
        cols[1].metric("Urgency", map_.get(ev["urgency"]["rating"], ev["urgency"]["rating"]))
        cols[2].metric("Top-3 Fit", map_.get(ev["top_fit"]["rating"], ev["top_fit"]["rating"]))
        cols[3].metric("Access", map_.get(ev["access"]["rating"], ev["access"]["rating"]))

        st.write("Key evidence (sample):")
        for k in ["budget","urgency","top_fit","access"]:
            bullets = ev[k]["evidence"][:2]
            for b in bullets:
                st.write(f"- {b}")

        st.divider()

    # Alternatives (compact list)
    with tabs[0]:
        alt_placeholder.subheader("Alternatives Under Consideration")
        if len(recommendations) == 1:
            st.write("No alternatives yet.")
        else:
            for rec in sorted(recommendations, key=score_eval, reverse=True)[1:3]:
                r = rec["eval"]["BUTA"]
                st.write(
                    f"- {rec['segment']} | "
                    f"B:{r['budget']['rating']} U:{r['urgency']['rating']} "
                    f"T:{r['top_fit']['rating']} A:{r['access']['rating']}"
                )

# Live feed renderer (latest 10 entries)
def render_feed():
    with tabs[1]:
        st.subheader("Live Research Feed")
        recent = feed[-10:]
        for line in recent:
            st.write(line)
        st.caption(f"Showing latest {len(recent)} entries. Full log appears in the Final Report tab.")

for round_idx in range(1, max_rounds + 1):
    progress_bar.progress((round_idx-1)/max_rounds, text=f"Round {round_idx} of {max_rounds}: planning…")

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
    plan_msg = [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=plan_input)]
    plan_raw = LLM.invoke(plan_msg)
    plan = PLANNER_SCHEMA.invoke(plan_raw)

    if plan.get("decision") == "stop":
        with tabs[1]:
            st.info(f"Planner decided to stop: {plan.get('reason','')}")
            if plan.get("candidate"):
                st.write(f"Best available candidate to present: {plan['candidate']}")
        break

    candidate = plan["candidate"]
    why_this = plan["why_this"]
    plan_queries = plan["plan"]

    with tabs[1]:
        st.write(f"Candidate to investigate: **{candidate}**")
        st.write(f"Why this next: {why_this}")

    progress_bar.progress((round_idx-1)/max_rounds, text=f"Round {round_idx}: researching {candidate} …")

    # Research across BUTA
    round_notes = ""
    for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
        queries = plan_queries.get(dim, [])[:2]
        feed.append(f"--- {dim} ---")
        round_notes += research_dimension(candidate, dim, queries, feed)
        render_feed()

    # Evaluate
    eval_input = f"Candidate: {candidate}\n\nResearch notes:\n{round_notes}"
    eval_msg = [SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=eval_input)]
    eval_raw = LLM.invoke(eval_msg)
    evaluation = EVALUATOR_SCHEMA.invoke(eval_raw)

    with tabs[0]:
        st.subheader(f"Round {round_idx} — Preliminary BUTA Scores for: {candidate}")
        cols = st.columns(4)
        m = {"High":"🟢 High","Medium":"🟡 Medium","Low":"🔵 Low"}
        cols[0].metric("Budget", m.get(evaluation["BUTA"]["budget"]["rating"], evaluation["BUTA"]["budget"]["rating"]))
        cols[1].metric("Urgency", m.get(evaluation["BUTA"]["urgency"]["rating"], evaluation["BUTA"]["urgency"]["rating"]))
        cols[2].metric("Top-3 Fit", m.get(evaluation["BUTA"]["top_fit"]["rating"], evaluation["BUTA"]["top_fit"]["rating"]))
        cols[3].metric("Access", m.get(evaluation["BUTA"]["access"]["rating"], evaluation["BUTA"]["access"]["rating"]))
        st.write(evaluation.get("notes",""))
        st.divider()

    attempts.append({"segment": candidate, "eval": evaluation, "notes": round_notes})
    recommendations.append({"segment": candidate, "eval": evaluation})
    update_best_dashboard()
    render_feed()

    if evaluation["decision"] == "good_fit":
        with tabs[1]:
            st.success("Strong fit found. Stopping early.")
        break

    previous_attempts_text += (
        f"\nAttempt on '{candidate}': "
        f"B:{evaluation['BUTA']['budget']['rating']} "
        f"U:{evaluation['BUTA']['urgency']['rating']} "
        f"T:{evaluation['BUTA']['top_fit']['rating']} "
        f"A:{evaluation['BUTA']['access']['rating']}. "
        f"{evaluation.get('notes','')}\n"
    )
    progress_bar.progress(round_idx/max_rounds, text=f"Round {round_idx} complete.")

# Select best 1–2 for final
recommendations = sorted(recommendations, key=score_eval, reverse=True)[:2]

# -------------------------- Final Report --------------------------
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

evaluations_text = ""
for rec in recommendations:
    seg = rec["segment"]
    ev = rec["eval"]["BUTA"]
    evaluations_text += f"\nSegment: {seg}\n"
    for k in ["budget","urgency","top_fit","access"]:
        evaluations_text += f"- {k.title()} rating: {ev[k]['rating']}\n"
        for e in ev[k]["evidence"][:3]:
            evaluations_text += f"  • {e}\n"
        if ev[k]["sources"]:
            evaluations_text += "  sources: " + ", ".join(ev[k]["sources"][:4]) + "\n"

final_msg = [
    SystemMessage(
        content=FINAL_WRITER_PROMPT.format(
            idea_overview="\n".join(idea_overview_lines),
            evaluations=evaluations_text,
        )
    )
]
final_raw = LLM.invoke(final_msg)
final_report_text = final_raw.content

with tabs[2]:
    st.subheader("Final BUTA Report")
    st.write(final_report_text)

    # Full log (optional)
    st.divider()
    st.subheader("Full Research Log")
    full_notes = "\n".join([a["notes"] for a in attempts])
    st.write(full_notes)

    # Download buttons
    st.download_button(
        "Download Report (.md)",
        data=final_report_text.encode("utf-8"),
        file_name="buta_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.download_button(
        "Download Full Log (.txt)",
        data=full_notes.encode("utf-8"),
        file_name="buta_research_log.txt",
        mime="text/plain",
        use_container_width=True,
    )

# Footer meta
all_notes = "\n".join([a["notes"] for a in attempts])
unique_links = normalize_urls(all_notes)
with tabs[0]:
    st.caption(
        f"Method: Iterative BUTA search with GPT-4.1 + Tavily. "
        f"Rounds run: {len(attempts)}. Unique sources referenced: {len(unique_links)}."
    )
