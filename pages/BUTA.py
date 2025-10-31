import os
import re
import time
from typing import Dict, List, Any

import streamlit as st

# LangChain / OpenAI & Tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults

# ============ Page & Keys ============
st.set_page_config(page_title="BUTA Beachhead Finder (Iterative)", layout="wide")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["apiKey"]
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets["tapiKey"]

# ============ Models & Tools ============
LLM = ChatOpenAI(model="gpt-4.1", temperature=0)
TAVILY_TOOL = TavilySearchResults(
    max_results=3,
    include_answer=True,
    include_raw_content=True,
    search_depth="advanced",
)
TOOLS = [TAVILY_TOOL]

# Agent that can call Tavily with a clean search prompt
SEARCH_AGENT_SYSTEM = """
You are a precise market research analyst. For each search query you receive:
- Use the Tavily tool to gather recent, credible facts.
- Return 3–6 short bullets.
- Each bullet MUST end with a source link in parentheses.
- Prefer primary data, reputable news, analyst reports, government/NGO stats.
- If evidence is mixed or uncertain, say so explicitly and still cite a source.
- No fluff. Facts only.
"""
search_prompt = ChatPromptTemplate(
    input_variables=["agent_scratchpad","input"],
    messages=[
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=SEARCH_AGENT_SYSTEM)),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="{input}")),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ],
)
SEARCH_AGENT = create_openai_tools_agent(LLM, TOOLS, search_prompt)
EXECUTOR = AgentExecutor(agent=SEARCH_AGENT, tools=TOOLS, verbose=False)

# ============ Prompts (Planner, Evaluator, Final Writer) ============
PLANNER_SCHEMA = JsonOutputParser()
PLANNER_PROMPT = """
You are an iterative beachhead-finder using the BUTA framework:

B = Budget (ability/willingness to pay),
U = Urgency (near-term reason to adopt),
T = Top-3 Fit (solution plausibly among the top options for the job),
A = Access (clear way to reach/convert the segment).

TASK:
Given the idea below and any previous attempts & findings, propose ONE best next candidate segment to investigate now, and a tight Tavily search plan (2 specific queries for each of B, U, T, A). The candidate should be concrete (e.g., “Dental clinics with 2–5 chairs in California using Open Dental”), not a broad persona.

Stop conditions:
- If a prior candidate already met strong BUTA evidence (≥2 dimensions High and none Low), you may return {"decision":"stop","reason":"good_fit_identified"}.
- If after multiple attempts it is unlikely to find a good fit, return {"decision":"stop","reason":"exhausted"} and suggest the best available candidate to present anyway.

Return STRICT JSON ONLY:

{
  "decision": "continue | stop",
  "reason": "short reason",
  "candidate": "Precise segment name",
  "why_this": "Brief rationale (what makes this promising to test next)",
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
Evaluate the candidate segment with the research notes provided. Score each BUTA dimension:

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
Write a clear, student-friendly BUTA summary for the recommended beachhead(s).
- Use headings and bullets (no code, no JSON).
- Include ratings and 2–4 key evidence bullets per dimension with source links inline.
- Conclude with: Go-to-market hypothesis, Key risks, and Immediate validation steps.

Input:
- Idea overview:
{idea_overview}

- Recommended (1–2) beachheads with BUTA evaluations:
{evaluations}

Produce a clean, readable report.
"""

# ============ Sidebar Input Form ============
st.sidebar.header("Tell us about your idea")

problem = st.sidebar.text_area("Problem / Job-to-be-Done (required)")
solution = st.sidebar.text_area("Your Solution & Differentiators (required)")

industry = st.sidebar.text_input("Industry / Category (optional)")
target_geos = st.sidebar.text_input("Target Geography (optional)")
buyer_roles = st.sidebar.text_input("Buyer / User Roles (optional)")
price_point = st.sidebar.text_input("Expected Price Point (optional)")
evidence = st.sidebar.text_area("Evidence/Traction (optional)")
competitors = st.sidebar.text_area("Known Alternatives/Competitors (optional)")
constraints = st.sidebar.text_area("Constraints (optional)")

max_rounds = 3  # hard stop
run = st.sidebar.button("Find My Beachhead", type="primary")

# ============ Helpers ============
def normalize_urls(text: str) -> List[str]:
    return list(set(re.findall(r'https?://[\w\.-/%\?\=&\+#]+', text)))

def run_tavily_query(q: str) -> str:
    """Run a single query through the Tavily-enabled agent and return bullet lines."""
    try:
        out = EXECUTOR.invoke({"input": q})["output"]
    except Exception as e:
        out = f"- Search error for '{q}': {e}"
    return out

def research_dimension(segment: str, dim: str, queries: List[str], log_area: st.delta_generator.DeltaGenerator) -> str:
    """Run the two queries for one BUTA dimension and render live output; return collected notes text."""
    notes = f"\n=== {segment} :: {dim} ===\n"
    for q in queries:
        q_box = log_area.container(border=True)
        q_box.caption(f"{dim} • Searching: {q}")
        bullets = run_tavily_query(q)
        q_box.write(bullets)
        notes += f"Query: {q}\n{bullets}\n"
    return notes

# ============ Main UX ============
st.title("🧭 BUTA Beachhead Finder — Iterative & Transparent")
st.write("We’ll iteratively **think → search → adapt** using the web until we converge on a strong beachhead—or we stop after 3 rounds. You’ll see every step and source.")

if not run:
    st.subheader("How this works")
    st.markdown(
        """
**Step 1.** You enter the problem and your solution (only these are required).  
**Step 2.** The agent proposes a specific candidate beachhead and a focused search plan.  
**Step 3.** It researches across **B**udget, **U**rgency, **T**op-3 Fit, **A**ccess using Tavily (with sources).  
**Step 4.** It evaluates the candidate. If fit is weak, it adapts and tries another segment.  
**Step 5.** It stops early on a strong fit, or after 3 rounds maximum, and prints a clear BUTA report.
"""
    )
    st.markdown("---")
    st.markdown("### What is BUTA?")
    st.markdown(
        "- **Budget** — Ability & willingness to pay\n"
        "- **Urgency** — Near-term reason to adopt\n"
        "- **Top-3 Fit** — Are you plausibly a top solution for their job?\n"
        "- **Access** — Can you reach and convert them?"
    )
    # Reference images (always visible) — UPDATED PARAM
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
    st.stop()

# Validate required fields
if not problem.strip() or not solution.strip():
    st.error("Please fill in both required fields: Problem and Solution.")
    st.stop()

# Overview panel (plain text)
st.subheader("Idea Overview")
overview_lines = [
    f"**Problem:** {problem}",
    f"**Solution:** {solution}",
]
if industry: overview_lines.append(f"**Industry:** {industry}")
if target_geos: overview_lines.append(f"**Target Geography:** {target_geos}")
if buyer_roles: overview_lines.append(f"**Buyer/User Roles:** {buyer_roles}")
if price_point: overview_lines.append(f"**Price Point:** {price_point}")
if evidence: overview_lines.append(f"**Evidence/Traction:** {evidence}")
if competitors: overview_lines.append(f"**Known Competitors:** {competitors}")
if constraints: overview_lines.append(f"**Constraints:** {constraints}")
st.write("\n\n".join(overview_lines))
st.markdown("---")

# Iterative loop
attempt_summaries: List[Dict[str, Any]] = []
final_recommendations: List[Dict[str, Any]] = []
progress = st.progress(0, text="Starting…")
log_area = st.container()  # live logs

previous_attempts_text = ""

for round_idx in range(1, max_rounds + 1):
    progress.progress((round_idx-1)/max_rounds, text=f"Round {round_idx} of {max_rounds}: planning…")

    # ---- Plan next candidate ----
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

Previous attempts & findings (if any):
{previous_attempts_text}
"""
    plan_msg = [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=plan_input)]
    plan_raw = LLM.invoke(plan_msg)
    plan = PLANNER_SCHEMA.invoke(plan_raw)

    if plan.get("decision") == "stop":
        with log_area:
            st.subheader(f"Round {round_idx} — Planner Decision: Stop")
            st.write(f"**Reason:** {plan.get('reason','')}")
            if plan.get("candidate"):
                st.write(f"**Best available candidate to present:** {plan['candidate']}")
        break

    candidate = plan["candidate"]
    why_this = plan["why_this"]
    plan_queries = plan["plan"]

    # Show plan in plain text
    with log_area:
        st.subheader(f"Round {round_idx}: Candidate to Investigate")
        st.write(f"**Segment:** {candidate}")
        st.write(f"**Why this next:** {why_this}")
        st.write("**What we’ll research now (BUTA):** Budget, Urgency, Top-3 Fit, Access")
    progress.progress((round_idx-1)/max_rounds, text=f"Round {round_idx}: researching {candidate} …")

    # ---- Research with Tavily for each dimension ----
    round_notes = ""
    for dim in ["Budget","Urgency","Top-3 Fit","Access"]:
        queries = plan_queries.get(dim, [])[:2]
        dim_header = log_area.container()
        dim_header.markdown(f"##### {dim}")
        round_notes += research_dimension(candidate, dim, queries, dim_header)

    # ---- Evaluate BUTA for this candidate ----
    eval_input = f"Candidate: {candidate}\n\nResearch notes:\n{round_notes}"
    eval_msg = [SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=eval_input)]
    eval_raw = LLM.invoke(eval_msg)
    evaluation = EVALUATOR_SCHEMA.invoke(eval_raw)

    # Render evaluation (plain text)
    with log_area:
        st.markdown("**Preliminary BUTA Scores:**")
        cols = st.columns(4)
        mapping = {"High":"🟢 High","Medium":"🟡 Medium","Low":"🔵 Low"}
        b = evaluation["BUTA"]["budget"]["rating"]
        u = evaluation["BUTA"]["urgency"]["rating"]
        t = evaluation["BUTA"]["top_fit"]["rating"]
        a = evaluation["BUTA"]["access"]["rating"]
        cols[0].metric("Budget", mapping.get(b,b))
        cols[1].metric("Urgency", mapping.get(u,u))
        cols[2].metric("Top-3 Fit", mapping.get(t,t))
        cols[3].metric("Access", mapping.get(a,a))
        st.write(evaluation.get("notes",""))

    attempt_summaries.append({
        "segment": candidate,
        "eval": evaluation,
        "notes": round_notes
    })

    # Check decision
    if evaluation["decision"] == "good_fit":
        with log_area:
            st.success("Strong fit found. Stopping further rounds.")
        final_recommendations.append({"segment": candidate, "eval": evaluation})
        break
    else:
        # Not yet — include brief summary into previous_attempts_text for the next planner round
        previous_attempts_text += f"\nAttempt {round_idx} on '{candidate}':\n"
        previous_attempts_text += f"B:{b} U:{u} T:{t} A:{a}\n"
        previous_attempts_text += "Key notes:\n" + evaluation.get("notes","") + "\n"
        # Keep the best-so-far (highest number of High/Medium) to present at the end even if we exhaust
        final_recommendations.append({"segment": candidate, "eval": evaluation})

    progress.progress(round_idx/max_rounds, text=f"Round {round_idx} complete.")

# If exhausted all rounds without "good_fit", we proceed with the best-so-far (top 1–2 by score)
def score_eval(ev: Dict[str,Any]) -> int:
    rank = {"High":2,"Medium":1,"Low":0}
    return sum(rank[ev["eval"]["BUTA"][k]["rating"]] for k in ["budget","urgency","top_fit","access"])
final_recommendations = sorted(final_recommendations, key=score_eval, reverse=True)[:2]

st.markdown("---")
st.subheader("Final BUTA Report")

# Build final readable report via LLM (no JSON shown to user)
idea_overview = "\n".join(overview_lines)
evals_text = ""
for rec in final_recommendations:
    seg = rec["segment"]
    ev = rec["eval"]["BUTA"]
    evals_text += f"\nSegment: {seg}\n"
    for k in ["budget","urgency","top_fit","access"]:
        evals_text += f"- {k.title()} rating: {rec['eval']['BUTA'][k]['rating']}\n"
        # Collect a couple of evidence lines with links for readability
        ev_bullets = rec['eval']['BUTA'][k]['evidence'][:3]
        ev_sources = rec['eval']['BUTA'][k]['sources'][:4]
        for line in ev_bullets:
            evals_text += f"  • {line}\n"
        if ev_sources:
            evals_text += "  sources: " + ", ".join(ev_sources) + "\n"

final_msg = [
    SystemMessage(content=FINAL_WRITER_PROMPT.format(idea_overview=idea_overview, evaluations=evals_text))
]
final_raw = LLM.invoke(final_msg)
final_text = final_raw.content

st.write(final_text)

# Meta footer: method & unique sources count
all_notes = "\n".join([a["notes"] for a in attempt_summaries])
unique_links = normalize_urls(all_notes)
st.caption(f"Method: Iterative BUTA search with GPT-4.1 + Tavily. Rounds run: {len(attempt_summaries)}. Unique sources referenced: {len(unique_links)}.")
