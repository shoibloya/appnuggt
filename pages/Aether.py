import streamlit as st
import os
import re
from datetime import datetime

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain import hub  # kept for consistency with your other tools (not required)
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Tavily
from langchain_community.tools.tavily_search import TavilySearchResults


# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="Aether-AI Location Strategy Workshop", layout="wide")

parser = JsonOutputParser()
str_parser = StrOutputParser()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets["tapiKey"]

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["apiKey"]


# -----------------------------
# Helpers
# -----------------------------
def _is_ctx_len_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("context length" in s) or ("maximum context length" in s) or ("context_length_exceeded" in s)

def _shrink_to_first_third(text: str) -> str:
    if not text:
        return text
    return text[:max(1, len(text) // 3)]

def _strip_code_fences(text: str) -> str:
    # If model wraps JSON in ```json ... ```
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()

def _safe_json_parse(model_output: str):
    cleaned = _strip_code_fences(model_output)
    return parser.parse(cleaned)

def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# -----------------------------
# Models + Tools
# -----------------------------
# Main model for planning + synthesis
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Research tool (tuned to reduce bloat but still include links)
tools = [
    TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=False,
        search_depth="advanced"
    )
]


# -----------------------------
# Sidebar Inputs (Students)
# -----------------------------
with st.sidebar:
    st.title("Workshop Inputs")

    strategy = st.radio(
        "Which setup are you evaluating?",
        ["Hub-First (Singapore)", "Spoke-First (Jakarta)", "Hybrid (Singapore + Jakarta)"]
    )

    customer_type = st.selectbox(
        "First beachhead customer type",
        ["Shipping lines", "Freight forwarders", "Port operators", "3PL / Logistics providers"]
    )

    pilot_geo = st.selectbox(
        "Primary pilot geography",
        ["Singapore", "Indonesia", "Southeast Asia (multi-country)", "Global"]
    )

    col_a, col_b = st.columns(2)
    with col_a:
        headcount_now = st.number_input("Current headcount", min_value=1, max_value=500, value=6, step=1)
    with col_b:
        hires_12m = st.number_input("Planned hires (next 12 months)", min_value=0, max_value=500, value=6, step=1)

    runway_months = st.number_input("Runway (months)", min_value=1, max_value=60, value=12, step=1)

    sales_motion = st.selectbox(
        "Sales motion (first 12 months)",
        ["Direct enterprise", "Partner-led", "JV / Strategic alliance"]
    )

    ip_sensitivity = st.selectbox(
        "IP sensitivity (how critical is protecting core algorithms/data?)",
        ["Low", "Medium", "High"]
    )

    burn_constraint = st.text_input("Max monthly burn (optional)", "")

    series_a_goal = st.toggle("Optimize for Series A in 12–18 months", value=True)

    team_notes = st.text_area(
        "Team notes / constraints (optional)",
        placeholder="Anything important? e.g., must hire fast, investor wants SG entity, need proximity to customers..."
    )

    run_btn = st.button("Generate Workshop Report", use_container_width=True)


# -----------------------------
# Prompts
# -----------------------------
# (1) Search plan prompt (JSON)
sys_prompt_search_plan = f"""
You are designing a fast research plan for a workshop team making a location strategy decision for a seed-stage maritime AI startup.

Decision being evaluated: {strategy}
Beachhead customer type: {customer_type}
Primary pilot geography: {pilot_geo}
Sales motion: {sales_motion}
IP sensitivity: {ip_sensitivity}
Runway: {runway_months} months
Team: {headcount_now} now, hiring {hires_12m} in next 12 months
{'Burn constraint: ' + burn_constraint if burn_constraint else ''}
Series A goal in 12–18 months: {"Yes" if series_a_goal else "No"}
{'Team notes: ' + team_notes if team_notes else ''}

Create a set of specific Google-style search queries (high precision, not generic) to compare Singapore vs Jakarta/Indonesia where relevant
and to support the frameworks below. Queries should be unique (no overlap) and tailored to the inputs above.

Return STRICT JSON in the exact structure below (no markdown, no extra keys):

{{
  "searches": {{
    "Ecosystem Audit — Government/Policy/IP (Gardener)": [
      {{"search":"...", "importance":"..."}},
      {{"search":"...", "importance":"..."}}
    ],
    "Ecosystem Audit — Capital (Water)": [
      {{"search":"...", "importance":"..."}},
      {{"search":"...", "importance":"..."}}
    ],
    "Ecosystem Audit — Talent (Sun)": [
      {{"search":"...", "importance":"..."}},
      {{"search":"...", "importance":"..."}}
    ],
    "GTM & Market Access (Hub/Spoke Fit)": [
      {{"search":"...", "importance":"..."}},
      {{"search":"...", "importance":"..."}}
    ],
    "Operating Setup & Costs": [
      {{"search":"...", "importance":"..."}},
      {{"search":"...", "importance":"..."}}
    ]
  }}
}}
"""

# (2) Research agent prompt (per search query, used with Tavily tool)
sys_prompt_research_agent = f"""
You are helping executive education participants do fast, credible research.

Context:
- Decision: {strategy}
- Customer: {customer_type}
- Pilot geography: {pilot_geo}
- Sales motion: {sales_motion}
- IP sensitivity: {ip_sensitivity}
- Runway: {runway_months} months
- Team: {headcount_now} now, hiring {hires_12m} next 12 months

Task:
For the user’s query, return 3–5 concise bullet points that are directly useful for deciding where to incorporate and where to base HQ/R&D/Sales.
Prefer numbers when possible (cost ranges, timelines, counts, rankings, etc.). If you can’t find reliable numbers, say so clearly.

Every bullet MUST include a source link.
Do not write long paragraphs. Be practical and decision-oriented.
"""

research_prompt = ChatPromptTemplate(
    input_variables=["agent_scratchpad", "input"],
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=[], template=sys_prompt_research_agent)
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["input"], template="{input}")
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)

# (3) Final synthesis prompt (MARKDOWN ONLY)
sys_prompt_synthesis = f"""
You are producing a workshop-ready report for a student team. You must tailor the report to the team's inputs and the evidence provided.

Team inputs:
- Strategy evaluated: {strategy}
- Beachhead customer: {customer_type}
- Pilot geography: {pilot_geo}
- Sales motion: {sales_motion}
- IP sensitivity: {ip_sensitivity}
- Current headcount: {headcount_now}
- Planned hires (12 months): {hires_12m}
- Runway: {runway_months} months
{'- Burn constraint: ' + burn_constraint if burn_constraint else ''}
- Series A goal in 12–18 months: {"Yes" if series_a_goal else "No"}
{'- Team notes: ' + team_notes if team_notes else ''}

Rules:
- Output MARKDOWN ONLY. No JSON. No code blocks.
- Use evidence from the provided research notes. Do not invent facts.
- When you make an evidence-based claim, include a link (reuse the links already present in the research notes).
- If evidence is missing, label it as an assumption.

Your markdown MUST include these headings, in this order:

## Recommendation
## Ecosystem Audit (Garden City)
## Hub & Spoke GTM
## 24-Month Roadmap
## Top 3 Risks & Mitigations
## Triple Helix Alignment
## What Would Change Our Decision
## Sources

Formatting requirements:
- In Ecosystem Audit, include a markdown table comparing Singapore vs Jakarta/Indonesia (and a short “Hybrid implication” note if strategy is Hybrid).
  Rows: Government/Policy/IP (Gardener), Capital (Water), Talent (Sun).
- In Roadmap, use phases: 0–3 months, 3–6 months, 6–12 months, 12–24 months.
- In Risks, present a compact markdown table with columns: Risk | Early warning sign | Mitigation | Residual risk.
- In Sources, list unique links only (bulleted).
"""


# -----------------------------
# Main UI
# -----------------------------
st.title("Aether-AI Location Strategy Workshop Assistant")
st.caption("Generates a workshop report aligned to the case frameworks. " + _now_str())

if not run_btn and "last_report_md" in st.session_state:
    st.success("Showing your most recent report.")
    st.markdown(st.session_state["last_report_md"])
    st.stop()

if not run_btn:
    st.info(
        "Enter your team’s inputs in the sidebar and click **Generate Workshop Report**.\n\n"
        "You’ll see progress updates as the report is built."
    )
    st.stop()


# -----------------------------
# Run Pipeline
# -----------------------------
# Friendly status area (no “technical” talk)
status_box = st.empty()
progress_box = st.empty()

status_box.info("Starting up… getting your workshop report ready.")
progress = st.progress(0)

# Step 1: Create search plan
status_box.info("Step 1 of 3 — Planning what to look up.")
progress.progress(10)

try:
    plan_result = model.invoke([
        SystemMessage(content=sys_prompt_search_plan),
        HumanMessage(content="Create the research plan now.")
    ])
    search_plan = _safe_json_parse(plan_result.content)
except Exception as e:
    st.error("Something went wrong while planning the research. Please try again.")
    st.exception(e)
    st.stop()

progress.progress(20)
status_box.info("Step 2 of 3 — Gathering useful information from the web (this can take a bit).")

# Create tools agent for research
agent = create_openai_tools_agent(model, tools, research_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# Prepare expanders to show progress + results
category_order = list(search_plan.get("searches", {}).keys())

expander_map = {}
first_open = True
for cat in category_order:
    expander_map[cat] = st.expander(f"Research results: {cat}", expanded=first_open)
    first_open = False

# Track all research for synthesis
research_notes = []
all_links = set()

# Helper: pull links out of markdown-ish text (best-effort)
link_regex = re.compile(r"\((https?://[^)]+)\)|\bhttps?://\S+")

total_queries = sum(len(search_plan["searches"][c]) for c in category_order) or 1
done_queries = 0

for cat in category_order:
    queries = search_plan["searches"][cat]
    with expander_map[cat]:
        st.write("We’re pulling a few relevant sources and turning them into decision-ready notes…")
        for item in queries:
            q = item["search"]
            done_queries += 1

            # Friendly “alive” updates
            progress_pct = 20 + int(60 * (done_queries / total_queries))
            progress.progress(min(progress_pct, 80))
            status_box.info(f"Gathering notes ({done_queries}/{total_queries})…")

            loader = st.empty()
            loader.info(f"Looking up: {q} ⏳")

            # Run research with context-length fallback
            try:
                out = agent_executor.invoke({"input": q})["output"]
            except Exception as e:
                if _is_ctx_len_error(e):
                    # Smaller tool payload fallback
                    tools_small = [
                        TavilySearchResults(
                            max_results=3,
                            include_answer=True,
                            include_raw_content=False,
                            search_depth="advanced"
                        )
                    ]
                    agent_small = create_openai_tools_agent(model, tools_small, research_prompt)
                    agent_executor_small = AgentExecutor(agent=agent_small, tools=tools_small, verbose=False)
                    out = agent_executor_small.invoke({"input": q})["output"]
                    out = _shrink_to_first_third(out)
                else:
                    raise

            loader.info(q)
            st.success(out)

            # Collect notes
            research_notes.append(f"### {cat}\nQuery: {q}\n{out}\n")

            # Best-effort link collection
            for m in link_regex.findall(out):
                if isinstance(m, tuple):
                    for part in m:
                        if part and part.startswith("http"):
                            all_links.add(part)
                elif m and isinstance(m, str) and m.startswith("http"):
                    all_links.add(m)

progress.progress(85)
status_box.info("Step 3 of 3 — Writing a clear workshop report you can present.")


# Step 3: Synthesis to MARKDOWN
research_blob = "\n\n".join(research_notes)

# Shrink-on-context loop
md_report = None
blob_tmp = research_blob
while True:
    try:
        synth = model.invoke([
            SystemMessage(content=sys_prompt_synthesis),
            HumanMessage(content=f"Research notes (with links):\n\n{blob_tmp}")
        ])
        md_report = synth.content.strip()
        break
    except Exception as e:
        if _is_ctx_len_error(e):
            blob_tmp = _shrink_to_first_third(blob_tmp)
            continue
        else:
            st.error("Something went wrong while writing the report. Please try again.")
            st.exception(e)
            st.stop()

progress.progress(100)
status_box.success("Done! Your workshop report is ready.")

# Render report
st.session_state["last_report_md"] = md_report
st.markdown(md_report)

# Optional: quick copy area
with st.expander("Copy-friendly version", expanded=False):
    st.text_area("Markdown report", md_report, height=300)
