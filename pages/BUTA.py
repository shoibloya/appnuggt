import streamlit as st
import os
from typing import Dict, Any, List

# LangChain / OpenAI / Tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="BUTA Beachhead Finder", layout="wide")

parser = JsonOutputParser()
str_parser = StrOutputParser()

# Secrets -> env vars
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets["tapiKey"]
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["apiKey"]

# LLM + Tool
model = ChatOpenAI(model="gpt-4.1", temperature=0)
tools = [
    TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=True,
        search_depth="advanced",
    )
]

# ---------------------------------------------------------------------------
# Sidebar: Collect Inputs about the Idea
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Your Idea (Student Inputs)")
    company_type = st.selectbox(
        "Company Stage", ["Well-known", "Greenfield Entrepreneurship"]
    )
    idea_name = st.text_input("Product/Startup Name (optional)", "")
    industry = st.text_input("Industry / Category", "")
    problem = st.text_area("Problem / Job-to-be-Done", "")
    solution = st.text_area("Solution & Key Differentiators", "")
    target_geos = st.text_input("Target Geography (e.g., US, SEA, Global)", "")
    buyer_roles = st.text_input(
        "Buyer / User Roles (comma-separated, e.g., Parents, CFOs, Sales Ops)", ""
    )
    price_point = st.text_input("Expected Price Point (e.g., $29/mo, $5k/yr, $199)", "")
    traction = st.text_area("Evidence/Traction (beta users, pilots, waitlist, etc.)", "")
    known_competitors = st.text_area("Known Alternatives/Competitors", "")
    constraints = st.text_area("Constraints (go-to-market, compliance, etc.)", "")
    num_hypotheses = st.slider("How many beachhead hypotheses?", 2, 6, 3)

    if st.button("Run BUTA Analysis", type="primary"):
        st.session_state["student_input"] = {
            "company_type": company_type,
            "idea_name": idea_name,
            "industry": industry,
            "problem": problem,
            "solution": solution,
            "target_geos": target_geos,
            "buyer_roles": buyer_roles,
            "price_point": price_point,
            "traction": traction,
            "known_competitors": known_competitors,
            "constraints": constraints,
            "num_hypotheses": num_hypotheses,
        }

# ---------------------------------------------------------------------------
# Helper: Transparent Renderer for Final BUTA Report
# ---------------------------------------------------------------------------
def render_buta_report(expander, data: Dict[str, Any]):
    report = data["buta_report"]
    with expander:
        st.header("BUTA Beachhead Report")

        # Ranked Recommendations
        st.subheader("Ranked Beachhead Recommendations")
        for i, seg in enumerate(report.get("ranked_recommendations", []), start=1):
            st.write(f"{i}. **{seg}**")

        st.markdown("---")

        # Candidates
        for cand in report["candidate_beachheads"]:
            st.subheader(f"Candidate: {cand['segment']}")
            buta = cand["BUTA"]

            cols = st.columns(4)
            badge_map = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔵 Low"}

            with cols[0]:
                st.markdown("**Budget**")
                st.write(badge_map.get(buta["budget"]["rating"], buta["budget"]["rating"]))
                for ev in buta["budget"]["evidence"]:
                    st.write(f"- {ev}")
                if buta["budget"].get("sources"):
                    st.caption("Sources:")
                    for s in buta["budget"]["sources"]:
                        st.write(f"- {s}")

            with cols[1]:
                st.markdown("**Urgency**")
                st.write(badge_map.get(buta["urgency"]["rating"], buta["urgency"]["rating"]))
                for ev in buta["urgency"]["evidence"]:
                    st.write(f"- {ev}")
                if buta["urgency"].get("sources"):
                    st.caption("Sources:")
                    for s in buta["urgency"]["sources"]:
                        st.write(f"- {s}")

            with cols[2]:
                st.markdown("**Top-3 Fit**")
                st.write(badge_map.get(buta["top_fit"]["rating"], buta["top_fit"]["rating"]))
                for ev in buta["top_fit"]["evidence"]:
                    st.write(f"- {ev}")
                if buta["top_fit"].get("sources"):
                    st.caption("Sources:")
                    for s in buta["top_fit"]["sources"]:
                        st.write(f"- {s}")

            with cols[3]:
                st.markdown("**Access**")
                st.write(badge_map.get(buta["access"]["rating"], buta["access"]["rating"]))
                for ev in buta["access"]["evidence"]:
                    st.write(f"- {ev}")
                if buta["access"].get("sources"):
                    st.caption("Sources:")
                    for s in buta["access"]["sources"]:
                        st.write(f"- {s}")

            st.info(f"**Go-To-Market Hypothesis:** {cand['go_to_market_hypothesis']}")
            if cand.get("risks"):
                st.warning("Risks:")
                for r in cand["risks"]:
                    st.write(f"- {r}")

            if cand.get("next_validation_steps"):
                st.success("Next Validation Steps:")
                for step in cand["next_validation_steps"]:
                    st.write(f"- {step}")

            st.write(f"**Overall Recommendation:** {cand['overall_recommendation']}")
            st.markdown("---")

        # Competition + Methodology
        if report.get("notes_on_competition"):
            st.subheader("Notes on Competition")
            st.write(report["notes_on_competition"])

        if report.get("methodology_and_source_count"):
            st.caption(
                f"Methodology/Source Count: {report['methodology_and_source_count']}"
            )

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Search Agent: Turns each search string into sourced findings (bullets + links).
search_agent_system = """
You are a market research analyst applying the BUTA beachhead framework:
B = Budget (do they have willingness/ability to pay?),
U = Urgency (is there a compelling near-term reason to adopt?),
T = Top-3 Fit (is the solution plausibly among the top options for this segment’s job-to-be-done?),
A = Access (how easily can we reach the segment? channels, lists, partnerships).

Use Tavily to gather recent, credible information. For EACH search query:
- Return 3–6 concise bullets.
- After each bullet, include a source link in parentheses. Prefer primary data, analyst or industry reports, reputable news, government/NGO stats.
- Avoid paywalled content when possible; if paywalled, still include the citation.
- Keep it tight and factual.

Important:
- Do not invent sources or content.
- If results are mixed or uncertain, state that explicitly.
- Do not include commentary outside of bullets and links.
"""

# Generate Beachhead Hypotheses from student inputs.
hypothesis_prompt = """
You are an expert in identifying startup beachhead segments using the BUTA framework.

Student inputs:
- Company Stage: {company_type}
- Idea Name: {idea_name}
- Industry/Category: {industry}
- Problem/JTBD: {problem}
- Solution & Differentiators: {solution}
- Target Geographies: {target_geos}
- Buyer/User Roles: {buyer_roles}
- Expected Price Point: {price_point}
- Evidence/Traction: {traction}
- Known Competitors: {known_competitors}
- Constraints: {constraints}

Using the examples of **correct and incorrect beachheads** (e.g., Google Wave’s misaligned developer focus; Tesla Roadster’s initial high-income tech-loving innovators who valued performance), propose {num_hypotheses} plausible, specific BEACHHEAD SEGMENTS for this idea (e.g., “Dental clinics with 2–5 chairs in the US North-East using Open Dental”; “Series B SaaS startups with >50 SDRs using Salesforce”).

Return STRICT JSON ONLY in the format:

{{
  "hypotheses": [
    {{
      "segment_name": "Precise segment name",
      "why_promising": "Short rationale tied to B, U, T, A",
      "assumptions_to_test": ["Assumption 1", "Assumption 2", "Assumption 3"]
    }}
  ]
}}
"""

# Generate research searches for each hypothesis and BUTA category.
buta_searches_prompt = """
Design web search queries for each beachhead hypothesis and each BUTA dimension.

For each hypothesis segment:
- Provide 2 distinct queries per dimension (Budget, Urgency, Top-3 Fit, Access).
- Make queries specific to the segment and geography.
- Avoid overlap; target facts, TAM/SOM proxies, purchasing behavior, alternatives, switching drivers, buying channels, lists/directories, partnerships, etc.

Output STRICT JSON ONLY:
{{
  "searches": [
    {{
      "segment": "Segment name",
      "queries": {{
        "Budget": [ "query 1", "query 2" ],
        "Urgency": [ "query 1", "query 2" ],
        "Top-3 Fit": [ "query 1", "query 2" ],
        "Access": [ "query 1", "query 2" ]
      }}
    }}
  ]
}}
"""

# Final synthesis to BUTA Report with rankings and sources.
final_buta_report_prompt = """
You are finalizing a BUTA Beachhead Report.

Use ONLY the collated research notes below (which include bullet points and explicit source links) to score each candidate segment. If evidence is weak or mixed, say so.

Scoring rules:
- rating: "High", "Medium", or "Low" per dimension (Budget, Urgency, Top-3 Fit, Access).
- evidence: 2–5 short justifications drawn from the research notes.
- sources: 2–6 URLs from the notes per dimension (no invented links).

Also include:
- go_to_market_hypothesis: the most realistic initial approach to reach & convert this segment.
- risks: key concerns or gaps.
- next_validation_steps: concrete scrappy experiments (interviews, list-building, cold outreach, landing test, channel test).
- overall_recommendation: "Pursue", "Defer", or "Monitor".

Return STRICT JSON ONLY in this schema:

{{
  "buta_report": {{
    "candidate_beachheads": [
      {{
        "segment": "Segment name",
        "BUTA": {{
          "budget":    {{"rating": "High|Medium|Low", "evidence": ["..."], "sources": ["..."]}},
          "urgency":   {{"rating": "High|Medium|Low", "evidence": ["..."], "sources": ["..."]}},
          "top_fit":   {{"rating": "High|Medium|Low", "evidence": ["..."], "sources": ["..."]}},
          "access":    {{"rating": "High|Medium|Low", "evidence": ["..."], "sources": ["..."]}}
        }},
        "go_to_market_hypothesis": "Short plan",
        "risks": ["..."],
        "next_validation_steps": ["..."],
        "overall_recommendation": "Pursue|Defer|Monitor"
      }}
    ],
    "ranked_recommendations": ["Segment A", "Segment B", "Segment C"],
    "notes_on_competition": "Short synthesis of alternatives from the notes.",
    "methodology_and_source_count": {{"notes_chars": INT, "unique_sources": INT}}
  }}
}}
"""

# Build a ChatPromptTemplate for the tool-enabled search agent
search_prompt = ChatPromptTemplate(
    input_variables=["agent_scratchpad", "input"],
    messages=[
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=search_agent_system)),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="{input}")),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)

# ---------------------------------------------------------------------------
# Main Flow
# ---------------------------------------------------------------------------
st.title("🧭 BUTA Beachhead Finder")
st.caption("Identify, research, and rank your most promising beachheads with Budget–Urgency–Top-3 Fit–Access scoring. Powered by GPT-4.1 + Tavily.")

# Optional: show reference slides/images if available
with st.expander("What is BUTA? (reference)", expanded=False):
    st.markdown(
        "- **B**udget — Willingness & ability to pay\n"
        "- **U**rgency — Compelling near-term reason to adopt\n"
        "- **T**op-3 Fit — Are you plausibly a top option for the job-to-be-done?\n"
        "- **A**ccess — Do you have a clear path to reach/convert them?"
    )
    # Show provided images if present
    for p in [
        "/mnt/data/83246a71-57ac-41b1-b7b2-a189c1564b83.png",
        "/mnt/data/f4e0e3f7-fb29-40d3-9a15-e6ac9fd54a26.png",
        "/mnt/data/e6ed0f72-3758-4ab2-9232-2f9bddb29922.png",
        "/mnt/data/c419f54e-6bd4-4da6-9c8f-839754b71f78.png",
        "/mnt/data/81b8e7b3-5e9a-4ae1-ba5f-876ebdebd4a0.png",
    ]:
        try:
            st.image(p)
        except Exception:
            pass

if "student_input" not in st.session_state:
    st.markdown(
        """
### How this works
1) Fill the **left sidebar** with your idea details and click **Run BUTA Analysis**.  
2) The app will:
   - Propose concrete **beachhead hypotheses** (transparent JSON).
   - For each hypothesis, run **Tavily web research** across **B**, **U**, **T**, **A** with live updates and **sources**.
   - Synthesize a final, **ranked BUTA report** with recommendations, risks, and next steps.
3) You’ll see progress in real time—no long silent wait.
"""
    )
    st.stop()

student = st.session_state["student_input"]

# ---------------------------------------------------------------------------
# STEP 1: Hypothesize Beachheads
# ---------------------------------------------------------------------------
st.header("Step 1 — Generate Beachhead Hypotheses")
hypo_msg = SystemMessage(
    content=hypothesis_prompt.format(
        company_type=student["company_type"],
        idea_name=student["idea_name"],
        industry=student["industry"],
        problem=student["problem"],
        solution=student["solution"],
        target_geos=student["target_geos"],
        buyer_roles=student["buyer_roles"],
        price_point=student["price_point"],
        traction=student["traction"],
        known_competitors=student["known_competitors"],
        constraints=student["constraints"],
        num_hypotheses=student["num_hypotheses"],
    )
)
hypo_result = model.invoke([hypo_msg])
hypotheses_data = parser.invoke(hypo_result)

hypo_exp = st.expander("Generated Hypotheses (JSON)", expanded=True)
with hypo_exp:
    st.json(hypotheses_data)

# Extract hypothesis names for downstream steps
hypo_names: List[str] = [h["segment_name"] for h in hypotheses_data.get("hypotheses", [])]

# ---------------------------------------------------------------------------
# STEP 2: Build BUTA Search Plan per Hypothesis
# ---------------------------------------------------------------------------
st.header("Step 2 — Build BUTA Research Plan")
buta_plan_msg = SystemMessage(
    content=buta_searches_prompt
)
buta_plan_human = HumanMessage(
    content=f"Hypotheses to research:\n{hypotheses_data}\n\nAlso consider known competitors:\n{student['known_competitors']}\nGeos: {student['target_geos']}\nPrice point: {student['price_point']}"
)
plan_result = model.invoke([buta_plan_msg, buta_plan_human])
plan_data = parser.invoke(plan_result)

plan_exp = st.expander("BUTA Search Plan (JSON)", expanded=False)
with plan_exp:
    st.json(plan_data)

# ---------------------------------------------------------------------------
# STEP 3: Execute Research with Tavily (transparent progress)
# ---------------------------------------------------------------------------
st.header("Step 3 — Run Web Research (Tavily)")

# Prepare the tool-enabled agent
agent = create_openai_tools_agent(model, tools, search_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

research_notes = ""  # Collated text fed into the final synthesis prompt

# Build expanders per hypothesis and per dimension
dimensions = ["Budget", "Urgency", "Top-3 Fit", "Access"]

for seg in plan_data.get("searches", []):
    seg_name = seg["segment"]
    seg_exp = st.expander(f"Hypothesis: {seg_name}", expanded=True)
    with seg_exp:
        for dim in dimensions:
            dim_queries = seg["queries"].get(dim, [])
            dim_box = st.container(border=True)
            with dim_box:
                st.markdown(f"**{dim} — Research**")
                for q in dim_queries:
                    loader = st.empty()
                    loader.info(f"Searching: {q} ⏳")
                    # Invoke the agent with the query; the agent will use the Tavily tool
                    try:
                        out = agent_executor.invoke({"input": q})["output"]
                    except Exception as e:
                        out = f"- Could not complete search due to error: {e}"
                    loader.success(q)
                    st.write(out)
                    research_notes += f"\n=== {seg_name} :: {dim} ===\nQuery: {q}\n{out}\n"

# ---------------------------------------------------------------------------
# STEP 4: Synthesize Final BUTA Report (strict JSON + sources)
# ---------------------------------------------------------------------------
st.header("Step 4 — Synthesize BUTA Report")
syn_msgs = [
    SystemMessage(content=final_buta_report_prompt),
    HumanMessage(
        content=f"Collated Research Notes (include bullets with links exactly as gathered):\n{research_notes}"
    ),
]
final_result = model.invoke(syn_msgs)
buta_report = parser.invoke(final_result)

final_exp = st.expander("Final BUTA Report (JSON)", expanded=True)
st.toast("BUTA analysis complete. See the report below.", icon="✅")
with final_exp:
    st.json(buta_report)

# Pretty renderer
view_exp = st.expander("Readable Report", expanded=True)
render_buta_report(view_exp, buta_report)
