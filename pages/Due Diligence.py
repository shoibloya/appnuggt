import streamlit as st
import os

# Import necessary LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults

# Set up our parsers
parser = JsonOutputParser()
str_parser = StrOutputParser()

# Ensure environment variables are properly set
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets['tapiKey']

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets['apiKey']

# Create the main LLM model (GPT-4o) for the entire pipeline
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Create the tool for advanced searching (via TavilySearchResults)
tools = [TavilySearchResults(max_results=3, include_answer=True, include_raw_content=True, search_depth="advanced")]

# Set Streamlit Page Config
st.set_page_config(page_title="Due Diligence Dashboard", layout="wide")

# ----------------------------------------------------------------------------
# Sidebar: Collecting Input for Due Diligence
# ----------------------------------------------------------------------------
with st.sidebar:
    company_type = st.selectbox(
        "Is this a well-known company or greenfield entrepreneurship?",
        ["Well-known", "Greenfield Entrepreneurship"]
    )
    target_company = st.text_input("Target Company (e.g., Meta)", "")
    industry = st.text_input("Industry (e.g., Social Media)", "")
    company_description = st.text_area("Target Company Description (What does the company do?)", "")
    financial_data = st.text_area("Financial Data (Revenue, Profits, Cash Flow, etc.)", "")
    key_risks = st.text_area("Key Risks (Legal, Financial, Operational, etc.)", "")
    regulatory_issues = st.text_area("Legal/Regulatory Issues", "")
    valuation = st.text_area("Preliminary Valuation or Deal Rationale", "")
    red_flags = st.text_area("Red Flags or Concerns", "")
    additional_comments = st.text_area("Additional Comments or Observations", "")

    if st.button("Conduct Due Diligence", key="create_due_diligence_button"):
        # Store inputs into session state
        st.session_state['user_input'] = {
            'company_type': company_type,
            'target_company': target_company,
            'industry': industry,
            'company_description': company_description,
            'financial_data': financial_data,
            'key_risks': key_risks,
            'regulatory_issues': regulatory_issues,
            'valuation': valuation,
            'red_flags': red_flags,
            'additional_comments': additional_comments
        }

# ----------------------------------------------------------------------------
# Main Logic: Generating and Displaying Due Diligence
# ----------------------------------------------------------------------------

# Retrieve user input from session state
user_input = st.session_state.get('user_input', {})

# Decide how to refer to the company in prompts (if it's Greenfield, do not use the company name)
if user_input.get('company_type') == 'Greenfield Entrepreneurship':
    company_mention = "(Greenfield startup in the specified industry)"
else:
    company_mention = user_input.get('target_company', 'the company')

# ----------------------------------------------------------------------------
# System Prompt (1) Initial Search Agent Instruction
# ----------------------------------------------------------------------------
sys_prompt_search_agent = f"""
You are a senior investment analyst conducting high-level due diligence on {company_mention} 
in the {user_input.get('industry', 'industry')} sector.

Your goal is to gather critical information about the target company's industry, 
business model, financial standing, market position, legal/regulatory issues, and any red flags 
that could impact the decision to invest or acquire.

Leverage the details provided:
{'Company Description: ' + user_input.get('company_description') if user_input.get('company_description') else ''}
{'Financial Data: ' + user_input.get('financial_data') if user_input.get('financial_data') else ''}
{'Key Risks: ' + user_input.get('key_risks') if user_input.get('key_risks') else ''}
{'Regulatory Issues: ' + user_input.get('regulatory_issues') if user_input.get('regulatory_issues') else ''}
{'Preliminary Valuation: ' + user_input.get('valuation') if user_input.get('valuation') else ''}
{'Red Flags: ' + user_input.get('red_flags') if user_input.get('red_flags') else ''}
{'Additional Comments: ' + user_input.get('additional_comments') if user_input.get('additional_comments') else ''}

Gather external information and compile the top findings with relevant sources.
Return the insights as bullet points. For each point, provide a link or source reference.
"""

# ----------------------------------------------------------------------------
# System Prompt (2) Generate Category-Specific Searches
#     -- Adjust to omit the company name if it's a greenfield startup
# ----------------------------------------------------------------------------
if user_input.get('company_type') == 'Greenfield Entrepreneurship':
    # Focus on the industry, not the specific company name
    dd_categories_subject = f"greenfield startup in the {user_input.get('industry', 'industry')} industry"
else:
    dd_categories_subject = f"{user_input.get('target_company', 'the company')} in the {user_input.get('industry', 'industry')}"

sys_prompt_dd_categories = f"""
You are an expert in designing due diligence frameworks. Given the target {dd_categories_subject}, 
propose a set of Google searches to investigate each category of due diligence:

1. Financial Overview
2. Legal & Regulatory
3. Market & Competition
4. Operational Efficiency & Synergies
5. Cultural Fit & HR (if relevant)
6. Other Key Risks

In the following JSON format, provide recommended searches for each category that do not overlap. 
Each search query should be unique and support the broader due diligence process:

{{
  "searches": {{
    "Financial Overview": [
      {{
        "search": "Google search 1",
        "importance": "Why this search is important"
      }},
      {{
        "search": "Google search 2",
        "importance": "Why this search is important"
      }}
    ],
    "Legal & Regulatory": [
      {{
        "search": "Google search 1",
        "importance": "Why this search is important"
      }},
      {{
        "search": "Google search 2",
        "importance": "Why this search is important"
      }}
    ],
    "Market & Competition": [
      {{
        "search": "Google search 1",
        "importance": "Why this search is important"
      }},
      {{
        "search": "Google search 2",
        "importance": "Why this search is important"
      }}
    ],
    "Operational Efficiency & Synergies": [
      {{
        "search": "Google search 1",
        "importance": "Why this search is important"
      }},
      {{
        "search": "Google search 2",
        "importance": "Why this search is important"
      }}
    ],
    "Cultural Fit & HR": [
      {{
        "search": "Google search 1",
        "importance": "Why this search is important"
      }},
      {{
        "search": "Google search 2",
        "importance": "Why this search is important"
      }}
    ],
    "Other Key Risks": [
      {{
        "search": "Google search 1",
        "importance": "Why this search is important"
      }},
      {{
        "search": "Google search 2",
        "importance": "Why this search is important"
      }}
    ]
  }}
}}
"""

# ----------------------------------------------------------------------------
# Create the ChatPromptTemplate for the search agent
# ----------------------------------------------------------------------------
prompt = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input'],
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template=sys_prompt_search_agent
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['input'],
                template='{input}'
            )
        ),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ]
)

# ----------------------------------------------------------------------------
# Final System Prompt for Consolidated Due Diligence Report
# ----------------------------------------------------------------------------
sys_prompt_due_diligence_report = f"""
You are a senior investment analyst finalizing a Due Diligence Report on {company_mention} 
in the {user_input.get('industry', 'industry')} sector. Combine all discovered information 
(financial, legal, market, operational, cultural, etc.) and produce a comprehensive due diligence summary 
in the following JSON format. Use any relevant statistics or references if they appeared in the research. 
Be concise and direct, focusing on key data points. 

Strictly adhere to the JSON format below and do not output anything else:

{{
  "due_diligence_summary": {{
    "financial_overview": {{
      "key_points": [
        "Point 1",
        "Point 2"
      ],
      "risks": [
        "Risk 1",
        "Risk 2"
      ],
      "opportunities": [
        "Opportunity 1",
        "Opportunity 2"
      ]
    }},
    "legal_and_regulatory": {{
      "key_points": [
        "Point 1",
        "Point 2"
      ],
      "risks": [
        "Risk 1",
        "Risk 2"
      ],
      "compliance_overview": "Short description of major compliance considerations."
    }},
    "market_and_competition": {{
      "key_points": [
        "Point 1",
        "Point 2"
      ],
      "competitors": [
        "Competitor 1",
        "Competitor 2"
      ],
      "competitive_position": "Short narrative on the target's position in the market."
    }},
    "operational_efficiency_and_synergies": {{
      "key_points": [
        "Point 1",
        "Point 2"
      ],
      "potential_synergies": [
        "Synergy 1",
        "Synergy 2"
      ],
      "challenges": [
        "Challenge 1",
        "Challenge 2"
      ]
    }},
    "cultural_fit_and_hr": {{
      "key_points": [
        "Point 1",
        "Point 2"
      ],
      "integration_issues": [
        "Issue 1",
        "Issue 2"
      ],
      "hr_considerations": "Short narrative on HR or cultural integration factors."
    }},
    "other_key_risks": [
      "Risk A",
      "Risk B"
    ],
    "preliminary_recommendation": "High-level recommendation on whether to proceed with investment or acquisition, with reasoning."
  }}
}}
"""

# ----------------------------------------------------------------------------
# A Helper Function to Display the Final Due Diligence Report
# ----------------------------------------------------------------------------
def present_due_diligence(expander, data):
    with expander:
        st.header("Due Diligence Summary")

        # Financial Overview
        st.subheader("Financial Overview")
        for pt in data['due_diligence_summary']['financial_overview']['key_points']:
            st.write(f"- **Key Point:** {pt}")
        st.warning("Risks:")
        for risk in data['due_diligence_summary']['financial_overview']['risks']:
            st.write(f"- {risk}")
        st.success("Opportunities:")
        for opp in data['due_diligence_summary']['financial_overview']['opportunities']:
            st.write(f"- {opp}")

        st.markdown("---")

        # Legal & Regulatory
        st.subheader("Legal & Regulatory")
        for pt in data['due_diligence_summary']['legal_and_regulatory']['key_points']:
            st.write(f"- **Key Point:** {pt}")
        st.warning("Risks:")
        for risk in data['due_diligence_summary']['legal_and_regulatory']['risks']:
            st.write(f"- {risk}")
        st.info(f"**Compliance Overview:** {data['due_diligence_summary']['legal_and_regulatory']['compliance_overview']}")

        st.markdown("---")

        # Market & Competition
        st.subheader("Market & Competition")
        for pt in data['due_diligence_summary']['market_and_competition']['key_points']:
            st.write(f"- **Key Point:** {pt}")
        st.info("Competitors:")
        for comp in data['due_diligence_summary']['market_and_competition']['competitors']:
            st.write(f"- {comp}")
        st.success(f"**Competitive Position:** {data['due_diligence_summary']['market_and_competition']['competitive_position']}")

        st.markdown("---")

        # Operational Efficiency & Synergies
        st.subheader("Operational Efficiency & Synergies")
        for pt in data['due_diligence_summary']['operational_efficiency_and_synergies']['key_points']:
            st.write(f"- **Key Point:** {pt}")
        st.info("Potential Synergies:")
        for syn in data['due_diligence_summary']['operational_efficiency_and_synergies']['potential_synergies']:
            st.write(f"- {syn}")
        st.warning("Challenges:")
        for ch in data['due_diligence_summary']['operational_efficiency_and_synergies']['challenges']:
            st.write(f"- {ch}")

        st.markdown("---")

        # Cultural Fit & HR
        st.subheader("Cultural Fit & HR")
        for pt in data['due_diligence_summary']['cultural_fit_and_hr']['key_points']:
            st.write(f"- **Key Point:** {pt}")
        st.warning("Integration Issues:")
        for issue in data['due_diligence_summary']['cultural_fit_and_hr']['integration_issues']:
            st.write(f"- {issue}")
        st.info(f"**HR Considerations:** {data['due_diligence_summary']['cultural_fit_and_hr']['hr_considerations']}")

        st.markdown("---")

        # Other Key Risks
        st.subheader("Other Key Risks")
        for risk in data['due_diligence_summary']['other_key_risks']:
            st.write(f"- {risk}")

        st.markdown("---")

        # Preliminary Recommendation
        st.subheader("Preliminary Recommendation")
        st.success(data['due_diligence_summary']['preliminary_recommendation'])

# ----------------------------------------------------------------------------
# If user input is provided, generate the due diligence
# ----------------------------------------------------------------------------
if 'user_input' in st.session_state:
    # ------------------------------------------------------------------------
    # Step 1: Generate recommended searches for each category
    # ------------------------------------------------------------------------
    messages = [
        SystemMessage(content=sys_prompt_dd_categories),
        HumanMessage(content="Generate due diligence searches based on the provided details.")
    ]
    result = model.invoke(messages)
    answer = parser.invoke(result)

    # Create an agent with our search prompt
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # Prepare expanders for each category
    category_names = [
        "Financial Overview", 
        "Legal & Regulatory", 
        "Market & Competition",
        "Operational Efficiency & Synergies",
        "Cultural Fit & HR",
        "Other Key Risks"
    ]

    expanders = {}
    expand_it = True
    for cat in category_names:
        expanders[cat] = st.expander(f"Category: {cat}", expanded=expand_it)
        expand_it = False  # only first one expanded by default
    
    # Show a "collating insights" message initially in each expander
    for cat in category_names:
        with expanders[cat]:
            st.warning(f"Gathering insights on '{cat}'...")

    # Conduct the searches and compile results
    dd_summary_text = ""
    for category, searches in answer['searches'].items():
        dd_summary_text += f"\n=== {category} ==="
        with expanders[category]:
            for srch in searches:
                loader_placeholder = st.empty()
                loader_placeholder.info(f"Researching: {srch['search']} ⏳")

                # Use the agent executor with the search query
                output = agent_executor.invoke({"input": srch['search']})["output"]
                
                # Replace the loader message
                loader_placeholder.info(srch['search'])
                st.success(output)

                dd_summary_text += f"\nSearch Query: {srch['search']}\n" + output

    # ------------------------------------------------------------------------
    # Step 2: Summarize the final findings into a single Due Diligence Report
    # ------------------------------------------------------------------------
    messages = [
        SystemMessage(content=sys_prompt_due_diligence_report),
        HumanMessage(content=f"Collated Research Data:\n{dd_summary_text}")
    ]
    result = model.invoke(messages)
    data = parser.invoke(result)

    # Display the final due diligence report
    final_report_expander = st.expander("Final Due Diligence Report", expanded=True)
    present_due_diligence(final_report_expander, data)

else:
    # ------------------------------------------------------------------------
    # If no input yet, show default instructions
    # ------------------------------------------------------------------------
    default_instructions = """
# **Welcome to the Due Diligence Dashboard**

### **Step-by-Step Instructions**

1. **Select Company Type:**  
   - **Well-known:** e.g., Apple, Google, etc.  
   - **Greenfield Entrepreneurship:** If it's a newly formed startup with no public data.

2. **Enter Target Company Information in the Sidebar**  
   - **Target Company (optional if greenfield):** The name of the company you wish to evaluate (if well-known).  
   - **Industry (required):** The industry sector (e.g., Social Media, Tech, etc.).  
   - **Company Description (optional):** A brief overview of the target or new venture.  
   - **Financial Data (optional):** Relevant financial info, such as revenue, net income, etc.  
   - **Key Risks (optional):** Any known risk factors or concerns.  
   - **Legal/Regulatory Issues (optional):** Pending lawsuits or compliance challenges.  
   - **Valuation (optional):** Preliminary valuation insights or multiples.  
   - **Red Flags (optional):** Notable warnings, controversies, or negative press.  
   - **Additional Comments (optional):** Any other relevant details.

3. **Click “Conduct Due Diligence”:**  
   The system will research and generate a multi-category due diligence framework (financial, legal, market, operations, etc.).

4. **Review Categories & Final Report:**  
   - **Financial Overview**  
   - **Legal & Regulatory**  
   - **Market & Competition**  
   - **Operational Efficiency & Synergies**  
   - **Cultural Fit & HR**  
   - **Other Key Risks**

   Finally, a JSON-structured summary will provide key points, risks, opportunities, and a preliminary recommendation.

---

### **Get Started:**
Select the **Company Type**, fill in the relevant fields, and press **Conduct Due Diligence** to begin.
"""
    st.markdown(default_instructions)
