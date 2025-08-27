import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults

# Setting up necessary environment variables
parser = JsonOutputParser()
str_parser = StrOutputParser()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets['tapiKey']

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets['apiKey']

model = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [TavilySearchResults(max_results=3, include_answer=True, include_raw_content=True, search_depth="advanced")]

st.set_page_config(page_title="Nuggt Dashboard", layout="wide")

# ---- helper functions for shrinking on context-length error (ADDED) ----
def _is_ctx_len_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("context length" in s) or ("maximum context length" in s) or ("context_length_exceeded" in s)

def _shrink_to_first_third(text: str) -> str:
    if not text:
        return text
    return text[:max(1, len(text) // 3)]
# ------------------------------------------------------------------------

# Sidebar to take user input
with st.sidebar:
    corporate = st.text_input("Corporate (e.g. Amazon), Please put 'Greenfield Entrepreneurship' otherwise.", "Amazon")
    industry = st.text_input("Industry", "E-commerce")
    company_overview = st.text_area("Company Overview (What does the company do?)", "")
    competitors = st.text_area("Competitors (Who are the key competitors?)", "")
    geographic_focus = st.text_input("Geographic Focus (Primary regions of operation)", "")
    customer_base = st.text_area("Customer Base (e.g., B2B, B2C, demographics)", "")
    recent_news = st.text_area("Recent News/Developments (Any recent major news?)", "")
    key_challenges = st.text_area("Key Challenges (Current challenges in the market?)", "")
    business_model = st.text_area("Business Model (How does the company generate revenue?)", "")
    rough_idea = st.text_area("Rough Idea (optional)", "")
    additional_insights = st.text_area("Additional Insights", "")
    
    if st.button("Create Innovation Pitch", key="create_pitch_button"):
        # Store the input values into session state to use later
        st.session_state['user_input'] = {
            'corporate': corporate,
            'industry': industry,
            'company_overview': company_overview,
            'competitors': competitors,
            'geographic_focus': geographic_focus,
            'customer_base': customer_base,
            'recent_news': recent_news,
            'key_challenges': key_challenges,
            'business_model': business_model,
            'rough_idea': rough_idea,
            'additional_insights': additional_insights
        }


# Fetch the user input from session state
user_input = st.session_state.get('user_input', {})

# Update the system prompts dynamically based on user input
sys_prompt_search_agent = f"""
You are an expert of the market analysis framework. In this case, you are conducting
market analysis for the team at {user_input.get('corporate', 'a company')} who are trying to identify the 
next disruptive opportunity in the {user_input.get('industry', 'industry')}. 

{'The company overview is as follows: ' + user_input.get("company_overview") if user_input.get("company_overview") else ''}
{'They compete with the following competitors: ' + user_input.get("competitors") if user_input.get("competitors") else ''}
{'They operate primarily in: ' + user_input.get("geographic_focus") if user_input.get("geographic_focus") else ''}
{'Their customer base is: ' + user_input.get("customer_base") if user_input.get("customer_base") else ''}
{'Their business model includes: ' + user_input.get("business_model") if user_input.get("business_model") else ''}
{'They are currently facing these key challenges: ' + user_input.get("key_challenges") if user_input.get("key_challenges") else ''}
{'Recent developments or news about the company include: ' + user_input.get("recent_news") if user_input.get("recent_news") else ''}
{'The company is considering the following rough idea: ' + user_input.get("rough_idea") if user_input.get("rough_idea") else ''}
{'Additional insights include: ' + user_input.get("additional_insights") if user_input.get("additional_insights") else ''}

From your research, extract key points that will be useful
for the team to come up with new innovations. Present your findings and data in a concise manner
in less than 5 detailed points. For all your answers, you always provide the source in the form of a link.
"""

sys_prompt_market_analysis = f"""
You are a corporate innovation expert. Given a company your job is 
to propose corporate innovation projects that the company can consider. 

You identify innovation opportunities based on the Market Analysis Framework
which includes the following:

1. Consumer Behavior
2. Economic Conditions
3. Technological Advances
4. Competitive Landscape
5. Regulatory Environment

Given the company {user_input.get('corporate', 'a company')} in the {user_input.get('industry', 'industry')} industry,
{',with the following overview: ' + user_input.get("company_overview") if user_input.get("company_overview") else ''}
{',with the following competitors: ' + user_input.get("competitors") if user_input.get("competitors") else ''}
{',that operate primarily in: ' + user_input.get("geographic_focus") if user_input.get("geographic_focus") else ''}
{',with the following customer base: ' + user_input.get("customer_base") if user_input.get("customer_base") else ''}
{',with the following business model: ' + user_input.get("business_model") if user_input.get("business_model") else ''}
{',that is currently facing the following key challenges: ' + user_input.get("key_challenges") if user_input.get("key_challenges") else ''}
{',has the following recent developments or news: ' + user_input.get("recent_news") if user_input.get("recent_news") else ''}
{',and the following rough idea: ' + user_input.get("rough_idea") if user_input.get("rough_idea") else ''}
{',and considering additional insights: ' + user_input.get("additional_insights") if user_input.get("additional_insights") else ''}

Come up with google searches in the following JSON format:

{{
  "searches": {{
    "Consumer Behavior": [
      {{
        "search": "Google search 1",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 2",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 3",
        "importance": "Rationale behind this search"
      }},
    ],
    "Economic Conditions": [
      {{
        "search": "Google search 1",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 2",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 3",
        "importance": "Rationale behind this search"
      }},
    ],
    "Technological Advances": [
      {{
        "search": "Google search 1",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 2",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 3",
        "importance": "Rationale behind this search"
      }},
    ],
    "Competitive Landscape": [
      {{
        "search": "Google search 1",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 2",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 3",
        "importance": "Rationale behind this search"
      }},
    ],
    "Regulatory Environment": [
      {{
        "search": "Google search 1",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 2",
        "importance": "Rationale behind this search"
      }},
      {{
        "search": "Google search 3",
        "importance": "Rationale behind this search"
      }},
    ]
  }}
}}

Ensure that the search queries within the same category do not overlap. Queries within
the same category must be different from each other in order to properly cover all major
topics under that category.
"""

sys_prompt_depth_research = f"""
You are an expert of the market dynamics analysis framework. 
Your job is to conduct market dynamics analysis using Google. When you are
presented with information and data from general research about {user_input.get('corporate', 'a company')} in the {user_input.get('industry', 'industry')} industry,
you come up with 2 specific google search queries to gain more information and new data on crucial findings.
Your queries must be specific. We do not want general queries. Strictly specific queries
to facilitate in-depth research. Think critically when asking for the following queries. 
Come up with google searches in the following JSON format: 

{{
  "queries": [
    {{
      "query": "<your first query>"
    }},
    {{
      "query": "<your second query>"
    }},
  ]
}}

Your answers must strictly be in this JSON format.
"""

prompt = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input'], # Define the input variables your prompt depends on
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
            input_variables=[], # No input variables for the system message
            template=sys_prompt_search_agent # Use your new text blob here
            )
        ),
        #MessagesPlaceholder(variable_name='chat_history', optional=True), # Include chat history if available
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['input'], # Assuming 'input' is the variable for the human message
                template='{input}' # Use the 'input' variable as the template directly
            )
        ),
        MessagesPlaceholder(variable_name='agent_scratchpad'), # Include agent scratchpad messages if available
    ]
)

sys_prompt_key_insights = f"""
You are an expert in identifying disruptive innovation opportunities for 
{user_input.get('corporate', 'a company')} in the {user_input.get('industry', 'industry')} industry
based on market dynamics analysis reports. 

Here are some additional points to consider for the company:
{'The company overview is as follows: ' + user_input.get("company_overview") if user_input.get("company_overview") else ''}
{'They compete with the following competitors: ' + user_input.get("competitors") if user_input.get("competitors") else ''}
{'They operate primarily in: ' + user_input.get("geographic_focus") if user_input.get("geographic_focus") else ''}
{'Their customer base is: ' + user_input.get("customer_base") if user_input.get("customer_base") else ''}
{'Their business model includes: ' + user_input.get("business_model") if user_input.get("business_model") else ''}
{'They are currently facing these key challenges: ' + user_input.get("key_challenges") if user_input.get("key_challenges") else ''}
{'Recent developments or news about the company include: ' + user_input.get("recent_news") if user_input.get("recent_news") else ''}
{'The company is considering the following rough idea: ' + user_input.get("rough_idea") if user_input.get("rough_idea") else ''}
{'Additional insights include: ' + user_input.get("additional_insights") if user_input.get("additional_insights") else ''}

Given the market dynamics analysis report and the additional points, identify opportunities in the
following JSON format: 

{{
  "market_analysis": {{
    "consumer_behavior": {{
      "key_trends": [
        "Trend 1",
        "Trend 2"
      ],
      "gaps_in_market": [
        "Gap 1",
        "Gap 2"
      ],
      "unmet_needs": [
        "Unmet Need 1",
        "Unmet Need 2"
      ],
      "underserved_segments": [
        "Segment 1",
        "Segment 2"
      ],
      "industry_state": {{
        "before": "Describe how consumer behavior was shaped by key trends, unmet needs, underserved segments, and market gaps in the past. Include what consumers valued before current trends emerged.",
        "current": "Explain how consumer behavior is currently evolving, taking into account key trends, unmet needs, underserved segments, and market gaps. Highlight what consumers value today.",
        "future": "Forecast how consumer behavior will evolve when current key trends become dominant, addressing how unmet needs and underserved segments will shift in the future. This should be based on the identified trends, gaps, and needs.",
        "innovation": "Identify the innovation that the company should pursue to meet the future state of consumer behavior. Justify this innovation based on the current gaps, unmet needs, and trends in the market."
      }},
      "statistics": [
        "Extract and clearly state (in points) all statistics related to consumer behaviour in the market dynamics analysis report. Do not miss a single statistic",
      ]
    }},
    "economic_conditions": {{
      "key_trends": [
        "Trend 1",
        "Trend 2"
      ],
      "gaps_in_market": [
        "Gap 1",
        "Gap 2"
      ],
      "unmet_needs": [
        "Unmet Need 1",
        "Unmet Need 2"
      ],
      "underserved_segments": [
        "Segment 1",
        "Segment 2"
      ],
      "industry_state": {{
        "before": "Describe the economic conditions that influenced the market in the past, focusing on how key trends, unmet needs, underserved segments, and market gaps were addressed or not.",
        "current": "Explain the current economic conditions and their impact on the industry, emphasizing how they align with key trends, gaps, and unmet needs in the market.",
        "future": "Forecast future economic conditions and how they will shape the industry, highlighting the expected dominance of current trends and how unmet needs and underserved segments will evolve.",
        "innovation": "Identify the economic-focused innovation the company should adopt, based on the forecasted future conditions and unmet needs. Justify why this innovation aligns with future economic trends and addresses market gaps."
      }},
      "statistics": [
        "Extract and clearly state (in points) all statistics related to economic conditions in the market dynamics analysis report. Do not miss a single statistic",
      ]
    }},
    "technological_advances": {{
      "key_trends": [
        "Trend 1",
        "Trend 2"
      ],
      "gaps_in_market": [
        "Gap 1",
        "Gap 2"
      ],
      "unmet_needs": [
        "Unmet Need 1",
        "Unmet Need 2"
      ],
      "underserved_segments": [
        "Segment 1",
        "Segment 2"
      ],
      "industry_state": {{
        "before": "Describe how the industry operated before the current technological trends emerged, addressing how unmet needs, underserved segments, and gaps in the market were impacted by prior technology levels.",
        "current": "Explain how technological advancements are currently shaping the industry, emphasizing key trends, unmet needs, underserved segments, and gaps in the market.",
        "future": "Forecast how technological advancements will dominate the industry and address unmet needs, underserved segments, and gaps. Base this forecast on current key trends becoming the standard in the future.",
        "innovation": "Identify the technological innovation that the company should pursue, based on the forecasted future state of technology. Justify this innovation based on current gaps, unmet needs, and trends in technological advancements."
      }},
      "statistics": [
        "Extract and clearly state (in points) all statistics related to technological advancements in the market dynamics analysis report. Do not miss a single statistic",
      ]
    }},
    "competitive_landscape": {{
      "key_trends": [
        "Trend 1",
        "Trend 2"
      ],
      "gaps_in_market": [
        "Gap 1",
        "Gap 2"
      ],
      "unmet_needs": [
        "Unmet Need 1",
        "Unmet Need 2"
      ],
      "underserved_segments": [
        "Segment 1",
        "Segment 2"
      ],
      "industry_state": {{
        "before": "Describe how the competitive landscape looked before the current trends took hold, particularly in relation to unmet needs, underserved segments, and gaps in the market.",
        "current": "Explain how the competitive landscape is evolving, focusing on key trends, unmet needs, underserved segments, and gaps in the market today.",
        "future": "Forecast how the competitive landscape will shift in the future, with key trends becoming dominant and how that will impact unmet needs, underserved segments, and gaps.",
        "innovation": "Identify the innovation that the company should focus on to gain a competitive advantage, based on the forecasted future landscape. Justify how this innovation addresses current market gaps and unmet needs."
      }},
      "statistics": [
        "Extract and clearly state (in points) all statistics related to competitive landscape in the market dynamics analysis report. Do not miss a single statistic",
      ]
    }},
    "regulatory_environment": {{
      "key_trends": [
        "Trend 1",
        "Trend 2"
      ],
      "gaps_in_market": [
        "Gap 1",
        "Gap 2"
      ],
      "unmet_needs": [
        "Unmet Need 1",
        "Unmet Need 2"
      ],
      "underserved_segments": [
        "Segment 1",
        "Segment 2"
      ],
      "industry_state": {{
        "before": "Describe how regulations shaped the industry before the current trends, and how unmet needs, underserved segments, and market gaps were addressed by previous regulatory conditions.",
        "current": "Explain the current regulatory environment, emphasizing how it is responding to key trends, unmet needs, underserved segments, and gaps in the market.",
        "future": "Forecast how regulations will change in the future and how they will impact key trends, unmet needs, underserved segments, and market gaps.",
        "innovation": "Identify the regulatory-focused innovation the company should pursue, based on future regulatory changes. Justify why this innovation will align with future regulations and address current gaps in the market."
      }},
      "statistics": [
        "Extract and clearly state (in points) all statistics related to regulatory environment in the market dynamics analysis report. Do not miss a single statistic",
      ]
    }}
  }}
}}

Strictly stick to this JSON format and do not reply anything else. 
"""

def pitch_idea(idea_expander, data):
  with idea_expander:
      # Story - Past Character
      st.subheader("1. Story-Based Characters")
      st.markdown("### Past Character")
      st.info(f"**Description:** {data['big_picture_innovation']['story']['past_character']['description']}")
      st.markdown(f"**Use Case:** {data['big_picture_innovation']['story']['past_character']['use_case']}")
      st.markdown(f"**Story:** {data['big_picture_innovation']['story']['past_character']['simple_story']}")
      
      # Story - Current Character
      st.markdown("### Current Character")
      st.info(f"**Description:** {data['big_picture_innovation']['story']['current_character']['description']}")
      st.markdown(f"**Use Case:** {data['big_picture_innovation']['story']['current_character']['use_case']}")
      st.markdown(f"**Story:** {data['big_picture_innovation']['story']['current_character']['simple_story']}")
      
      # Story - Future Character
      st.markdown("### Future Character")
      st.info(f"**Description:** {data['big_picture_innovation']['story']['future_character']['description']}")
      st.markdown(f"**Use Case:** {data['big_picture_innovation']['story']['future_character']['use_case']}")
      st.markdown(f"**Story:** {data['big_picture_innovation']['story']['future_character']['simple_story']}")
      
      # Future Product
      st.markdown("### Future Product")
      st.success(f"**Description:** {data['big_picture_innovation']['story']['future_product']['description']}")
      
      # Innovation Plan
      st.header("Innovation Plan")

      # Existing Solution
      st.subheader("2. Existing Solution")
      st.info(f"**Current Solution:** {data['innovation_plan']['existing_solution']['description']}")
      
      # Most Crucial Change
      st.subheader("3. Most Crucial Change")
      st.warning(f"**One Key Change:** {data['innovation_plan']['most_crucial_change']['description']}")
      
      # Justification
      st.markdown(f"**Justification:** {data['innovation_plan']['justification']['description']}")
      
      # MVP, POC, GTM
      st.header("MVP, POC, and Go-To-Market Strategy")

      # Leap of Faith Assumption
      st.subheader("4. Leap of Faith Assumption")
      st.info(f"**Assumption:** {data['mvp_poc_gtm']['leap_of_faith_assumption']['description']}")
      
      # MVP Section
      st.subheader("5. Minimum Viable Product (MVP)")
      st.success(f"**MVP Description:** {data['mvp_poc_gtm']['mvp']['description']}")
      st.markdown("### Key Features")
      for feature in data['mvp_poc_gtm']['mvp']['key_features']:
          st.markdown(f"- {feature}")
      st.markdown(f"**Success Criteria:** {data['mvp_poc_gtm']['mvp']['success_criteria']}")
      
      # POC Section
      st.subheader("6. Proof of Concept (POC)")
      st.success(f"**POC Description:** {data['mvp_poc_gtm']['poc']['description']}")
      st.markdown("### Validation Goals")
      for goal in data['mvp_poc_gtm']['poc']['validation_goals']:
          st.markdown(f"- {goal}")
      st.markdown("### Potential Obstacles")
      for obstacle in data['mvp_poc_gtm']['poc']['potential_obstacles']:
          st.markdown(f"- {obstacle}")
      
      # Beachhead Market
      st.subheader("7. Beachhead Market")
      st.info(f"**Description:** {data['mvp_poc_gtm']['beachhead_market']['description']}")
      st.markdown("### Market Characteristics")
      for characteristic in data['mvp_poc_gtm']['beachhead_market']['market_characteristics']:
          st.markdown(f"- {characteristic}")
      st.markdown("### Key Assumptions")
      for assumption in data['mvp_poc_gtm']['beachhead_market']['key_assumptions']:
          st.markdown(f"- {assumption}")
      
      # Go-To-Market Strategy
      st.subheader("8. Go-To-Market (GTM) Strategy")
      st.success(f"**GTM Strategy:** {data['mvp_poc_gtm']['gtm_strategy']['strategy']}")
      st.markdown("### Tests and Targets")
      for test in data['mvp_poc_gtm']['gtm_strategy']['tests_and_targets']['tests']:
          st.markdown(f"- Test: {test}")
      for target in data['mvp_poc_gtm']['gtm_strategy']['tests_and_targets']['targets']:
          st.markdown(f"- Target: {target}")
      st.markdown(f"**Cost:** {data['mvp_poc_gtm']['gtm_strategy']['cost']}")
      st.markdown(f"**Timeline:** {data['mvp_poc_gtm']['gtm_strategy']['timeline']}")
      
      # Financial Projections
      st.header("Financial Projections")
      st.markdown("### Projection Methodology")
      st.success(f"**Methodology:** {data['financial_projections']['projection_methodology']}")
      
      st.markdown("### Revenue Growth Projections")
      st.markdown(f"**Year 1:** {data['financial_projections']['estimated_revenue_growth']['year_1']['revenue']} "
                  f"({data['financial_projections']['estimated_revenue_growth']['year_1']['reasoning']})")
      st.markdown(f"**Year 2:** {data['financial_projections']['estimated_revenue_growth']['year_2']['revenue']} "
                  f"({data['financial_projections']['estimated_revenue_growth']['year_2']['reasoning']})")
      st.markdown(f"**Year 3:** {data['financial_projections']['estimated_revenue_growth']['year_3']['revenue']} "
                  f"({data['financial_projections']['estimated_revenue_growth']['year_3']['reasoning']})")
      
      st.markdown("### Profit Margins")
      st.markdown(f"**Year 1:** {data['financial_projections']['profit_margins']['year_1']}")
      st.markdown(f"**Year 2:** {data['financial_projections']['profit_margins']['year_2']}")
      st.markdown(f"**Year 3:** {data['financial_projections']['profit_margins']['year_3']}")
      
      st.markdown("### Break-Even Analysis")
      st.markdown(f"**Break-Even Timeline:** {data['financial_projections']['break_even_analysis']['break_even_timeline']}")
      st.markdown(f"**Conditions for Break-Even:** {data['financial_projections']['break_even_analysis']['conditions']}")

if 'user_input' in st.session_state:
    # Define the interaction for invoking the model with updated prompts
    messages = [
        SystemMessage(content=sys_prompt_market_analysis),
        HumanMessage(content=user_input.get('corporate', 'Give me search queries')),
    ]

    result = model.invoke(messages)
    answer = parser.invoke(result)

    # Create the agent executor
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    market_dynamics = ""
    
    expand_it = True
    expanders = {}
    for category in ["Consumer Behavior", "Economic Conditions", "Technological Advances", "Competitive Landscape", "Regulatory Environment"]:
        # Create an expander for each category and store in a dictionary
        expanders[category] = st.expander(f"Category: {category}", expanded=expand_it)
        expand_it = False

    # First loop to show "Collating insights" inside the expander for each category
    for category in ["Consumer Behavior", "Economic Conditions", "Technological Advances", "Competitive Landscape", "Regulatory Environment"]:
        with expanders[category]:
            st.warning(f"Collating insights on '{category}'")
    
    report_expander = st.expander(f"Identified Areas of Innovation", expanded=False)
    with report_expander:
      st.warning(f"Key Insights Report")
    
    idea_simple = st.expander(f"Proposed Innovation By GPT-4o-mini [Simple Model]", expanded=False)
    with idea_simple:
      st.warning(f"Innovation Pitch Deck (GPT-4o-mini)")

    idea_standard = st.expander(f"Proposed Innovation By GPT-4o [Standard Model]", expanded=False)
    with idea_standard:
      st.warning(f"Innovation Pitch Deck (GPT-4o)")
        
    idea_expander = st.expander(f"Proposed Innovation By gpt-4.1 [Powerful Model]", expanded=False)
    with idea_expander:
      st.warning(f"Innovation Pitch Deck (gpt-4.1)")
        
    # Second loop to populate the expanders with the content
    for category, searches in answer['searches'].items():
        market_dynamics += f"\n{category}"
        output_list = ""
        with expanders[category]:
            for search in searches:
                # Create a placeholder for the loader
                loader_placeholder = st.empty()

                # Simulate a loading message with a spinner
                loader_placeholder.info(f"Researching on: {search['search']} ⏳")

                # Invoke the agent to get the output for the search
                try:
                    output = agent_executor.invoke({"input": search['search']})["output"]
                except Exception as e:
                    if _is_ctx_len_error(e):
                        # On context error: retry with smaller tool AND 1/3rd the final output string
                        tools_small = [TavilySearchResults(
                            max_results=3,
                            include_answer=True,
                            include_raw_content=False,
                            search_depth="advanced"
                        )]
                        agent_small = create_openai_tools_agent(model, tools_small, prompt)
                        agent_executor_small = AgentExecutor(agent=agent_small, tools=tools_small, verbose=False)
                        output = agent_executor_small.invoke({"input": search['search']})["output"]
                        output = _shrink_to_first_third(output)
                    else:
                        raise

                # Replace the loader with a success message (tick emoji)
                loader_placeholder.info(search['search'])

                st.success(output)

                output_list = output_list + "\n" + output

                # Update market_dynamics with the output
                market_dynamics += f"\n{search['search']}\n{output}"
                
    
    messages = [
        SystemMessage(content=sys_prompt_key_insights),
        HumanMessage(content=f"Marketing Dynamics Analysis:\n{market_dynamics}"),
    ]

    # --- SHRINK-ON-ERROR WRAP (ADDED) for key insights step ---
    _md_tmp = market_dynamics
    while True:
        try:
            messages = [
                SystemMessage(content=sys_prompt_key_insights),
                HumanMessage(content=f"Marketing Dynamics Analysis:\n{_md_tmp}"),
            ]
            result = model.invoke(messages)
            break
        except Exception as e:
            if _is_ctx_len_error(e):
                _md_tmp = _shrink_to_first_third(_md_tmp)
                continue
            else:
                raise
    # ----------------------------------------------------------

    data = parser.invoke(result)

    market_dynamics = str(data)

    for key, value in data["market_analysis"].items():
      # Create an expander for each section
      with report_expander:
          # Key Trends, Gaps, Unmet Needs, Underserved Segments side by side in columns
          col1, col2, col3, col4 = st.columns(4)
          
          st.warning(f'## {key.replace("_", " ")} insights')
          # Key Trends
          with col1:
            st.info("### Key Trends")
            for trend in value["key_trends"]:
                st.write(f"- {trend}")

          # Gaps in Market
          with col2:
            st.warning("### Gaps in Market")
            for gap in value["gaps_in_market"]:
                st.write(f"- {gap}")

          # Unmet Needs
          with col3:
            st.success("### Unmet Needs")
            for need in value["unmet_needs"]:
                st.write(f"- {need}")

          # Underserved Segments
          with col4:
            st.info("### Underserved Segments")
            for segment in value["underserved_segments"]:
                st.write(f"- {segment}")

          # Industry State: Before, Current, Future, Innovation (spanning all columns)
          st.write("### Industry State")
          st.warning(f"**Before:** {value['industry_state']['before']}")
          st.info(f"**Current:** {value['industry_state']['current']}")
          st.success(f"**Future:** {value['industry_state']['future']}")
          st.success(f"**Innovation:** {value['industry_state']['innovation']}")

          st.write("### Key Statistics")
          for point in value["statistics"]:
            st.write(f"- {point}")
          
          st.markdown("---")
    
    sys_final_innovation = f"""
    You are an expert in coming up with practical MVP solutions to test leap of
    faith assumptions extracted from the market dynamics analysis report for {user_input.get('corporate', 'a company')} 
    in the {user_input.get('industry', 'industry')} industry. Use the provided statistics to make your points. Your answer must have
    statistics to back your assumptions. 
    
    Here are some additional things to consider:
    {'The company overview is as follows: ' + user_input.get("company_overview") if user_input.get("company_overview") else ''}
    {'They compete with the following competitors: ' + user_input.get("competitors") if user_input.get("competitors") else ''}
    {'They operate primarily in: ' + user_input.get("geographic_focus") if user_input.get("geographic_focus") else ''}
    {'Their customer base is: ' + user_input.get("customer_base") if user_input.get("customer_base") else ''}
    {'Their business model includes: ' + user_input.get("business_model") if user_input.get("business_model") else ''}
    {'They are currently facing these key challenges: ' + user_input.get("key_challenges") if user_input.get("key_challenges") else ''}
    {'Recent developments or news about the company include: ' + user_input.get("recent_news") if user_input.get("recent_news") else ''}
    {'The company is considering the following rough idea: ' + user_input.get("rough_idea") if user_input.get("rough_idea") else ''}
    {'Additional insights include: ' + user_input.get("additional_insights") if user_input.get("additional_insights") else ''}

    
    From a given market analysis insight and the additional points, you design the validation experiment in the following
    JSON format:

    {{
      "Overview": "Clear description of the innovation and the idea.",
      "big_picture_innovation": {{
        "story": {{
          "past_character": {{
            "description": "Describe the character from the past, how they used the product, and how it aligned with the trends, unmet needs, and gaps in the market during that time.",
            "use_case": "How this character used the product, focusing on past market trends and gaps in the market.",
            "simple_story": "Based on the description and usecase give this character a name and persona and write a simple 4-5 line paragraph describing how he or she used the product."
          }},
          "current_character": {{
            "description": "Describe the current user, what key trends, unmet needs, and gaps are shaping their usage today.",
            "use_case": "How the current user is using the product to meet their present needs and how it serves them based on today’s market dynamics.",
            "simple_story": "Based on the description and usecase give this character a name and persona and write a simple 4-5 line paragraph describing how he or she using the product."
          }},
          "future_character": {{
            "description": "Describe the future character, their needs, the future key trends, and gaps in the market that will affect their usage.",
            "use_case": "How this character in the future will use the ideal product to meet their future needs based on forecasted trends and market gaps.",
            "simple_story": "Based on the description and usecase give this character a name and persona and write a simple 4-5 line paragraph describing how he or she will use the product."
          }},
          "future_product": {{
            "description": "Define the ideal product that can meet the demands of the future character. How this product solves future unmet needs, addresses future gaps, and capitalizes on future market trends."
          }}
        }}
      }},
      "innovation_plan": {{
        "existing_solution": {{
          "description": "Define the existing solution as it is today, explaining how it meets the current market needs."
        }},
        "most_crucial_change": {{
          "description": "State the one most crucial change that needs to be made to the current product. This change should address an unmet need of today’s users and be a step toward the future ideal product described earlier. This change must be technically feasible and testable."
        }},
        "justification": {{
          "description": "Explain why this change is the most crucial, how it serves current users better, and how it is a step toward the future product based on trends, gaps, and unmet needs."
        }}
      }},
      "mvp_poc_gtm": {{
        "leap_of_faith_assumption": {{
          "description": "Define the leap of faith assumption based on the one crucial change. This should be a core hypothesis that, if proven, will validate that the chosen change will move the product toward the ideal future state."
        }},
        "mvp": {{
          "description": "Describe the Minimum Viable Product (MVP) designed to test the leap of faith assumption. This should offer only the core value proposition of the crucial change with minimal features. Give a practical MVP that can be easily build by clearly laying out the core features",
          "key_features": [
            "Core feature 1",
            "Core feature 2"
          ],
          "success_criteria": "Metrics or benchmarks that will determine if the MVP is successful (e.g., X number of users, Y amount of revenue)."
        }},
        "poc": {{
          "description": "Description of the Proof of Concept (POC) to validate the feasibility of the MVP. The POC should aim to prove that the innovation can work from a technical and business standpoint.",
          "validation_goals": [
            "Technical feasibility goal",
            "Business validation goal"
          ],
          "potential_obstacles": [
            "Technical challenge 1",
            "Market adoption challenge 2"
          ]
        }},
        "beachhead_market": {{
          "description": "Identification of the beachhead market for initial testing. This market should be small enough to control costs but large enough to provide meaningful insights for scaling.",
          "market_characteristics": [
            "Characteristic 1 (e.g., specific demographic or geographic area)",
            "Characteristic 2 (e.g., industry vertical or early adopters)"
          ],
          "key_assumptions": [
            "Assumption 1 about the market",
            "Assumption 2 about customer behavior"
          ]
        }},
        "gtm_strategy": {{
          "strategy": "High-level Go-To-Market (GTM) strategy for the beachhead market, including marketing, sales, and distribution plans.",
          "tests_and_targets": {{
            "tests": [
              "Test 1: Hypothesis that will be validated (e.g., customer willingness to pay)",
              "Test 2: Hypothesis related to product performance"
            ],
            "targets": [
              "Target 1: Key metric (e.g., number of beta users)",
              "Target 2: Key result (e.g., minimum viable revenue)"
            ]
          }},
          "cost": "Estimated cost for the validation and testing process (e.g., $X for product development, $Y for marketing).",
          "timeline": "Estimated timeline for the validation process, including milestones for MVP launch, POC, and GTM phases."
        }}
      }},
      "financial_projections": {{
        "projection_methodology": "Define the financial projection methodology used, based on lean startup assumptions, market potential, and estimated cost structures.",
        "estimated_revenue_growth": {{
          "year_1": {{
            "revenue": "$X",
            "reasoning": "Explanation for how Year 1 revenue is estimated based on market size and MVP success."
          }},
          "year_2": {{
            "revenue": "$Y",
            "reasoning": "Reasoning behind Year 2 revenue growth based on market expansion, product adoption, and scaling efforts."
          }},
          "year_3": {{
            "revenue": "$Z",
            "reasoning": "Projections for Year 3 based on product improvements, market penetration, and customer retention."
          }}
        }},
        "profit_margins": {{
          "year_1": "Estimated profit margin in Year 1, with a justification based on early-stage product costs.",
          "year_2": "Expected profit margin for Year 2, considering market adoption and scaling.",
          "year_3": "Profit margin for Year 3, as the product reaches more maturity and efficiency in cost structure."
        }},
        "break_even_analysis": {{
          "break_even_timeline": "Estimated timeline for reaching the break-even point.",
          "conditions": "Conditions necessary for the break-even point to be achieved (e.g., revenue target, cost reductions, market growth)."
        }}
      }}
    }}

    Strictly stick to this JSON format and do not reply anything else. 
    """
    messages = [
        SystemMessage(content=sys_final_innovation),
        HumanMessage(content=f"Marketing Dynamics Analysis:\n{market_dynamics}"),
    ]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- SHRINK-ON-ERROR WRAP (ADDED) for first final-innovation step ---
    _md_tmp = market_dynamics
    while True:
        try:
            messages = [
                SystemMessage(content=sys_final_innovation),
                HumanMessage(content=f"Marketing Dynamics Analysis:\n{_md_tmp}"),
            ]
            result = model.invoke(messages)
            break
        except Exception as e:
            if _is_ctx_len_error(e):
                _md_tmp = _shrink_to_first_third(_md_tmp)
                continue
            else:
                raise
    # --------------------------------------------------------------------

    data = parser.invoke(result)
    idea_1 = data["Overview"]
    pitch_idea(idea_simple, data)

    messages = [
        SystemMessage(content=sys_final_innovation),
        HumanMessage(content=f"Marketing Dynamics Analysis:\n{market_dynamics}\n\nCome up with an idea that is different from {idea_1}"),
    ]
    model = ChatOpenAI(model="gpt-4o", temperature=1)

    # --- SHRINK-ON-ERROR WRAP (ADDED) for second final-innovation step ---
    _md_tmp = market_dynamics
    while True:
        try:
            messages = [
                SystemMessage(content=sys_final_innovation),
                HumanMessage(content=f"Marketing Dynamics Analysis:\n{_md_tmp}\n\nCome up with an idea that is different from {idea_1}"),
            ]
            result = model.invoke(messages)
            break
        except Exception as e:
            if _is_ctx_len_error(e):
                _md_tmp = _shrink_to_first_third(_md_tmp)
                continue
            else:
                raise
    # ---------------------------------------------------------------------

    data = parser.invoke(result)
    idea_2 = data["Overview"]
    pitch_idea(idea_standard, data)
    
    messages = [
        #SystemMessage(content=sys_final_innovation),
        HumanMessage(content=f"{sys_final_innovation}\n\nMarketing Dynamics Analysis:\n{market_dynamics}\n\nRemember to make use of statistics to make your case.\n\nCome up with an idea that is different from {idea_1} and different from {idea_2}"),
    ]
    model = ChatOpenAI(model="gpt-4.1", temperature=1)

    # --- SHRINK-ON-ERROR WRAP (ADDED) for o1-preview step ---
    _md_tmp = market_dynamics
    while True:
        try:
            messages = [
                #SystemMessage(content=sys_final_innovation),
                HumanMessage(content=f"{sys_final_innovation}\n\nMarketing Dynamics Analysis:\n{_md_tmp}\n\nRemember to make use of statistics to make your case.\n\nCome up with an idea that is different from {idea_1} and different from {idea_2}"),
            ]
            result = model.invoke(messages)
            break
        except Exception as e:
            if _is_ctx_len_error(e):
                _md_tmp = _shrink_to_first_third(_md_tmp)
                continue
            else:
                raise
    # --------------------------------------------------------

    data = parser.invoke(result)
    pitch_idea(idea_expander, data)
    
else:
  default_instructions = """
# **Welcome to the Market Dynamics Report Generator to Validate Startup Ideas and Identify Opportunities**

### **Step-by-Step Instructions**

### **1. Provide Input in the Sidebar:**

- **Company (required):** Enter the name of the company you’re working with.  
  **Example:** `Amazon`

- **Industry (required):** Specify the industry you want to innovate in.  
  **Example:** `E-commerce, Cloud Services`

- **Company Overview (optional):** Briefly describe the company.  
  **Example:** `Online retail giant offering a wide range of products and services.`

- **Competitors (optional):** List key competitors. Multiple competitors can be entered, separated by commas.  
  **Example:** `Walmart, eBay, Alibaba`

- **Geographic Focus (optional):** Specify the regions or countries where the company operates.  
  **Example:** `Global, with focus in North America and Europe`

- **Customer Base (optional):** Describe the company's customer base.  
  **Example:** `B2C, Millennial shoppers, high-income urban professionals`

- **Recent News (optional):** Mention any recent developments or news.  
  **Example:** `Launched new AI-powered shopping tools, Acquired Zoox`

- **Key Challenges (optional):** Describe current challenges the company faces.  
  **Example:** `Supply chain disruptions, Increased competition in delivery services`

- **Business Model (optional):** Explain how the company generates revenue.  
  **Example:** `Marketplace commissions, Subscription services (Amazon Prime), Advertising`

- **Rough Idea (optional):** Share any rough innovation idea you may have in mind.  
  **Example:** `Expanding into grocery delivery using drones`

- **Additional Insights (optional):** Provide any other useful insights or information.

  > **Note:** You can input multiple values separated by commas for all fields other than the Company name.

---

### **2. Click ‘Create Innovation Pitch’:**

After entering the required and optional fields, click the button to generate your customized pitch deck.

---

### **3. Categories of Research:**

After clicking the button, the system will research and gather insights based on five key categories:
- **Consumer Behavior**
- **Economic Conditions**
- **Technological Advances**
- **Competitive Landscape**
- **Regulatory Environment**

---

### **4. Market Insight Report:**

The tool will generate a **Market Insight Report** covering:
- **Key Trends:** Major trends shaping the market.
- **Gaps in the Market:** Unfulfilled needs or inefficiencies.
- **Unmet Needs:** Specific needs that are not currently being addressed.
- **Underserved Segments:** Market segments that are not receiving adequate attention.
- **Industry State:** A historical, current, and future analysis of the industry, including suggested innovations for future challenges.

---

### **5. Innovation Pitch:**

The final section will generate an **Innovation Pitch**, which includes:
- **Past, Current, and Future Character Stories:** Illustrates how users from the past, present, and future interact with the product or service, showing how market trends and gaps impact them.
- **Ideal Future Product:** A description of the future product that meets the needs of future users based on identified trends and gaps.
- **Innovation Plan:** Outlines the current solution and identifies the most crucial change required to meet market demands.
- **MVP, POC, and Go-To-Market Strategy:** A detailed plan to create a Minimum Viable Product (MVP) for testing the core innovation, followed by a Proof of Concept (POC) and initial Go-To-Market strategy. This includes:
  - **Leap of Faith Assumption:** A core assumption that will be tested through the MVP.
  - **Market Characteristics:** Target market (beachhead market) for testing.
  - **Tests, Targets, and Timeline:** Specific metrics for validating the innovation, cost estimates, and timelines.
- **Financial Projections:** Financial forecasts including revenue growth, profit margins, and break-even analysis.

---

### **6. Pitch Deck Models:**

You will receive **three pitch decks**:
- **gpt-4.1 (most powerful model):** Provides in-depth insights.
- **gpt-4o (standard model):** Offers detailed insights with practical considerations.
- **gpt-4o-mini (simpler model):** Generates concise reports for quick insights.

---

Use this tool to explore innovation opportunities, create MVPs, and propose data-backed strategies aligned with your business goals and the future industry trends.
"""

  st.markdown(default_instructions)
