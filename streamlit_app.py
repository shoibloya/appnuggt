"""The nuggt workspace: choose the right tool for your next question."""

from html import escape

import streamlit as st


st.set_page_config(page_title="nuggt · Innovation workspace", page_icon="🧠", layout="wide")

TOOLS = [
    {
        "name": "Ideation", "path": "pages/Ideation.py", "number": "02",
        "color": "#c58a49", "purpose": "Discover the next opportunity",
        "description": "Explore market shifts, customer needs and emerging technologies to shape an innovation pitch.",
        "detail": "Market dynamics · Opportunity discovery",
    },
    {
        "name": "Due Diligence", "path": "pages/Due Diligence.py", "number": "03",
        "color": "#7193c7", "purpose": "See the bigger picture",
        "description": "Investigate a company’s financial position, market, operations and risks before your next decision.",
        "detail": "Company research · Risk assessment",
    },
    {
        "name": "Aether Workshop", "path": "pages/Aether.py", "number": "04",
        "color": "#56a69c", "purpose": "Choose where to build",
        "description": "Compare Singapore, Jakarta and hybrid operating setups through a focused location strategy workshop.",
        "detail": "Location strategy · Operating setup",
    },
    {
        "name": "BUTA", "path": "pages/BUTA.py", "number": "05",
        "color": "#a08acc", "purpose": "Find your first customers",
        "description": "Compare beachhead segments through Budget, Urgency, Top-3 Fit and Access.",
        "detail": "Customer segments · Beachhead selection",
    },
]

st.markdown(
    """
<style>
.block-container { max-width: 1180px; padding-top: 2.3rem; padding-bottom: 3rem; }
/* The homepage uses one ordered menu of native page links. */
[data-testid="stSidebarNav"] { display: none; }
.ng-brand { display: flex; align-items: center; gap: .7rem; margin: .3rem 0 2rem;
  font-size: 1.65rem; font-weight: 700; letter-spacing: -.06em; }
.ng-brand-mark { display: grid; place-items: center; width: 33px; height: 33px;
  border: 1px solid rgba(154,132,204,.45); border-radius: 10px;
  background: rgba(154,132,204,.12); font-size: 1.3rem; letter-spacing: -.04em; }
.ng-sidebar-label { font-size: .67rem; letter-spacing: .13em; font-weight: 650;
  opacity: .5; margin: 1.5rem 0 .6rem; text-transform: uppercase; }
.ng-sidebar-note { font-size: .8rem; line-height: 1.65; opacity: .6; margin-top: 2rem;
  padding-top: 1.1rem; border-top: 1px solid rgba(128,128,128,.18); }
.ng-hero { margin: .2rem 0 1.9rem; }
.ng-eyebrow { display: flex; align-items: center; gap: .55rem; font-size: .7rem;
  letter-spacing: .15em; text-transform: uppercase; font-weight: 650; opacity: .65; }
.ng-eyebrow-dot { width: 7px; height: 7px; background: #a08acc; border-radius: 50%; }
.ng-hero h1 { font-size: clamp(2.4rem, 4.3vw, 3.7rem); line-height: 1.08;
  letter-spacing: -.045em; font-weight: 700; margin: .85rem 0 1rem; padding: 0; }
.ng-hero p { font-size: 1.02rem; line-height: 1.65; opacity: .68;
  max-width: 610px; margin: 0; }
.ng-featured { display: grid; grid-template-columns: minmax(0,1fr) 190px; gap: 1.5rem;
  align-items: center; padding: .65rem .6rem .25rem; }
.ng-featured-meta { display: flex; align-items: center; gap: .65rem; margin-bottom: .65rem; }
.ng-section-label { font-size: .67rem; letter-spacing: .13em; text-transform: uppercase;
  font-weight: 650; opacity: .6; }
.ng-new { display: inline-flex; border-radius: 99px; padding: .22rem .55rem;
  color: inherit; background: rgba(154,132,204,.19); border: 1px solid rgba(154,132,204,.25);
  font-size: .65rem; font-weight: 750; letter-spacing: .05em; }
.ng-featured h2 { font-size: 1.9rem; font-weight: 650; letter-spacing: -.035em;
  line-height: 1.2; padding: 0; margin: 0 0 .7rem; }
.ng-featured p { max-width: 540px; font-size: .97rem; line-height: 1.65;
  opacity: .75; margin: 0; }
.ng-lens-tags { display: flex; flex-wrap: wrap; gap: .55rem 1rem; margin: 1rem 0 .25rem;
  font-size: .75rem; opacity: .8; }
.ng-lens-tags span { display: flex; align-items: center; gap: .4rem; }
.ng-lens-tags i { display: inline-block; height: 5px; width: 5px; border-radius: 50%; }
.ng-venn { position: relative; width: 180px; height: 172px; margin: 0 auto; }
.ng-venn span { position: absolute; width: 113px; height: 113px; border-radius: 50%; border: 1px solid; }
.ng-venn-d { top: 0; left: 0; background: rgba(197,138,73,.07); border-color: rgba(197,138,73,.45) !important; }
.ng-venn-f { top: 0; right: 0; background: rgba(86,166,156,.07); border-color: rgba(86,166,156,.45) !important; }
.ng-venn-v { bottom: 0; left: 34px; background: rgba(160,138,204,.09); border-color: rgba(160,138,204,.5) !important; }
.ng-venn b { position: absolute; width: 64px; left: 58px; top: 66px; text-align: center;
  font-size: .69rem; line-height: 1.4; font-weight: 600; opacity: .8; }
.block-container p.ng-featured-note { margin: .7rem .6rem 0; font-size: .75rem; opacity: .6; }
.ng-grid-heading { display: flex; justify-content: space-between; align-items: baseline;
  flex-wrap: wrap; gap: .5rem; margin: 2rem 0 .95rem; }
.ng-grid-heading h2 { font-size: 1.3rem; letter-spacing: -.025em; padding: 0; margin: 0; font-weight: 650; }
.ng-grid-heading p { font-size: .8rem; margin: 0; opacity: .55; }
.ng-tool { padding: .45rem .35rem .15rem; min-height: 210px; }
.ng-tool-meta { display: flex; align-items: center; gap: .5rem; margin-bottom: 1rem; }
.ng-tool-dot { width: 7px; height: 7px; border-radius: 50%; }
.ng-tool-number { font-size: .68rem; font-weight: 600; opacity: .5; }
.ng-tool h3 { font-size: 1.23rem; letter-spacing: -.025em; font-weight: 650; margin: 0 0 .3rem; padding: 0; }
.ng-tool-purpose { font-size: .8rem; font-weight: 550; opacity: .8; margin-bottom: .7rem; }
.ng-tool-description { font-size: .85rem; line-height: 1.65; opacity: .66; margin: 0 0 .8rem; }
.ng-tool p.ng-tool-detail { font-size: .73rem; line-height: 1.5; opacity: .6; margin: 0; }
.ng-footer { font-size: .78rem; line-height: 1.65; opacity: .55; margin-top: 1.6rem;
  padding-top: 1rem; border-top: 1px solid rgba(128,128,128,.16); }
/* Native Streamlit page links retain client-side navigation and session state. */
.block-container [data-testid="stPageLink-NavLink"] {
  border: 1px solid rgba(128,128,128,.25); border-radius: 8px;
  padding: .6rem .85rem; margin-top: .4rem; width: fit-content;
  font-size: .83rem; font-weight: 600; }
.block-container [data-testid="stPageLink-NavLink"][href$="/Design_Thinking"] {
  background: rgba(154,132,204,.17); border-color: rgba(154,132,204,.4); }
.block-container [data-testid="stPageLink-NavLink"]:focus-visible {
  outline: 2px solid #a08acc; outline-offset: 3px; }
@media (hover: hover) and (pointer: fine) {
  .block-container [data-testid="stPageLink-NavLink"]:hover {
    background: rgba(154,132,204,.12); border-color: rgba(154,132,204,.6); }
}
@media (max-width: 760px) {
  .ng-featured { grid-template-columns: minmax(0,1fr); padding: .4rem .1rem .15rem; }
  .ng-venn { display: none; }
  .ng-tool { min-height: 0; }
  .ng-hero { margin-bottom: 1.5rem; }
  .ng-grid-heading { margin-top: 1.5rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="ng-brand"><span class="ng-brand-mark">n</span>nuggt</div>', unsafe_allow_html=True)
    st.page_link("streamlit_app.py", label="Workspace")
    st.markdown('<div class="ng-sidebar-label">Your tools</div>', unsafe_allow_html=True)
    st.page_link("pages/Design Thinking.py", label="Design Thinking · New")
    for tool in TOOLS:
        st.page_link(tool["path"], label=tool["name"])
    st.markdown(
        '<div class="ng-sidebar-note">A broad idea is enough to start.<br>Explore it with Design Thinking, then choose a focused tool for your next question.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="ng-hero">
  <div class="ng-eyebrow"><span class="ng-eyebrow-dot"></span>nuggt · Innovation workspace</div>
  <h1>Turn ideas into<br>clearer decisions.</h1>
  <p>A workspace for exploring opportunities, testing assumptions, and deciding what comes next.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        """
<div class="ng-featured">
  <div>
    <div class="ng-featured-meta"><span class="ng-section-label">01 / Start with a broad idea</span><span class="ng-new">New</span></div>
    <h2>Design Thinking</h2>
    <p>Explore what people need, what you can deliver, and what can sustain itself. Turn a rough idea into a focused opportunity and a plan to test it.</p>
    <div class="ng-lens-tags">
      <span><i style="background:#c58a49"></i>Desirability</span>
      <span><i style="background:#56a69c"></i>Feasibility</span>
      <span><i style="background:#a08acc"></i>Viability</span>
    </div>
  </div>
  <div class="ng-venn" role="img" aria-label="Desirability, feasibility and viability overlap around your idea.">
    <span class="ng-venn-d" aria-hidden="true"></span><span class="ng-venn-f" aria-hidden="true"></span><span class="ng-venn-v" aria-hidden="true"></span>
    <b aria-hidden="true">Your<br>next idea</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.page_link("pages/Design Thinking.py", label="Explore Design Thinking →", use_container_width=False)
    st.markdown('<p class="ng-featured-note">15 guiding questions · Cited research · Practical experiments</p>', unsafe_allow_html=True)

st.markdown(
    '<div class="ng-grid-heading"><h2>Go deeper with a focused tool</h2><p>Choose the question you’re working on.</p></div>',
    unsafe_allow_html=True,
)

for offset in range(0, len(TOOLS), 2):
    columns = st.columns(2, gap="medium")
    for column, tool in zip(columns, TOOLS[offset:offset + 2]):
        with column, st.container(border=True):
            st.markdown(
                f"""<div class="ng-tool">
<div class="ng-tool-meta"><span class="ng-tool-dot" style="background:{tool['color']}"></span><span class="ng-tool-number">{tool['number']}</span></div>
<h3>{escape(tool['name'])}</h3>
<div class="ng-tool-purpose">{escape(tool['purpose'])}</div>
<p class="ng-tool-description">{escape(tool['description'])}</p>
<p class="ng-tool-detail">{escape(tool['detail'])}</p>
</div>""",
                unsafe_allow_html=True,
            )
            st.page_link(tool["path"], label=f"Open {tool['name']} →", use_container_width=False)

st.markdown(
    '<div class="ng-footer">Start broad. Make the assumptions visible. Find the next thing worth learning.</div>',
    unsafe_allow_html=True,
)
