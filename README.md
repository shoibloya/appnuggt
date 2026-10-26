# nuggt

A Streamlit workshop with Ideation, Due Diligence, Aether, BUTA, and Design Thinking tools.

## Design Thinking

Start with a broad idea; audience, geography, constraints, funding and prior evidence are optional. The new tool frames a provisional problem and researches **desirability, feasibility and viability** using the same OpenAI (`gpt-4.1`) and Tavily APIs as the existing tools.

- 15 substantive questions: real user behavior and alternatives, delivery capability and dependencies, and sustainable economics/funding.
- Six focused searches by default, or nine for deeper coverage, run concurrently. Three lens assessments also run concurrently. Calls have timeouts and bounded retries; no open-ended agent loops.
- Distinct signals and evidence confidence, traceable citations, explicit unknowns, alternative concepts, and the assumptions that could change the decision.
- An independent evidence check corrects unsupported claims and checks proposed experiments before the final synthesis. Corrected web claims require a matching quotation from a retrieved excerpt.
- Three proposed experiments with participants, observable measures, pass/fail thresholds and timeboxes, followed by a non-leading interview guide.
- Session-specific persistent reports, a source/search appendix, and full PDF/Markdown downloads. Input edits and downloads do not rerun the APIs. Submit the form to generate a new report.

The report is an **early evidence review, not customer validation**. Team claims are treated as unverified input. Public research does not prove demand for the concept. Estimates, experiment thresholds and proposed segments must remain labeled as assumptions. There is deliberately no averaged success score that could hide a critical blocker.

### Research basis

The question bank is our synthesis of these methods, not an official IDEO questionnaire or scoring model:

- [IDEO: Design Thinking](https://www.ideo.com/methodology/design-thinking) — iterative human-centered design and the three connected lenses.
- [IDEO Design Kit: Frame Your Design Challenge](https://www.designkit.org/methods/frame-your-design-challenge.html) — define the people, context, desired outcome and constraints without prescribing a solution.
- [IDEO Design Kit: Interview](https://www.designkit.org/methods/interview.html) and [Get Feedback](https://www.designkit.org/methods/get-feedback.html) — learn from real experiences and realistic prototypes, including negative feedback.
- [Strategyzer: Reduce Risk and Uncertainty](https://www.strategyzer.com/library/how-to-systematically-reduce-the-risk-uncertainty-of-new-ideas) — break customer, implementation and financial uncertainty into testable assumptions.
- [Strategyzer: Test Card](https://www.strategyzer.com/library/validate-your-ideas-with-the-test-card) — specify the hypothesis, experiment, measurement and threshold before running a test.

The research pattern follows BUTA's tailored queries and cited evaluations, while adding solution-neutral framing, distinct evidence labels, explicit gaps, and a practical learning plan. Questions live in `design_thinking.LENSES`; analysis, research and exports are independent of Streamlit in `design_thinking.py`.

## Run locally

Use Python 3.11 or 3.12 in an isolated environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Provide the existing services through environment variables `OPENAI_API_KEY` and `TAVILY_API_KEY`, or an ignored `.streamlit/secrets.toml`:

```toml
apiKey = "your-existing-openai-key"
tapiKey = "your-existing-tavily-key"
```

The Design Thinking landing page loads without credentials. It reads them only when an evaluation is submitted. Keys are never placed in report data or passed to the model. The application does not persist reports to a server-side database; each report is held in its Streamlit session. Idea context is sent to OpenAI and generated research queries to Tavily.

## Validate

```sh
python -m unittest discover -s tests -v
```

Automated tests use deterministic API doubles and Streamlit AppTest; they do not consume API credits. Real-service behavior must additionally be checked in the deployed app with its configured secrets.

## Deploy

Streamlit Community Cloud serves `streamlit_app.py` from this repository's `main` branch at [nuggt-ai.streamlit.app](https://nuggt-ai.streamlit.app). Pushes to that branch update the deployment automatically. Existing Streamlit secrets are reused. Design Thinking is available at `/Design_Thinking` and in the multipage navigation.
