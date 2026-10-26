"""Regression checks for batching edits and explicitly starting research.

Browser tests cover actual Enter/Cmd+Enter events and visual hint removal. These
checks exercise Streamlit's emitted form configuration and server-side run gates.
No provider call is allowed to escape the mocks.
"""

from contextlib import ExitStack
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.fixtures import BRIEF, report


ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "Design Thinking.py": "Evaluate my idea",
    "Ideation.py": "Create Innovation Pitch",
    "Due Diligence.py": "Conduct Due Diligence",
    "Aether.py": "Generate Workshop Report",
    "BUTA.py": "Run BUTA",
}


def run_button(app, label):
    return next(button for button in app.button if button.label == label)


class ButtonOnlyFormsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "TAVILY_API_KEY": "test-tavily-key",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def providers(self, stack):
        model = stack.enter_context(patch("langchain_openai.ChatOpenAI.invoke", side_effect=AssertionError("Unexpected model request")))
        agent = stack.enter_context(patch("langchain.agents.AgentExecutor.invoke", side_effect=AssertionError("Unexpected research request")))
        evaluate = stack.enter_context(patch("design_thinking.evaluate_idea", side_effect=AssertionError("Unexpected evaluation request")))
        return model, agent, evaluate

    def app(self, filename):
        return AppTest.from_file(str(ROOT / "pages" / filename), default_timeout=20)

    def assert_clean(self, app):
        self.assertEqual([item.message for item in app.exception], [])

    def test_all_input_forms_disable_keyboard_submit_and_batch_fields(self):
        for filename in PAGES:
            with self.subTest(page=filename), ExitStack() as stack:
                providers = self.providers(stack)
                app = self.app(filename).run()
                self.assert_clean(app)
                forms = app.get("form")
                self.assertEqual(len(forms), 1)
                form = forms[0].proto.form
                self.assertFalse(form.enter_to_submit)
                self.assertTrue(form.form_id)
                fields = [*app.sidebar.text_input, *app.sidebar.text_area]
                self.assertTrue(fields)
                for field in fields:
                    self.assertEqual(field.proto.form_id, form.form_id)
                for provider in providers:
                    provider.assert_not_called()

    def test_editing_inputs_and_ordinary_reruns_do_not_start_research(self):
        for filename in PAGES:
            with self.subTest(page=filename), ExitStack() as stack:
                providers = self.providers(stack)
                app = self.app(filename).run()
                self.assert_clean(app)
                # A rerun must remain inert even if the browser has pending edits.
                app.sidebar.text_area[0].set_value("A detailed draft to finish editing before running research.")
                app.run()
                self.assert_clean(app)
                app.run()
                self.assert_clean(app)
                for provider in providers:
                    provider.assert_not_called()

    def test_stale_shared_inputs_cannot_restart_ideation_or_due_diligence(self):
        for filename in ("Ideation.py", "Due Diligence.py"):
            with self.subTest(page=filename), ExitStack() as stack:
                providers = self.providers(stack)
                app = self.app(filename)
                app.session_state["user_input"] = {
                    "corporate": "Previously submitted company",
                    "target_company": "Previously submitted company",
                    "company_type": "Well-known", "industry": "Example industry",
                }
                app.run()
                self.assert_clean(app)
                app.run()
                self.assert_clean(app)
                for provider in providers:
                    provider.assert_not_called()

    def test_explicit_legacy_buttons_still_reach_the_research_pipeline(self):
        for filename, label in PAGES.items():
            if filename == "Design Thinking.py":
                continue
            with self.subTest(page=filename), ExitStack() as stack:
                model, agent, evaluate = self.providers(stack)
                # Stop at the first mocked model request, rather than fabricate
                # entire old reports or accidentally make subsequent requests.
                model.side_effect = lambda *args, **kwargs: st.stop()
                app = self.app(filename).run()
                self.assert_clean(app)
                for field in app.sidebar.text_input:
                    field.set_value("Example pilot information")
                for field in app.sidebar.text_area:
                    field.set_value("Detailed example pilot information for an explicit submission.")
                run_button(app, label).click().run()
                self.assert_clean(app)
                self.assertEqual(model.call_count, 1)
                agent.assert_not_called()
                evaluate.assert_not_called()

    def test_explicit_design_thinking_button_submits_the_completed_brief_once(self):
        with ExitStack() as stack:
            model, agent, evaluate = self.providers(stack)
            evaluate.side_effect = lambda brief, *args, **kwargs: report(brief)
            app = self.app("Design Thinking.py").run()
            app.text_area(key="dt_input_idea").set_value(BRIEF["idea"])
            app.text_input(key="dt_input_target_users").set_value(BRIEF["target_users"])
            run_button(app, PAGES["Design Thinking.py"]).click().run()
            self.assert_clean(app)
            evaluate.assert_called_once()
            submitted_brief = evaluate.call_args.args[0]
            self.assertEqual(submitted_brief["idea"], BRIEF["idea"])
            self.assertEqual(submitted_brief["target_users"], BRIEF["target_users"])
            self.assertIn("dt_report", app.session_state)
            app.run()
            self.assert_clean(app)
            evaluate.assert_called_once()
            model.assert_not_called()
            agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
