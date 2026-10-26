from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.fixtures import BRIEF, report


PAGE = Path(__file__).resolve().parents[1] / "pages" / "Design Thinking.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


class DesignThinkingUITests(unittest.TestCase):
    def setUp(self):
        environment = {key: value for key, value in os.environ.items() if key not in ("OPENAI_API_KEY", "TAVILY_API_KEY")}
        self.environment = patch.dict(os.environ, environment, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def app(self):
        app = AppTest.from_file(str(PAGE), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_landing_loads_without_secrets_and_example_does_not_call_api(self):
        with patch("design_thinking.evaluate_idea") as evaluate:
            app = self.app()
            self.assertEqual(len(app.error), 0)
            button(app, "Try an example").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("bakeries", app.text_area(key="dt_input_idea").value)
            evaluate.assert_not_called()

    def test_short_input_is_rejected_before_api_access(self):
        with patch("design_thinking.evaluate_idea") as evaluate:
            app = self.app()
            app.text_area(key="dt_input_idea").set_value("A bakery app")
            button(app, "Evaluate my idea").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("30 characters" in item.value for item in app.error))
            evaluate.assert_not_called()

    def test_missing_credentials_gives_friendly_configuration_error(self):
        with patch("design_thinking.evaluate_idea") as evaluate:
            app = self.app()
            app.text_area(key="dt_input_idea").set_value(BRIEF["idea"])
            button(app, "Evaluate my idea").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("not configured" in item.value for item in app.error))
            evaluate.assert_not_called()

    def test_completed_report_persists_on_input_rerun_and_failed_new_evaluation(self):
        def evaluate(brief, *args, **kwargs):
            kwargs["on_progress"](0.5, "Gathering evidence")
            return report(brief)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key", "TAVILY_API_KEY": "test-tavily-key"}), patch("design_thinking.evaluate_idea", side_effect=evaluate) as evaluator:
            app = self.app()
            app.text_area(key="dt_input_idea").set_value(BRIEF["idea"])
            button(app, "Evaluate my idea").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(evaluator.call_count, 1)
            original = deepcopy(app.session_state["dt_report"])
            downloads = deepcopy(app.session_state["dt_downloads"])
            self.assertTrue(downloads["pdf"].startswith(b"%PDF-"))
            self.assertEqual(len(app.tabs), 6)
            # Form edits are submitted as one batch. The example is an outside-
            # form widget callback that causes a real rerun with a changed brief.
            button(app, "Try an example").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["dt_report"], original)
            self.assertEqual(app.session_state["dt_downloads"], downloads)
            self.assertEqual(evaluator.call_count, 1)
            self.assertTrue(any("last completed evaluation" in item.value for item in app.info))
            # A plain rerun models download/tab/widget reruns: no repeat API spend.
            app.run()
            self.assertEqual(evaluator.call_count, 1)
            evaluator.side_effect = RuntimeError("private-provider-error test-openai-key")
            button(app, "Evaluate my idea").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["dt_report"], original)
            self.assertTrue(any("couldn’t finish" in item.value for item in app.error))
            self.assertFalse(any("private-provider" in item.value for item in app.error))


if __name__ == "__main__":
    unittest.main()
