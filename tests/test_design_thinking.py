from copy import deepcopy
import json
import threading
import unittest
from unittest.mock import patch

import design_thinking as dt
from tests.fixtures import BRIEF, FakeClient, assessment, report, search_result


class BriefAndSourceTests(unittest.TestCase):
    def test_requires_meaningful_idea_and_bounds_context(self):
        for value in ({"idea": "short"}, {"idea": "x" * 5001}, {"idea": ["text"]}, None):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                dt.validate_brief(value)
        clean = dt.validate_brief({"idea": "  " + BRIEF["idea"] + "  "})
        self.assertEqual(clean["idea"], BRIEF["idea"])
        self.assertEqual(set(clean), set(dt.BRIEF_FIELDS))
        self.assertEqual(clean["evidence"], "")

    def test_rejects_unsafe_or_nonpublic_source_urls(self):
        rejected = ["javascript:alert(1)", "file:///etc/passwd", "https://localhost/path", "http://127.0.0.1/", "http://192.168.1.2/a", "https://x.internal/a", "https://user:pass@example.org/a", "https://example.org:9000/a", "https://example.org/a b", None]
        for value in rejected:
            with self.subTest(url=value):
                self.assertIsNone(dt.safe_source_url(value))
        self.assertEqual(dt.safe_source_url("https://example.org/a?x=1#section"), "https://example.org/a?x=1")

    def test_model_links_are_not_accepted_as_research_sources(self):
        clean = dt._remove_model_links({"answer": "See [invented study](https://bad.example/a) and https://bad.example/b", "ids": ["S1"]})
        self.assertNotIn("https://", json.dumps(clean))
        self.assertEqual(clean["ids"], ["S1"])


class ResearchTests(unittest.TestCase):
    def test_partial_search_failure_keeps_usable_research_and_callback_on_main_thread(self):
        callbacks = []
        main_thread = threading.get_ident()

        def search(key, query):
            if query.endswith(" 0"):
                raise RuntimeError("provider-private-response")
            return search_result(key, query)

        with patch.object(dt, "_search", side_effect=search):
            sources, research = dt._collect_research(
                {key: [f"Singapore bakery {key} 0", f"Singapore bakery {key} 1"] for key in dt.LENSES},
                "test-tavily-key", lambda *args: callbacks.append((threading.get_ident(), args)),
            )
        self.assertEqual(research["completed_searches"], 3)
        self.assertEqual(len(sources), 3)
        self.assertEqual(len(research["warnings"]), 3)
        self.assertEqual({item[0] for item in callbacks}, {main_thread})
        self.assertEqual(len(callbacks), 6)
        self.assertNotIn("provider-private-response", json.dumps(research))

    def test_all_search_failure_is_an_actionable_safe_error(self):
        with patch.object(dt, "_search", side_effect=RuntimeError("do-not-expose-private-provider-payload")):
            with self.assertRaisesRegex(dt.EvaluationError, "No usable research sources") as raised:
                dt._collect_research({key: [key] for key in dt.LENSES}, "test-key", lambda *args: None)
        self.assertNotIn("private-provider", str(raised.exception))

    def test_sources_have_stable_ids_and_duplicate_urls_per_lens_are_removed(self):
        from tests.fixtures import plan
        with patch.object(dt, "_search", side_effect=search_result):
            sources, research = dt._collect_research(plan()["queries"], "test-key", lambda *args: None)
        self.assertEqual([s["id"] for s in sources], ["S1", "S2", "S3"])
        self.assertEqual([s["lens"] for s in sources], list(dt.LENSES))
        self.assertEqual(research["completed_searches"], 6)


class ModelAndPipelineTests(unittest.TestCase):
    def run_pipeline(self, client=None, depth=2):
        client = client or FakeClient()
        with patch("openai.OpenAI", return_value=client), patch.object(dt, "_search", side_effect=search_result):
            result = dt.evaluate_idea(BRIEF, "test-openai-key", "test-tavily-key", queries_per_lens=depth)
        return result, client

    def test_full_pipeline_has_complete_ordered_answers_and_only_retrieved_sources(self):
        result, client = self.run_pipeline()
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(result["research"]["total_searches"], 6)
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(result["brief"], BRIEF)
        for key, spec in dt.LENSES.items():
            answers = result["lenses"][key]["questions"]
            self.assertEqual([a["id"] for a in answers], [q["id"] for q in spec["questions"]])
            allowed = {s["id"] for s in result["sources"]}
            self.assertTrue(all(set(a["source_ids"]) <= allowed for a in answers))
        self.assertEqual(sum(len(a["questions"]) for a in result["lenses"].values()), 15)

    def test_all_lenses_share_evidence_and_retain_cross_lens_citations(self):
        competitor_finding = "An existing resident surplus-sales alternative operates in the target geography."

        def search(key, query):
            response = search_result(key, query)
            if "viability" in query:
                response["results"][0]["content"] = competitor_finding
            return response

        def mutate(name, value, kwargs):
            if name == "desirability_assessment":
                # The evidence was retrieved by viability research, but it also
                # directly answers D2 about existing user alternatives.
                value["questions"][1]["answer"] = competitor_finding
                value["questions"][1]["source_ids"] = ["S3"]
            return value

        client = FakeClient(mutate)
        with patch("openai.OpenAI", return_value=client), patch.object(dt, "_search", side_effect=search):
            result = dt.evaluate_idea(BRIEF, "test-openai-key", "test-tavily-key")
        expected_ids = {source["id"] for source in result["sources"]}
        lens_calls = [(name, kwargs) for name, kwargs in client.calls if name.endswith("_assessment")]
        self.assertEqual(len(lens_calls), 3)
        for name, kwargs in lens_calls:
            with self.subTest(lens=name):
                shared_sources = json.loads(kwargs["messages"][1]["content"])["sources"]
                self.assertEqual({source["id"] for source in shared_sources}, expected_ids)
                self.assertTrue(any(source["content"] == competitor_finding for source in shared_sources))
        desirability_answer = result["lenses"]["desirability"]["questions"][1]
        self.assertEqual(desirability_answer["source_ids"], ["S3"])
        self.assertEqual(desirability_answer["basis"], "Web evidence")
        self.assertEqual(next(s["lens"] for s in result["sources"] if s["id"] == "S3"), "viability")

    def test_deep_mode_is_bounded_to_nine_searches(self):
        result, _ = self.run_pipeline(depth=3)
        self.assertEqual(result["research"]["total_searches"], 9)
        self.assertTrue(all(len(q) == 3 for q in result["research"]["queries"].values()))

    def test_duplicate_search_plan_retries_once_then_fails(self):
        def mutate(name, value, kwargs):
            if name == "design_research_plan":
                value["queries"]["viability"][0] = value["queries"]["feasibility"][0]
            return value
        client = FakeClient(mutate)
        with patch("openai.OpenAI", return_value=client), patch.object(dt, "_search") as search:
            with self.assertRaises(dt.EvaluationError):
                dt.evaluate_idea(BRIEF, "test-key", "test-key")
        self.assertEqual(len(client.calls), 2)
        search.assert_not_called()

    def test_hallucinated_citation_ids_are_removed_and_unsupported_answers_downgraded(self):
        def mutate(name, value, kwargs):
            if name.endswith("_assessment"):
                value["questions"][0]["source_ids"] = ["S999"]
            return value
        result, _ = self.run_pipeline(FakeClient(mutate))
        for lens in result["lenses"].values():
            self.assertEqual(lens["questions"][0]["source_ids"], [])
            self.assertEqual(lens["questions"][0]["basis"], "Hypothesis")
            self.assertEqual(lens["confidence"], "Low")
            self.assertIn("could not be verified", lens["questions"][0]["gap"])

    def test_missing_lens_sources_forces_unknown_signal(self):
        client = FakeClient()
        result = dt._assess_lens(client, "desirability", BRIEF, report()["framing"], [])
        self.assertEqual((result["signal"], result["confidence"]), ("Unknown", "Low"))

    def test_all_hallucinated_citations_force_unknown_signal(self):
        def mutate(name, value, kwargs):
            if name.endswith("_assessment"):
                value["signal"] = "Promising"
                for question in value["questions"]:
                    question["source_ids"] = ["S999"]
            return value
        result, _ = self.run_pipeline(FakeClient(mutate))
        for lens in result["lenses"].values():
            self.assertEqual((lens["signal"], lens["confidence"]), ("Unknown", "Low"))
            self.assertTrue(all(q["basis"] == "Hypothesis" for q in lens["questions"]))

    def test_one_lens_failure_preserves_other_lenses_and_marks_gap(self):
        def mutate(name, value, kwargs):
            if name == "feasibility_assessment":
                raise RuntimeError("private-provider-payload")
            return value
        result, _ = self.run_pipeline(FakeClient(mutate))
        self.assertEqual(result["lenses"]["feasibility"]["signal"], "Unknown")
        self.assertEqual(result["lenses"]["desirability"]["signal"], "Mixed")
        self.assertTrue(result["research"]["warnings"])
        self.assertNotIn("private-provider-payload", json.dumps(result))

    def test_provider_failure_does_not_leak_sensitive_error_response(self):
        client = FakeClient()
        client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("test-key private-provider-payload"))
        with self.assertRaises(dt.EvaluationError) as raised:
            dt._json_call(client, "test", "JSON", {}, dt.SYNTHESIS_SCHEMA)
        self.assertNotIn("test-key", str(raised.exception))
        self.assertNotIn("private-provider-payload", str(raised.exception))

    def test_synthesis_failure_preserves_usable_lens_reports(self):
        def mutate(name, value, kwargs):
            if name == "design_synthesis":
                raise RuntimeError("private-provider-payload")
            return value
        result, _ = self.run_pipeline(FakeClient(mutate))
        self.assertEqual(result["synthesis"]["recommendation"], "Explore")
        self.assertIn("incomplete", result["synthesis"]["rationale"])
        self.assertTrue(any("combined synthesis" in w for w in result["research"]["warnings"]))

    def test_missing_keys_and_invalid_depth_do_not_contact_providers(self):
        with patch("openai.OpenAI") as client:
            with self.assertRaises(dt.EvaluationError):
                dt.evaluate_idea(BRIEF, "", "")
            with self.assertRaises(ValueError):
                dt.evaluate_idea(BRIEF, "test-key", "test-key", queries_per_lens=100)
        client.assert_not_called()


class ExportTests(unittest.TestCase):
    def test_markdown_and_pdf_include_the_complete_review(self):
        sample = report()
        sample["brief"]["context"] = "A <b>plain-text</b> input & a café's closing shift."
        markdown = dt.to_markdown(sample)
        for spec in dt.LENSES.values():
            for question in spec["questions"]:
                self.assertIn(question["id"], markdown)
                self.assertIn(question["question"], markdown)
        self.assertIn(sample["lenses"]["desirability"]["experiment"]["pass_threshold"], markdown)
        self.assertIn("https://example.org/desirability", markdown)
        pdf = dt.build_pdf(sample)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 10000)


if __name__ == "__main__":
    unittest.main()
