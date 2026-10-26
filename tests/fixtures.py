from copy import deepcopy
from types import SimpleNamespace
import json
import threading

import design_thinking as dt


BRIEF = {
    "idea": "Help independent neighbourhood bakeries reduce unsold bread while making good food affordable.",
    "target_users": "Independent bakery owners and nearby residents",
    "geography": "Singapore",
    "context": "Owners forecast demand manually.",
    "constraints": "A two-person team and a four-week pilot.",
    "business_model": "Commission or partner funding is an untested hypothesis.",
    "evidence": "No customer interviews have been conducted.",
}


def framing():
    return {
        "focus": "Provisional focus: independent Singapore bakeries and nearby residents.",
        "problem_statement": "Bakery owners need to reduce unsold bread without adding closing-time work.",
        "how_might_we": "How might we reduce unsold bread while preserving convenience and food quality?",
        "assumptions": ["Owners have predictable surpluses.", "Residents can collect before closing."],
        "alternative_directions": ["Improve forecasting before producing food.", "Partner with existing community collection networks."],
    }


def plan(queries_per_lens=2):
    return {
        "framing": framing(),
        "queries": {
            key: [f"Singapore bakery {key} evidence {i}" for i in range(queries_per_lens)]
            for key in dt.LENSES
        },
    }


def assessment(key, source_ids=None):
    if source_ids is None:
        source_ids = [f"S{list(dt.LENSES).index(key) + 1}"]
    return {
        "signal": "Mixed",
        "confidence": "Medium",
        "summary": "Relevant sector evidence is available, but this bakery offer remains untested.",
        "questions": [
            {"id": q["id"], "answer": "Sector evidence suggests an opportunity. Adoption by these owners is still an inference.", "basis": "Web evidence", "source_ids": source_ids, "gap": "Observe owners during an actual closing shift."}
            for q in dt.LENSES[key]["questions"]
        ],
        "critical_assumption": "Owners can offer surplus without adding substantial closing-time effort.",
        "experiment": {
            "hypothesis": "Owners will offer surplus during real closing shifts.",
            "method": "Run a manual collection pilot with a paper order sheet.",
            "participants": "Five independent neighbourhood bakeries and nearby residents.",
            "metric": "Number of owners completing three surplus listings.",
            "pass_threshold": "Proposed threshold: at least three of five owners complete all three listings.",
            "fail_signal": "If fewer than three owners participate, reframe the workflow before building software.",
            "timebox": "One week; assume one team member can coordinate the pilot.",
        },
    }


def synthesis():
    return {
        "recommendation": "Explore",
        "headline": "Test the closing-time workflow before building a marketplace.",
        "rationale": "Demand and collection feasibility are unproven. Start with direct observation.",
        "tensions": ["Lower prices could reduce waste but add owner effort.", "Collection convenience may conflict with a bakery's closing time."],
        "next_steps": [
            {"action": "Observe five closing shifts.", "learning_goal": "Understand the existing workaround."},
            {"action": "Run a manual collection pilot.", "learning_goal": "Measure completion and owner effort."},
            {"action": "Review contribution costs.", "learning_goal": "Understand whether the pilot can sustain itself."},
        ],
        "interview_questions": ["Tell me about the last time you had unsold bread.", "What did you do next?", "How long did that take?", "What alternatives have you tried?", "What made you stop using an alternative?"],
        "concepts": [
            {"name": "Manual collection", "description": "Coordinate nearby residents without new software.", "tradeoff": "Low build cost but high coordination effort."},
            {"name": "Forecasting support", "description": "Help owners plan the next day's production.", "tradeoff": "Requires reliable sales and demand data."},
        ],
        "priority_lens": "desirability",
    }


def report(brief=None):
    sources = [
        {"id": f"S{i}", "title": f"Public bakery research {key}", "url": f"https://example.org/{key}", "content": "Reported sector evidence about surplus bread.", "lens": key, "query": f"Singapore bakery {key}"}
        for i, key in enumerate(dt.LENSES, 1)
    ]
    return {
        "brief": deepcopy(brief or BRIEF), "framing": framing(),
        "lenses": {key: assessment(key) for key in dt.LENSES},
        "synthesis": synthesis(), "sources": sources,
        "research": {"queries": plan()["queries"], "warnings": [], "completed_searches": 6, "total_searches": 6},
        "created_at": "2026-10-26T07:00:00+00:00", "elapsed_seconds": 1.5,
    }


class FakeClient:
    """Schema-aware fake: exercises orchestration while never contacting a provider."""

    def __init__(self, mutator=None, **kwargs):
        self.mutator = mutator
        self.calls = []
        self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def create(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        data = json.loads(kwargs["messages"][1]["content"])
        with self.lock:
            self.calls.append((name, kwargs))
        if name == "design_research_plan":
            schema = kwargs["response_format"]["json_schema"]["schema"]
            count = schema["properties"]["queries"]["properties"]["desirability"]["minItems"]
            result = plan(count)
        elif name.endswith("_assessment"):
            result = assessment(data["lens"], [source["id"] for source in data["sources"]][:2])
        else:
            result = synthesis()
        if self.mutator:
            result = self.mutator(name, result, kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(result), refusal=None)
        )])


def search_result(key, query):
    # The same source is deliberately returned twice per lens to test deduplication.
    return {"results": [{"url": f"https://example.org/{query.split()[2]}", "title": "Public evidence", "content": "Retrieved sector research. The proposed concept has not been tested."}]}
