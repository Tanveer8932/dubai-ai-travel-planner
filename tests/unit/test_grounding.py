"""AC2.6: research and composer outputs contain only ids present in tool results."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from factories import compose, research_bundle, trip_request

from dubai_planner.agents import attractions, composer, dining
from dubai_planner.agents.selection import FALLBACK, TOPPED_UP
from dubai_planner.llm import StructuredOutputError
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models.trace import TraceEventType
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import capture


def _candidates(toolbox: Toolbox) -> list[str]:
    return [a.id for a in toolbox.search_attractions(trip_request(),
                                                     limit=attractions.CANDIDATE_LIMIT)]


def _picks(*ids: str) -> str:
    return json.dumps({"picks": [{"id": i, "score": 0.9, "rationale": "r"} for i in ids]})


def test_injected_unknown_id_is_rejected(mock_llm: MockLLM, toolbox: Toolbox) -> None:
    real = _candidates(toolbox)
    mock_llm.script("attractions", _picks(real[3], "att_burj_al_imaginary", real[0],
                                          "htl_rove_downtown"))
    with capture() as events:
        out = attractions.run(attractions.AttractionsInput(request=trip_request()), mock_llm,
                              toolbox)
    ids = [o.id for o in out.options]
    assert "att_burj_al_imaginary" not in ids and "htl_rove_downtown" not in ids
    assert out.rejected_ids == ["att_burj_al_imaginary", "htl_rove_downtown"]
    assert ids[:2] == [real[3], real[0]]  # the model's valid picks keep its order
    assert set(ids) <= set(real)
    guard = [e for e in events if e.event is TraceEventType.GUARDRAIL]
    assert guard and guard[0].payload["rejected_ids"] == out.rejected_ids


def test_only_unknown_ids_triggers_repair(mock_llm: MockLLM, toolbox: Toolbox) -> None:
    real = _candidates(toolbox)
    mock_llm.script("attractions", _picks("att_fake_one", "att_fake_two"), _picks(real[1]))
    out = attractions.run(attractions.AttractionsInput(request=trip_request()), mock_llm,
                          toolbox)
    assert out.options[0].id == real[1] and out.llm_calls == 2
    assert out.rejected_ids == ["att_fake_one", "att_fake_two"]
    repair = mock_llm.calls[-1].messages[-1].content
    assert "candidate list" in repair


def test_persistent_hallucination_fails_loudly(mock_llm: MockLLM, toolbox: Toolbox) -> None:
    mock_llm.script("attractions", _picks("att_fake_one"), _picks("att_fake_two"))
    with pytest.raises(StructuredOutputError, match="attractions"):
        attractions.run(attractions.AttractionsInput(request=trip_request()), mock_llm, toolbox)


def test_opt_in_fallback_uses_tool_ranking(mock_llm: MockLLM, toolbox: Toolbox) -> None:
    mock_llm.script("dining", "garbage", "more garbage")
    with capture() as events:
        out = dining.run(dining.DiningInput(request=trip_request()), mock_llm, toolbox,
                         fallback=True)
    ranked = [r.id for r in toolbox.search_restaurants(trip_request(),
                                                       limit=dining.CANDIDATE_LIMIT)]
    assert out.used_fallback and [o.id for o in out.options] == ranked[: len(out.options)]
    assert all(o.rationale == FALLBACK for o in out.options)
    assert any(e.payload.get("fallback") == "deterministic_ranking" for e in events)


def test_short_selection_is_topped_up_from_ranking(mock_llm: MockLLM, toolbox: Toolbox) -> None:
    real = _candidates(toolbox)
    mock_llm.script("attractions", _picks(real[5]))
    out = attractions.run(attractions.AttractionsInput(request=trip_request()), mock_llm,
                          toolbox)
    assert len(out.options) == attractions.picks_wanted(trip_request())
    assert out.options[0].id == real[5]
    assert all(o.rationale == TOPPED_UP for o in out.options[1:])
    assert len({o.id for o in out.options}) == len(out.options)  # no duplicates


def test_duplicate_picks_are_collapsed(mock_llm: MockLLM, toolbox: Toolbox) -> None:
    real = _candidates(toolbox)
    mock_llm.script("attractions", _picks(real[0], real[0], real[1]))
    out = attractions.run(attractions.AttractionsInput(request=trip_request()), mock_llm,
                          toolbox)
    ids = [o.id for o in out.options]
    assert ids[:2] == [real[0], real[1]] and len(ids) == len(set(ids))


def test_composer_rejects_unknown_venue_then_repairs(mock_llm: MockLLM, toolbox: Toolbox) -> None:
    req = trip_request(budget_aed=Decimal("15000"))
    research = research_bundle(req, mock_llm, toolbox)
    bad = {"title": "t", "days": [
        {"day": d, "slots": [{"start": "10:00", "end": "11:00", "kind": "attraction",
                              "ref_id": "att_burj_al_imaginary"}]}
        for d in range(1, req.num_days + 1)]}
    mock_llm.script("composer", json.dumps(bad))  # then the generator's valid draft
    _r, _p, itin = compose(req, mock_llm, toolbox, research=research)
    assert "att_burj_al_imaginary" not in itin.venue_ids()
    assert itin.venue_ids() <= research.known_ids()
    assert "att_burj_al_imaginary" in mock_llm.calls[-1].messages[-1].content


def test_composer_hallucination_twice_fails(mock_llm: MockLLM, toolbox: Toolbox) -> None:
    req = trip_request(budget_aed=Decimal("15000"))
    research = research_bundle(req, mock_llm, toolbox)
    bad = json.dumps({"hotel_id": "htl_imaginary", "days": [{"day": 1, "slots": []}]})
    mock_llm.script("composer", bad, bad)
    with pytest.raises(StructuredOutputError, match="composer"):
        compose(req, mock_llm, toolbox, research=research)
    assert composer.AGENT == "composer"
