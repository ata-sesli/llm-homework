import numpy as np
import pytest

from src.models import RouterModel
from src.router import HUMAN_AGENT, Router
from src.rules import GREETING, INTENT_REPLIES

CLASSES = np.array(["appliance_repair", "balance", "greeting", "oos"])
INTENT_TO_DOMAIN = {"appliance_repair": "appliance_support", "balance": "banking",
                    "greeting": "small_talk", "oos": "oos"}


def fixed_model(name, probs, tau=0.5, appliance_tau=0.3):
    seen = []

    def predict_proba(texts):
        seen.extend(texts)
        return np.array([probs] * len(texts))
    model = RouterModel(name, CLASSES, tau, appliance_tau, predict_proba)
    model.seen = seen
    return model


def make_router(primary_probs, other_probs=(0.1, 0.8, 0.05, 0.05)):
    models = {"baseline": fixed_model("Baseline", other_probs),
              "student_int8": fixed_model("Student INT8", primary_probs)}
    return Router(models, INTENT_TO_DOMAIN, known_words=frozenset(), primary="student_int8")


def test_confident_prediction_is_routed_to_the_intent_domain():
    result = make_router((0.9, 0.05, 0.03, 0.02)).route("my oven is broken")
    assert result.intent == "appliance_repair"
    assert result.team == "appliance_support"
    assert result.confidence == pytest.approx(0.9)
    assert result.reply is None


def test_message_is_normalized_and_model_code_extracted_before_classification():
    router = make_router((0.9, 0.05, 0.03, 0.02))
    result = router.route("Bulaşık makinem çalışmıyor, model DW-6534")
    assert result.model_codes == ["DW-6534"]
    assert "dishwasher" in result.normalized
    assert "dw" not in result.normalized.split()
    assert router.models["student_int8"].seen == [result.normalized]


def test_low_confidence_goes_to_human_agent():
    result = make_router((0.2, 0.35, 0.25, 0.2)).route("something about my thing")
    assert result.intent == "oos"
    assert result.team == HUMAN_AGENT


def test_out_of_scope_prediction_goes_to_human_agent():
    result = make_router((0.05, 0.05, 0.05, 0.85)).route("what is the capital of peru")
    assert result.team == HUMAN_AGENT


def test_small_talk_rule_replies_without_routing():
    result = make_router((0.9, 0.05, 0.03, 0.02)).route("Hello there")
    assert result.reply == GREETING
    assert result.team is None


def test_predicted_small_talk_intent_gets_canned_reply():
    result = make_router((0.05, 0.05, 0.85, 0.05)).route("good to meet you assistant")
    assert result.intent == "greeting"
    assert result.reply == INTENT_REPLIES["greeting"]
    assert result.team is None


def test_every_model_is_reported_side_by_side():
    result = make_router((0.2, 0.35, 0.25, 0.2)).route("my balance please")
    rows = {row.key: row for row in result.models}
    assert set(rows) == {"baseline", "student_int8"}
    assert rows["baseline"].intent == "balance" and rows["baseline"].team == "banking"
    assert rows["baseline"].confidence == pytest.approx(0.8)
    assert rows["student_int8"].team == HUMAN_AGENT
    assert all(row.latency_ms >= 0 for row in result.models)


def test_empty_message_is_not_classified():
    router = make_router((0.9, 0.05, 0.03, 0.02))
    result = router.route("  !!! ")
    assert result.team is None and result.models == []
    assert result.reply
    assert router.models["student_int8"].seen == []
