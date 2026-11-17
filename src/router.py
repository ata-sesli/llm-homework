"""Route one support message: normalize, reply to small talk, classify, pick a team.

The decision comes from the primary model (the quantized student in the app).
The other models are run on the same normalized text so they can be compared.
"""

import json
import time
from dataclasses import dataclass, field

from src.evaluation import OOS, predict_with_threshold
from src.paths import PROCESSED_DIR
from src.preprocess import preprocess
from src.rules import reply_for_intent, small_talk_reply

HUMAN_AGENT = "human agent"
SMALL_TALK_DOMAIN = "small_talk"
EMPTY_REPLY = "Please type a message describing your problem."


@dataclass
class ModelPrediction:
    key: str
    name: str
    intent: str  # "oos" when the model is out of scope or not confident enough
    top_intent: str  # the highest-scoring class, before the threshold
    confidence: float  # probability of top_intent
    team: str | None  # None for small talk
    latency_ms: float


@dataclass
class RouteResult:
    normalized: str
    model_codes: list
    corrections: list
    intent: str | None = None
    confidence: float | None = None
    team: str | None = None
    reply: str | None = None
    models: list = field(default_factory=list)


def load_intent_to_domain():
    return json.loads((PROCESSED_DIR / "labels.json").read_text())["intent_to_domain"]


class Router:
    def __init__(self, models, intent_to_domain, known_words, primary="student_int8"):
        self.models = models
        self.intent_to_domain = intent_to_domain
        self.known_words = known_words
        self.primary = primary

    def team_for(self, intent):
        if intent == OOS:
            return HUMAN_AGENT
        domain = self.intent_to_domain[intent]
        return None if domain == SMALL_TALK_DOMAIN else domain

    def predict(self, key, model, text):
        start = time.perf_counter()
        probs = model.predict_proba([text])
        latency_ms = (time.perf_counter() - start) * 1000
        top = int(probs[0].argmax())
        intent = str(predict_with_threshold(probs, model.classes, model.tau, model.appliance_tau)[0])
        return ModelPrediction(key, model.name, intent, str(model.classes[top]), float(probs[0, top]),
                               self.team_for(intent), latency_ms)

    def route(self, message):
        pre = preprocess(message, self.known_words)
        result = RouteResult(pre.text, pre.model_codes, pre.corrections)
        if not pre.text:
            result.reply = EMPTY_REPLY
            return result

        result.models = [self.predict(key, model, pre.text) for key, model in self.models.items()]
        rule_reply = small_talk_reply(pre.text)
        if rule_reply is not None:  # ELIZA rules come before any classifier
            result.reply = rule_reply
            return result

        decision = next(p for p in result.models if p.key == self.primary)
        result.intent, result.confidence, result.team = decision.intent, decision.confidence, decision.team
        if decision.team is None:
            result.reply = reply_for_intent(decision.intent)
        return result
