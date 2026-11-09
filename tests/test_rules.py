import pytest

from src.preprocess import preprocess
from src.rules import reflect, reply_for_intent, small_talk_reply


def norm(text):
    return preprocess(text).text


@pytest.mark.parametrize("text", ["Hi!", "hello there", "Good morning", "Merhaba", "heyyyy"])
def test_greetings_get_a_reply(text):
    assert small_talk_reply(norm(text)).startswith("Hello")


def test_thanks_reflects_pronouns():
    reply = small_talk_reply(norm("Thanks for fixing my order"))
    assert reply == "You're welcome! Glad I could help with fixing your order."


def test_plain_thanks_and_turkish_thanks():
    assert small_talk_reply(norm("thank you so much")) == "You're welcome!"
    assert small_talk_reply(norm("Teşekkürler")) == "You're welcome!"


def test_name_statements_are_left_to_the_classifiers():
    # CLINC labels "my name is ..." as meta:change_user_name, not small talk.
    assert small_talk_reply(norm("my name is jason")) is None


def test_bot_and_goodbye():
    assert "automated assistant" in small_talk_reply(norm("are you a bot?"))
    assert small_talk_reply(norm("bye")).startswith("Goodbye")


@pytest.mark.parametrize(
    "text",
    ["hi my dishwasher is not draining", "thanks but the oven is still broken", "what is my balance"],
)
def test_messages_with_a_real_request_are_not_small_talk(text):
    assert small_talk_reply(norm(text)) is None


def test_reflect():
    assert reflect("i lost my card") == "you lost your card"


def test_reply_for_intent():
    assert reply_for_intent("tell_joke")
    assert reply_for_intent("greeting").startswith("Hello")
    assert reply_for_intent("balance") is None
