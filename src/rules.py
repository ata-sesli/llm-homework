"""ELIZA-style replies for greetings and small talk.

small_talk_reply() runs on preprocessed text before any classifier. Each rule must
match the whole message, so "hi my dishwasher is broken" is not treated as small
talk and still goes to the classifiers. Captured text is pronoun-reflected
("my order" -> "your order") as in ELIZA.

reply_for_intent() gives a canned reply when a classifier predicts one of the
CLINC small_talk intents.
"""

import re

REFLECTIONS = {
    "i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself",
    "am": "are", "i'm": "you're", "i've": "you've", "i'll": "you'll", "we": "you", "our": "your",
}

GREETING = "Hello! Describe your problem and I'll route it to the right team."

RULES = [
    (r"(hi+|hello+|hey+|hiya|good (morning|afternoon|evening)|merhaba|selam)( there| team| everyone)?",
     lambda m: GREETING),
    (r"((hi|hello|hey) )?how are you( doing)?( today)?",
     lambda m: "I'm doing well, thanks for asking! How can I help you today?"),
    (r"(thanks|thank you|thx|tesekkurler|tesekkur ederim)( so much| a lot)?( for (?P<what>.+))?",
     lambda m: f"You're welcome! Glad I could help with {reflect(m['what'])}." if m["what"] else "You're welcome!"),
    (r"(bye|goodbye|see you( later)?|good night|have a nice day|gorusuruz)",
     lambda m: "Goodbye! Have a nice day."),
    (r"are you (a )?(bot|robot|human|real person|real)",
     lambda m: "I'm an automated assistant. Requests I can't handle are passed to a human agent."),
]
COMPILED_RULES = [(re.compile(pattern), respond) for pattern, respond in RULES]

INTENT_REPLIES = {
    "greeting": GREETING,
    "goodbye": "Goodbye! Have a nice day.",
    "thank_you": "You're welcome!",
    "are_you_a_bot": "I'm an automated assistant. Requests I can't handle are passed to a human agent.",
    "what_is_your_name": "I'm the support assistant.",
    "who_made_you": "I was built as a course project for SWE015.",
    "who_do_you_work_for": "I help the customer support team route your messages.",
    "what_can_i_ask_you": "Tell me about a problem with an appliance, an order or an account and I'll route it.",
    "tell_joke": "Why did the fridge apply for a job? It wanted to be cool at work.",
    "fun_fact": "Fun fact: the first electric dishwasher was sold in the 1920s.",
    "how_old_are_you": "I'm younger than your oldest appliance.",
    "where_are_you_from": "I live on a server.",
    "do_you_have_pets": "No pets, just a lot of support tickets.",
    "what_are_your_hobbies": "Routing tickets is my favourite hobby.",
    "meaning_of_life": "To keep every appliance running, of course.",
}


def reflect(fragment):
    return " ".join(REFLECTIONS.get(word, word) for word in fragment.split())


def small_talk_reply(text):
    for pattern, respond in COMPILED_RULES:
        match = pattern.fullmatch(text)
        if match:
            return respond(match)
    return None


def reply_for_intent(intent):
    return INTENT_REPLIES.get(intent)
