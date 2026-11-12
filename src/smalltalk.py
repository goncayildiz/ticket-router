import re

from preprocess import normalize

# Mesajın başındaki/sonundaki nezaket kalıpları
PRE = r"(?:(?:hey|hi|hello|ok|okay|so|please|can you|could you|would you|will you)\s+)*"
POST = r"(?:\s+(?:please|thanks|thank you))?"

RULES = {
    "greeting": [
        r"(?:hi|hello|hey|howdy|hiya|greetings|good (?:morning|afternoon|evening))(?: there| everyone| assistant)?",
    ],
    "goodbye": [
        r"(?:bye|goodbye|good bye|bye bye|farewell|see you(?: later| soon| tomorrow)?|talk to you later|have a good (?:day|night))",
    ],
    "thank_you": [
        r"(?:thanks|thank you|thanks a lot|thanks so much|thank you so much|thank you very much|many thanks|much appreciated|i appreciate it|appreciate it)",
    ],
    "tell_joke": [
        r"(?:tell me|tell us|give me|say|got|do you have|know) (?:a |any |another |one )?(?:funny |good )?jokes?",
        r"make me laugh",
    ],
    "what_is_your_name": [
        r"(?:what(?:'s|s| is)? your name|what do they call you|what should i call you|who are you|what are you called)",
    ],
    "who_made_you": [
        r"(?:who (?:made|created|built|designed|developed|programmed|invented) you|who(?:'s|s| is) your (?:creator|maker|developer))",
    ],
    "where_are_you_from": [
        r"(?:where (?:are you from|do you (?:come from|live)|were you (?:born|made))|where is your home)",
    ],
    "how_old_are_you": [
        r"(?:how old are you|what(?:'s|s| is)? your age|what age are you|when were you (?:born|made|created))",
    ],
    "are_you_a_bot": [
        r"(?:are you (?:a |an )?(?:real )?(?:bot|robot|human|person|ai|machine|computer)(?: or (?:a |an )?(?:bot|robot|human|person|ai|machine|computer))?"
        r"|am i (?:talking|speaking) to (?:a |an )?(?:bot|robot|human|person|machine|computer))",
    ],
    "what_can_i_ask_you": [
        r"(?:what can i ask you|what can i ask|what can you do|what do you do|what are you able to do|what are your capabilities|how can you help(?: me)?|what can you help (?:me )?with)",
    ],
    "what_are_your_hobbies": [
        r"(?:what are your hobbies|what(?:'s|s| is) your hobby|what do you do for fun|what do you like to do|do you have (?:any )?hobbies)",
    ],
    "do_you_have_pets": [
        r"(?:do you have (?:any |a )?(?:pets?|dogs?|cats?)|do you own (?:a |any )?pets?|have you got (?:any )?pets?)",
    ],
        "meaning_of_life": [
        r".*(?:meaning of life|meaning behind life|purpose of life|purpose of existence|life's purpose|lifes purpose|purpose in life|what is our purpose|why are humans on earth|why are we here).*",
    ],
    "who_do_you_work_for": [
        r"(?:who do you work for|who(?:'s|s| is) your (?:boss|employer)|which company do you work for|who employs you|who are you working for)",
    ],
        "fun_fact": [
        r"(?:tell me |give me |share )?(?:a |another |one )?(?:fun|interesting|random|cool) facts?(?: about .+)?",
        r"tell me something (?:interesting|fun|cool)(?: about .+)?",
    ],
}

REPLIES = {
    "greeting": "Hello! I'm the support assistant. How can I help you today?",
    "goodbye": "Goodbye! Contact us any time you need help.",
    "thank_you": "You're welcome! Is there anything else I can help with?",
    "tell_joke": "Why did the dishwasher go to therapy? It had too many unresolved issues!",
    "what_is_your_name": "I'm the Support Ticket Router, your virtual support assistant.",
    "who_made_you": "I was built as a course project on intelligent applications.",
    "where_are_you_from": "I live in the cloud, but I was built right here as a student project.",
    "how_old_are_you": "I'm brand new! I was just built for this project.",
    "are_you_a_bot": "Yes, I'm an automated assistant. I route your request to the right team.",
    "what_can_i_ask_you": "Tell me your problem and I'll send it to the right team, such as billing, kitchen appliances or smart home.",
    "what_are_your_hobbies": "I enjoy sorting support tickets quickly and correctly!",
    "do_you_have_pets": "No pets for me, but I hear dishwashers are low-maintenance.",
    "meaning_of_life": "That's a big question! I stick to support tickets, but I hope you find a great answer.",
    "who_do_you_work_for": "I work for the support team and help route your request.",
    "fun_fact": "Fun fact: the first dishwasher was invented in 1886 by Josephine Cochrane.",
}

_COMPILED = {
    intent: re.compile(PRE + "(?:" + "|".join(pats) + ")" + POST)
    for intent, pats in RULES.items()
}


def match_small_talk(text):
    """Metni normalize edip kurallara bakar. Eşleşirse intent adını, yoksa None döner."""
    t = normalize(text)
    for intent, rx in _COMPILED.items():
        if rx.fullmatch(t):
            return intent
    return None


def small_talk_reply(intent):
    return REPLIES.get(intent)


if __name__ == "__main__":
    import json
    import pandas as pd
    from teams import team_of, SMALL_TALK_TEAM

    names = json.load(open("data/raw/label_names.json"))
    df = pd.read_csv("data/raw/validation.csv")
    df["pred"] = df.text.map(match_small_talk)
    df["team"] = df.label_name.map(team_of)

    st = df[df.team == SMALL_TALK_TEAM]
    ok = int((st.pred == st.label_name).sum())
    wrong = int(((st.pred.notna()) & (st.pred != st.label_name)).sum())
    miss = int(st.pred.isna().sum())
    print(f"small talk rows: {len(st)} | correct: {ok} | wrong intent: {wrong} | no match: {miss}")

    other = df[df.team != SMALL_TALK_TEAM]
    fp = other[other.pred.notna()]
    print(f"other rows: {len(other)} | FALSE small-talk matches: {len(fp)}")
    print(fp[["text", "label_name", "pred"]].head(15).to_string())

    print("\n--- small talk examples with NO match ---")
    print(st[st.pred.isna()][["text", "label_name"]].head(25).to_string())