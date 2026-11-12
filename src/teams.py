import json
from pathlib import Path

DOMAINS = {
    "Billing & Payments": [
        # banking
        "transfer", "transactions", "balance", "freeze_account", "pay_bill",
        "bill_balance", "bill_due", "interest_rate", "routing", "min_payment",
        "order_checks", "pin_change", "report_fraud", "account_blocked",
        "spending_history",
        # credit cards
        "credit_score", "report_lost_card", "credit_limit", "rewards_balance",
        "new_card", "application_status", "card_declined", "international_fees",
        "apr", "redeem_rewards", "credit_limit_change", "damaged_card",
        "replacement_card_duration", "improve_credit_score", "expiration_date",
    ],
    "Kitchen Appliance Support": [
        "recipe", "restaurant_reviews", "calories", "nutrition_info",
        "restaurant_suggestion", "ingredients_list", "ingredient_substitution",
        "cook_time", "food_last", "meal_suggestion", "restaurant_reservation",
        "confirm_reservation", "how_busy", "cancel_reservation",
        "accept_reservations",
    ],
    "Smart Home & Orders": [
        "shopping_list", "shopping_list_update", "next_song", "play_music",
        "update_playlist", "todo_list", "todo_list_update", "calendar",
        "calendar_update", "what_song", "order", "order_status", "reminder",
        "reminder_update", "smart_home",
    ],
    "Travel & Commute": [
        # auto & commute
        "traffic", "directions", "gas", "gas_type", "distance",
        "current_location", "mpg", "oil_change_when", "oil_change_how",
        "jump_start", "uber", "schedule_maintenance", "last_maintenance",
        "tire_pressure", "tire_change",
        # travel
        "book_flight", "book_hotel", "car_rental", "travel_suggestion",
        "travel_alert", "travel_notification", "carry_on", "timezone",
        "vaccines", "translate", "flight_status", "international_visa",
        "lost_luggage", "plug_type", "exchange_rate",
    ],
    "Device & Assistant Settings": [
        # utility
        "time", "alarm", "share_location", "find_phone", "weather", "text",
        "spelling", "make_call", "timer", "date", "calculator",
        "measurement_conversion", "flip_coin", "roll_dice", "definition",
        # meta
        "change_accent", "change_user_name", "cancel", "user_name",
        "reset_settings", "whisper_mode", "repeat", "no", "yes", "maybe",
        "change_language", "change_ai_name", "change_speed", "change_volume",
        "sync_device",
    ],
    "Employee Services": [
        "direct_deposit", "pto_request", "taxes", "payday", "w2", "pto_balance",
        "pto_request_status", "next_holiday", "insurance", "insurance_change",
        "schedule_meeting", "pto_used", "meeting_schedule", "rollover_401k",
        "income",
    ],
}

SMALL_TALK = [
    "greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you",
    "what_is_your_name", "who_made_you", "thank_you", "what_can_i_ask_you",
    "what_are_your_hobbies", "do_you_have_pets", "are_you_a_bot",
    "meaning_of_life", "who_do_you_work_for", "fun_fact",
]

SMALL_TALK_TEAM = "Small talk (rule-based reply)"
HUMAN_TEAM = "Human agent"

INTENT_TO_TEAM = {}
for team, intents in DOMAINS.items():
    for i in intents:
        INTENT_TO_TEAM[i] = team
for i in SMALL_TALK:
    INTENT_TO_TEAM[i] = SMALL_TALK_TEAM
INTENT_TO_TEAM["oos"] = HUMAN_TEAM


def team_of(intent_name: str) -> str:
    return INTENT_TO_TEAM.get(intent_name, HUMAN_TEAM)


if __name__ == "__main__":
    names = json.load(open(Path("data/raw/label_names.json")))
    mapped = set(INTENT_TO_TEAM)
    real = set(names)
    print("labels in dataset:", len(real))
    print("NOT mapped (dataset'te var, tabloda yok):", sorted(real - mapped))
    print("UNKNOWN (tabloda var, dataset'te yok):", sorted(mapped - real))
    from collections import Counter
    print(Counter(INTENT_TO_TEAM[n] for n in names if n in INTENT_TO_TEAM))