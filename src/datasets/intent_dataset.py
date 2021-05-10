from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.io import json_load


class IntentDataset(Dataset):
    LABELS = (
        "accept_reservations",
        "account_blocked",
        "alarm",
        "application_status",
        "apr",
        "are_you_a_bot",
        "balance",
        "bill_balance",
        "bill_due",
        "book_flight",
        "book_hotel",
        "calculator",
        "calendar",
        "calendar_update",
        "calories",
        "cancel",
        "cancel_reservation",
        "car_rental",
        "card_declined",
        "carry_on",
        "change_accent",
        "change_ai_name",
        "change_language",
        "change_speed",
        "change_user_name",
        "change_volume",
        "confirm_reservation",
        "cook_time",
        "credit_limit",
        "credit_limit_change",
        "credit_score",
        "current_location",
        "damaged_card",
        "date",
        "definition",
        "direct_deposit",
        "directions",
        "distance",
        "do_you_have_pets",
        "exchange_rate",
        "expiration_date",
        "find_phone",
        "flight_status",
        "flip_coin",
        "food_last",
        "freeze_account",
        "fun_fact",
        "gas",
        "gas_type",
        "goodbye",
        "greeting",
        "how_busy",
        "how_old_are_you",
        "improve_credit_score",
        "income",
        "ingredient_substitution",
        "ingredients_list",
        "insurance",
        "insurance_change",
        "interest_rate",
        "international_fees",
        "international_visa",
        "jump_start",
        "last_maintenance",
        "lost_luggage",
        "make_call",
        "maybe",
        "meal_suggestion",
        "meaning_of_life",
        "measurement_conversion",
        "meeting_schedule",
        "min_payment",
        "mpg",
        "new_card",
        "next_holiday",
        "next_song",
        "no",
        "nutrition_info",
        "oil_change_how",
        "oil_change_when",
        "order",
        "order_checks",
        "order_status",
        "pay_bill",
        "payday",
        "pin_change",
        "play_music",
        "plug_type",
        "pto_balance",
        "pto_request",
        "pto_request_status",
        "pto_used",
        "recipe",
        "redeem_rewards",
        "reminder",
        "reminder_update",
        "repeat",
        "replacement_card_duration",
        "report_fraud",
        "report_lost_card",
        "reset_settings",
        "restaurant_reservation",
        "restaurant_reviews",
        "restaurant_suggestion",
        "rewards_balance",
        "roll_dice",
        "rollover_401k",
        "routing",
        "schedule_maintenance",
        "schedule_meeting",
        "share_location",
        "shopping_list",
        "shopping_list_update",
        "smart_home",
        "spelling",
        "spending_history",
        "sync_device",
        "taxes",
        "tell_joke",
        "text",
        "thank_you",
        "time",
        "timer",
        "timezone",
        "tire_change",
        "tire_pressure",
        "todo_list",
        "todo_list_update",
        "traffic",
        "transactions",
        "transfer",
        "translate",
        "travel_alert",
        "travel_notification",
        "travel_suggestion",
        "uber",
        "update_playlist",
        "user_name",
        "vaccines",
        "w2",
        "weather",
        "what_are_your_hobbies",
        "what_can_i_ask_you",
        "what_is_your_name",
        "what_song",
        "where_are_you_from",
        "whisper_mode",
        "who_do_you_work_for",
        "who_made_you",
        "yes",
    )

    @classmethod
    def load(cls, json_path: Path, **kwargs):
        data = json_load(json_path)
        return cls(data, **kwargs)

    def __init__(self, data, tokenizer=None, is_bert_tokenizer=False):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_to_label = {label: i for i, label in enumerate(self.LABELS)}
        self.is_bert_tokenizer = is_bert_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        ret = {
            "id": sample["id"],
            "input_ids": (
                torch.as_tensor(self.tokenizer(sample["text"]), dtype=torch.long)
                if not self.is_bert_tokenizer
                else torch.as_tensor(self.tokenizer(sample["text"])["input_ids"], dtype=torch.long)
            ),
        }

        if "intent" in sample:
            ret.update({"label": self.intent_to_label[sample["intent"]]})

        return ret
