### Mock version for testing without actual DB connection
import logging
from datetime import datetime

def save_chat(user_id, user_name, direction, content, metadata=None):
    logging.info(f"🧪 MOCK save_chat called : user_id={user_id}, direction={direction}, content={content[:40]}")

def fetch_recent_history(user_id, top_n=5):
    logging.info(f"🧪 MOCK fetch_recent_history called : user_id={user_id}, top_n={top_n}")
    # Return dummy conversation turns
    q_hist = "mock question 1 | mock question 2"
    a_hist = "mock answer 1 | mock answer 2"
    return q_hist, a_hist