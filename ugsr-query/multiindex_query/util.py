import os
import re
import pandas as pd

def set_env_vars(ENV_VARS=None):
    """
    Set environment variables for Azure/OpenAI config.
    If a variable is set in ENV_VARS and is not a placeholder, use it as override.
    Otherwise, use the value from .env (already loaded via load_dotenv()).
    """
    if ENV_VARS is None:
        ENV_VARS = {}
    for k in ENV_VARS:
        value = ENV_VARS[k]
        # Use ENV_VARS value if not a placeholder, else keep .env value
        if value and not value.startswith('<') and not value.endswith('>'):
            os.environ[k] = value

