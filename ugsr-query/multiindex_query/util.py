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


def title_case_filename(filename):
    try:
        name, ext = os.path.splitext(filename)
        # Special case: if the name matches all caps with periods or digits (e.g., OP2.12U)
        if re.fullmatch(r'[a-z]{2,}[0-9.]*[a-z]?', name, re.IGNORECASE):
            return f"{name.upper()}{ext}"
        
        parts = re.split(r'[\s_\-\.]+', name)
        title_cased = ' '.join(p.capitalize() if not p.isupper() else p for p in parts)
        return f"{title_cased}{ext}"
    except Exception:
        return filename


def title_case_name(name):
    try:
        return ' '.join(part.capitalize() for part in name.strip().split())
    except Exception:
        return name


def resolve_reference_url(filename: str, original_url: str, supplement_files: dict) -> str:
    """
    Returns a reference link for the document.
    If the filename is listed in supplement_files, return the associated reference_link.
    Otherwise, return the original_url.
    """
    cleaned_filename = filename.strip().lower()
    for index_docs in supplement_files.values():
        for entry in index_docs:
            if entry.get("file_name", "").strip().lower() == cleaned_filename:
                return entry.get("reference_link", original_url)
    return original_url