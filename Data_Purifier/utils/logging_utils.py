import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def log_process(name, reason, code, impact):
    logging.info(f"--- Process: {name} ---")
    logging.info(f"Why: {reason}")
    logging.info(f"Code: {code}")
    logging.info(f"Impact: {impact}")