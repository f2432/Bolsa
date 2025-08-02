import logging

# Configure a logger for the application
logger = logging.getLogger("TradingApp")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    # File handler for logging to a file
    fh = logging.FileHandler("trading_app.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Optional: also log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
