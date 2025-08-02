import logging

def setup_logger(logfile='trading_app.log'):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

logger = setup_logger()
