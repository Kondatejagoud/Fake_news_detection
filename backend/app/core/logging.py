import logging
import sys

def setup_logging():
    # Clear existing handlers
    logging.root.handlers = []
    
    # Configure logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger("fake_detector")
