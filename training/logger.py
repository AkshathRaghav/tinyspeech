import logging
import os

class LoggerSetup:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.level = level
        self.setup_logger()

    def setup_logger(self):
        self.logger.setLevel(self.level)

        if not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)

        self.logger.propagate = False

    def get_logger(self):
        return self.logger
