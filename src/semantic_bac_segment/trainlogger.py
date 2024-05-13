import logging

class TrainLogger:
    """
    A custom logger class for logging training information.

    Args:
        name (str): The name of the logger.
        file_name (str, optional): The name of the log file. Defaults to 'log.txt'.
        level (str, optional): The logging level. Defaults to 'INFO'.
    """
    
    def __init__(self, name: str, file_name: str = 'log.txt', level: str = 'INFO') -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler and file handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fh = logging.FileHandler(file_name)
        fh.setLevel(level)

        # Add formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add ch and fh to logger if not present
        if not self.logger.handlers:
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def log(self, message: str, level: str = 'INFO') -> None:
        """
        Logs a message at the specified level.

        Args:
            message (str): The message to be logged.
            level (str, optional): The logging level. Defaults to 'INFO'.

        Raises:
            AssertionError: If the specified log level is invalid.
        """
        self._check_level(level)
        
        log_level=getattr(logging, level)
        self.logger.log(log_level, message)

    def is_level(self, level_str: str) -> bool:
        """
        Checks if the current logging level matches the specified level.

        Args:
            level_str (str): The logging level to check.

        Returns:
            bool: True if the current logging level matches the specified level, False otherwise.

        Raises:
            AssertionError: If the specified log level is invalid.
        """
        self._check_level(level_str)
        return self.logger.level == getattr(logging, level_str.upper())    

    def _check_level(self, level_str: str) -> None:
        """
        Checks if the specified logging level is valid.

        Args:
            level_str (str): The logging level to check.

        Raises:
            AssertionError: If the specified log level is invalid.
        """
        valid_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
        assert level_str in valid_levels, f'Invalid log level: {level_str}. Must be one of: {valid_levels}'
