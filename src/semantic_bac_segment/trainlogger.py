import logging

class TrainLogger:
    def __init__(self, name, file_name='log.txt', level='INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create file handler and set level to debug
        fh = logging.FileHandler(file_name)
        fh.setLevel(level)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to ch and fh
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add ch and fh to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def log(self, message, level='INFO'):
        self._check_level(level)
        
        log_level=getattr(logging, level)
        self.logger.log(log_level, message)


    def is_level(self, level_str):
        self._check_level(level_str)
        return self.logger.level == getattr(logging, level_str.upper())
    

    def _check_level(self, level_str):
        valid_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
        assert level_str in valid_levels, f'Invalid log level: {level_str}. Must be one of: {valid_levels}'
