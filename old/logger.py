import logging

class Logger:

    def __init__(self, nome='', nivel=logging.DEBUG, nome_arquivo='application.logs'):
        self.nome = nome
        self.nivel = nivel
        self.nome_arquivo = nome_arquivo
        self._logger = self.iniciar_logger()
    
    def iniciar_logger(self):
        logger = logging.getLogger(self.nome)
        logger.setLevel(self.nivel)
        logger.propagate = False

        if not logger.handlers: 
            file_handler = logging.FileHandler(f'logs/{self.nome_arquivo}', encoding="utf-8")
            file_handler.setLevel(self.nivel)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.nivel)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def get_logger(self):
        return self._logger
