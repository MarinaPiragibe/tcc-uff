import logging

class Logger:
    @staticmethod
    def configurar_logger(nome_arquivo="application.log", nivel=logging.DEBUG):
        logging.basicConfig(
            level=nivel,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"logs/{nome_arquivo}", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
