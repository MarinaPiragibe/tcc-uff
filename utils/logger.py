import logging
import os


class Logger:
    @staticmethod
    def configurar_logger(nome_arquivo="application.log", nivel=logging.DEBUG):
        """Configura o logger raiz de forma segura.

        - Garante que a pasta `logs/` exista.
        - Fecha e remove handlers de arquivo antigos antes de adicionar o novo.
        - Usa FileHandler com delay=True para evitar criação de arquivos vazios
          até que o primeiro registro seja escrito.
        """
        os.makedirs("logs", exist_ok=True)

        raiz = logging.getLogger()

        # Formato comum para todos os handlers
        formato = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Remove e fecha handlers de arquivo antigos para evitar múltiplos arquivos abertos
        for h in list(raiz.handlers):
            if isinstance(h, logging.FileHandler):
                try:
                    raiz.removeHandler(h)
                    h.close()
                except Exception:
                    pass

        # Remove handlers de stream extras para evitar mensagens duplicadas
        for h in list(raiz.handlers):
            if isinstance(h, logging.StreamHandler):
                try:
                    raiz.removeHandler(h)
                except Exception:
                    pass

        # Cria handlers novos (FileHandler com delay evita arquivo vazio até o primeiro write)
        file_handler = logging.FileHandler(f"logs/{nome_arquivo}", encoding="utf-8", delay=True)
        file_handler.setFormatter(formato)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formato)

        raiz.setLevel(nivel)
        raiz.addHandler(file_handler)
        raiz.addHandler(stream_handler)
