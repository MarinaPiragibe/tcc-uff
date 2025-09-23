class ArquivoUtils:
    @staticmethod
    def salvar_no_arquivo(nome_arquivo: str, conteudo: str):
        with open(nome_arquivo, "a", encoding="utf-8") as arquivo_saida:
            arquivo_saida.write(conteudo)