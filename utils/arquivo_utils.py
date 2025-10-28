import logging
import os
import csv
import numpy as np
import torch


class ArquivoUtils:

    @staticmethod
    def salvar_no_arquivo(nome_arquivo: str, conteudo: str):
        with open(nome_arquivo, "a", encoding="utf-8") as arquivo_saida:
            arquivo_saida.write(conteudo)

    @staticmethod
    def salvar_modelo(args, estado_modelo, estado_otimizador):
        try:
            logging.info(f"Salvando modelo executado")
            nome_arquivo = f"{args['modelo_base']}_{args['data_execucao']}.pth"

            caminho_salvamento = "checkpoints"
            os.makedirs(caminho_salvamento, exist_ok=True)

            caminho_completo = os.path.join(caminho_salvamento, nome_arquivo)

            estado = {
                'epoca': args['num_epocas'],
                'estado_modelo': estado_modelo,
                'estado_otimizador': estado_otimizador,
                'args': args,
                'data_execucao_utc': args['data_execucao']
            }

            torch.save(estado, caminho_completo)
            logging.info(f"Modelo salvo com sucesso em: {caminho_completo}")
            return caminho_completo

        except Exception as e:
            logging.error(f"Erro ao salvar o modelo: {e}")
            return None

    @staticmethod
    def salvar_csv(args, dados):
        try:
            logging.info(f"Salvando informações da época no arquivo CSV")

            caminho_completo = ArquivoUtils.gerar_caminho_do_arquivo(f"resultados_gerais_{args['modelo_base']}_{args['data_execucao']}", "results")

            arquivo_existe = os.path.exists(caminho_completo)
            with open(caminho_completo, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=dados.keys())
                if not arquivo_existe:
                    writer.writeheader()
                writer.writerow(dados)
        except Exception as e:
            logging.error(f"Erro ao salvar o modelo no arquivo csv: {e}")
            return None

    @staticmethod
    def gerar_caminho_do_arquivo(nome_arquivo, diretorio):
        nome_arquivo = f"{nome_arquivo}.csv"
        os.makedirs(diretorio, exist_ok=True)

        return os.path.join(diretorio, nome_arquivo)
    
    @staticmethod
    def salvar_features_imagem(
        nome_tecnica_ext,
        nome_dataset,
        dados_treino,
        classes_treino,
        dados_teste,
        classes_teste):
        
        nome_arquivo = f"{nome_tecnica_ext}_{nome_dataset}_features.npz"
        
        np.savez_compressed(
            nome_arquivo,
            dados_treino=dados_treino,
            classes_treino=classes_treino,
            dados_teste=dados_teste,
            classes_teste=classes_teste,
        )

        logging.info(f"Features salvas em '{nome_arquivo}'")
        logging.info(f"Shape treino: {dados_treino.shape}, Shape teste: {dados_teste.shape}")

    
    @staticmethod
    def carregar_caracteristicas_salvas(caminho_npz: str):
        pack = np.load(caminho_npz, allow_pickle=False)
        dados_treino = pack["dados_treino"].astype(np.float32)
        classes_treino = pack["classes_treino"].astype(str)
        dados_teste = pack["dados_teste"].astype(np.float32)
        classes_teste = pack["classes_teste"].astype(str)
        
        return dados_treino, classes_treino, dados_teste, classes_teste

    @staticmethod
    def transformar_dados_memmap(dados, nome_arquivo):
        N, D = len(dados), dados[0].shape[0]
        dados_memmap = np.memmap(nome_arquivo, dtype=np.float32, mode='w+', shape=(N, D))

        for i, feat in enumerate(dados):
            dados_memmap[i, :] = feat
        dados_memmap.flush()

        return dados_memmap

