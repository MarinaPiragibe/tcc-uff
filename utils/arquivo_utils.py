import logging
import os
import csv
import h5py
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
    def salvar_csv(args, dados, nome_arquivo=None):
        try:
            logging.info(f"Salvando informações no arquivo CSV")
            
            caminho_completo = ArquivoUtils.gerar_caminho_do_arquivo(nome_arquivo if nome_arquivo else f"resultados_gerais_{args['modelo_base']}_{args['data_execucao']}", "results")

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
        
        nome_arquivo = f"features/{nome_tecnica_ext}_{nome_dataset}_features.npz"
        
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
    def salvar_features_vgg16_h5(
        nome_dataset,
        dados_treino,
        classes_treino,
        dados_teste,
        classes_teste
    ):

        nome_arquivo = f"vgg16_{nome_dataset}_features.h5"
        logging.info(f"Salvando features VGG16 em '{nome_arquivo}'...")

        with h5py.File(nome_arquivo, "w") as f:
            # Dataset de treino
            f.create_dataset("dados_treino", data=dados_treino, dtype=np.float32)
            f.create_dataset("classes_treino", data=classes_treino.astype(np.int64))
            # Dataset de teste
            f.create_dataset("dados_teste", data=dados_teste, dtype=np.float32)
            f.create_dataset("classes_teste", data=classes_teste.astype(np.int64))

        logging.info(f"Features salvas com sucesso!")
        logging.info(f"Shape treino: {dados_treino.shape}, Shape teste: {dados_teste.shape}")


    @staticmethod
    def carregar_features_vgg16_h5(caminho_h5: str):
        """
        Carrega as features da VGG16 salvas em HDF5 e converte as classes para str,
        mantendo compatibilidade com o formato .npz antigo.
        """
        logging.info(f"Carregando features de '{caminho_h5}'...")

        with h5py.File(caminho_h5, "r") as f:
            dados_treino = np.array(f["dados_treino"], dtype=np.float32)
            dados_teste = np.array(f["dados_teste"], dtype=np.float32)

            # Converte as classes para str
            classes_treino = np.array(f["classes_treino"], dtype=np.int64).astype(str)
            classes_teste = np.array(f["classes_teste"], dtype=np.int64).astype(str)

        logging.info(f"Features carregadas com sucesso!")
        logging.info(f"Shape treino: {dados_treino.shape}, Shape teste: {dados_teste.shape}")

        return dados_treino, classes_treino, dados_teste, classes_teste
    
    @staticmethod
    def carregar_caracteristicas_salvas(caminho_npz: str):
        pack = np.load(caminho_npz, mmap_mode="r")
        dados_treino = pack["dados_treino"]
        classes_treino = pack["classes_treino"]
        dados_teste = pack["dados_teste"]
        classes_teste = pack["classes_teste"]
        
        return dados_treino, classes_treino, dados_teste, classes_teste

    @staticmethod
    def transformar_dados_memmap(dados, nome_arquivo):
        N, D = len(dados), dados[0].shape[0]
        dados_memmap = np.memmap(nome_arquivo, dtype=np.float32, mode='w+', shape=(N, D))

        for i, feat in enumerate(dados):
            dados_memmap[i, :] = feat
        dados_memmap.flush()

        return dados_memmap

