# Detecção de Faces para Anonimização de Imagens

Projeto final da disciplina de Introdução à Ciência de Dados e Aprendizado de Máquina (IFSC).

## Objetivo

Detectar rostos humanos em imagens usando deep learning (YOLO11n), com o propósito de anonimização: em plataformas que recebem fotos enviadas por cidadãos (por exemplo, registros de problemas urbanos), rostos de terceiros precisam ser borrados antes do armazenamento, em conformidade com a LGPD (Lei Geral de Proteção de Dados, Lei nº 13.709/2018).

> Este projeto realiza apenas detecção de faces (localização de bounding boxes). Ele não faz reconhecimento, identificação ou verificação de pessoas.

## Estrutura do repositório

```
├── data/         # Dataset 
├── docs/         # Documentação e relatório
├── models/       # Pesos do modelo final (best.pt) e artefatos de treino
├── notebooks/    # Notebooks dos experimentos (Google Colab)
├── results/      # Métricas dos experimentos em CSV
└── src/          # Scripts (preparação do dataset e demo de anonimização)
```

## Dataset

[WIDER Face](http://shuoyang1213.me/WIDERFACE/) - benchmark padrão para detecção de faces, com 32.203 imagens e 393.703 faces anotadas, alta variação de escala, pose e oclusão.

Como o dataset completo é grande demais para o Colab gratuito, o script `src/prepare_dataset.py` baixa o WIDER Face, converte as anotações para o formato YOLO e gera um **subset** com splits de treino, validação e teste. 

## Dependências

- Python 3.10+
- [ultralytics](https://docs.ultralytics.com/) (YOLO11)
- opencv-python
- matplotlib
- pandas

### No Google Colab

Nenhuma instalação manual: a primeira célula de cada notebook instala o que falta. Basta selecionar o ambiente de execução com GPU (T4).

### Em máquina local (GPU NVIDIA)

Requisitos: driver NVIDIA atualizado (verifique com `nvidia-smi`) e ~15 GB livres em disco (dataset + ambiente).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
```

**Atenção:** instale o torch do índice CUDA **antes** do `requirements.txt` já que no Windows, a wheel padrão do PyPI é CPU-only e os treinos ficariam inviáveis. Escolha o índice (`cu126`, `cu130`, `cu132`...) compatível com a versão CUDA do seu driver. Para rodar os notebooks, selecione o kernel do `.venv` no VS Code/Jupyter.

Os resultados reportados foram obtidos em uma RTX 3060 (12 GB); com menos VRAM, reduza `BATCH` na célula de parâmetros dos notebooks. Sem GPU, use o Colab. As versões exatas usadas nos experimentos estão registradas em `docs/experimentos.md`.

## Como executar

### 1. Preparar o dataset

```bash
python src/prepare_dataset.py --train-size 1500 --test-size 300
```

O script baixa o WIDER Face (treino + validação + anotações), converte as anotações para o formato YOLO e gera o subset em `data/wider_subset/`, junto com o arquivo `data/wider_subset/dataset.yaml` usado pela Ultralytics.

### 2. Rodar os experimentos

Os notebooks em `notebooks/` foram pensados para o Google Colab (com GPU T4 gratuita) e instalam as próprias dependências na primeira célula:

| Notebook | Experimento |
|---|---|
| `01_baseline.ipynb` | Baseline: YOLO11n pré-treinado (COCO), sem treino, avaliado no conjunto de teste |
| `02_feature_extraction.ipynb` | Transfer learning com feature extraction (`freeze=10`, treina só a cabeça de detecção) |
| `03_fine_tuning.ipynb` | Fine-tuning completo com learning rate baixa (`lr0=1e-4`) |
| `04_ajuste_lr_e_epocas.ipynb` | Experimentos 4 e 5: fine-tuning com `lr0=1e-3` (20 e 50 épocas) e análise do ponto de operação |

Cada notebook salva as métricas do run (mAP@0.5, precision, recall) em um CSV em `results/`, permitindo montar a tabela comparativa do relatório.

### 3. Anonimizar imagens (demo)

Os pesos do modelo final (experimento 5, fine-tuning com `lr0=1e-3` e 50 épocas) estão versionados em `models/best.pt`. O script aceita uma imagem ou uma pasta e salva as versões com rostos borrados em `results/anonimizadas/`:

```bash
python src/anonymize.py caminho/para/imagem_ou_pasta
```

O limiar de confiança padrão é `--conf 0.10`, o ponto de operação recomendado para a aplicação: como rosto não detectado significa dado pessoal exposto, prioriza-se recall sobre precision (justificativa em `docs/experimentos.md`, seção "Análise do ponto de operação"). Outras opções: `--weights` e `--out-dir`. A mesma demonstração existe em forma de notebook na seção 9 do `03_fine_tuning.ipynb`.

## Como reproduzir os resultados

1. Clone o repositório.
2. Rode `python src/prepare_dataset.py --seed 42` (a seed fixa garante o mesmo subset e os mesmos splits).
3. Execute os notebooks na ordem (01 → 02 → 03 → 04).
4. As métricas consolidadas ficam em `results/metrics.csv`.

## Métricas principais

Resultados no conjunto de teste (300 imagens, seed 42), treinos de 20 épocas:

| Experimento | mAP@0.5 | Precision | Recall |
|---|---|---|---|
| Baseline (pré-treinado) | 0.0009 | 0.0054 | 0.0522 |
| Feature extraction (freeze=10, 20 épocas) | 0.7101 | 0.8426 | 0.6312 |
| Fine-tuning completo (lr0=1e-4, 20 épocas) | 0.6816 | 0.8455 | 0.5896 |
| Fine-tuning completo (lr0=1e-3, 20 épocas) | 0.7193 | 0.8652 | 0.6392 |
| Fine-tuning completo (lr0=1e-3, 50 épocas) | **0.7415** | **0.8777** | **0.6553** |

O modelo pré-treinado no COCO não possui classe de rosto (avalia-se a classe `person`), o que explica o baseline próximo de zero e evidencia a necessidade do transfer learning. Com `lr0=1e-4`, o fine-tuning completo ficou abaixo do feature extraction por subtreino; os experimentos 4 e 5, variando uma variável por vez (learning rate e depois orçamento de épocas), confirmaram o diagnóstico e produziram o modelo final, que satura por volta da época 40. A análise detalhada, com evidências das curvas de treinamento e a escolha do limiar de operação para a anonimização, está em `docs/experimentos.md`; a tabela completa (incluindo mAP@0.5:0.95 e data/hora de cada run) está em `results/metrics.csv`.

## Limitações

- Subset reduzido do WIDER Face por restrição de computação (Colab gratuito).
- Possíveis vieses do dataset (distribuição demográfica, condições de captura).
- Faces muito pequenas, ocluídas ou borradas tendem a ser mais difíceis de detectar e falsos negativos significam rostos não anonimizados.

## Considerações éticas

- O sistema detecta faces, nunca identifica pessoas.
- O caso de uso é proteção de privacidade (borramento antes do armazenamento), não vigilância.
- O dataset não é redistribuído neste repositório; apenas instruções de download são fornecidas.

## Relatório

O relatório final (artigo no formato do template IFSC/ABNT) está em [`docs/Artigo ICD - Davi Souza e Caio Aguiar.pdf`](docs/Artigo ICD - Davi Souza e Caio Aguiar.pdf) (versão em PDF) e [`docs/Artigo ICD - Davi Souza e Caio Aguiar.docx`](docs/Artigo ICD - Davi Souza e Caio Aguiar.docx) (versão formatada para entrega). 

## Licença

[MIT](LICENSE)
