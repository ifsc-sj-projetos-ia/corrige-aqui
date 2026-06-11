# Dados

O dataset **WIDER Face** não é versionado neste repositório (é grande e possui termos próprios de uso). Para obtê-lo e gerar o subset usado nos experimentos, rode a partir da raiz do repositório:

```bash
python src/prepare_dataset.py --train-size 1500 --test-size 300 --seed 42
```

O script:

1. Baixa `WIDER_train.zip`, `WIDER_val.zip` e `wider_face_split.zip` (espelho oficial no Hugging Face: [CUHK-CSE/wider_face](https://huggingface.co/datasets/CUHK-CSE/wider_face)) para `data/raw/`.
2. Converte as anotações (formato `x y w h` em pixels) para o formato YOLO (`classe cx cy w h` normalizados entre 0 e 1).
3. Sorteia um subset reprodutível (seed fixa) e gera os splits em `data/wider_subset/`:
   - `train/` e `val/` sorteados do conjunto de treino original do WIDER Face
   - `test/` sorteado do conjunto de validação original (o teste oficial do WIDER Face não tem anotações públicas)
4. Gera `data/wider_subset/dataset.yaml`, o arquivo de configuração lido pela Ultralytics.

Estrutura resultante:

```
data/
├── raw/                  # Zips e arquivos extraídos do WIDER Face
└── wider_subset/
    ├── dataset.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

Parâmetros úteis do script: `--train-size` (nº de imagens de treino+validação), `--val-fraction` (fração destinada à validação, padrão 0.2), `--test-size`, `--max-faces` (exclui imagens com mais faces que o limite, padrão 80, e controla o custo de memória do treino e remove cenas de multidão fora do escopo da aplicação), `--seed` e `--skip-download` (se os zips já foram baixados).

Fonte original do dataset: Yang, S.; Luo, P.; Loy, C. C.; Tang, X. **WIDER FACE: A Face Detection Benchmark**. CVPR, 2016. http://shuoyang1213.me/WIDERFACE/
