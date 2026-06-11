# Diário de experimentos

Registro das execuções, achados e decisões metodológicas. Insumo para o relatório final.

## Configuração comum

- Modelo: YOLO11n pré-treinado no COCO (Ultralytics 8.4.65, torch 2.12.0+cu132).
- Dataset: subset do WIDER Face gerado por `src/prepare_dataset.py` com seed 42 - 1200 treino / 300 validação / 300 teste (teste sorteado da partição de validação oficial do WIDER, nunca usada no treino).
- Imagens com mais de 80 faces excluídas (`--max-faces 80`) ver "Decisões metodológicas".
- Treinos: `imgsz=640`, `batch=8`, `workers=2`, `seed=42`, 20 épocas.
- Hardware: GPU NVIDIA RTX 3060 12 GB (local), 16 GB RAM.
- Métricas sempre medidas no conjunto de **teste** com o `best.pt` selecionado pela validação; a validação é usada apenas para seleção de checkpoint.

## Resultados no conjunto de teste

| Experimento | Configuração | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|---|
| exp1_baseline_pretreinado | COCO puro, sem treino, classe `person` | 0.0009 | 0.0001 | 0.0054 | 0.0522 |
| exp2_feature_extraction | freeze=10, optimizer auto (AdamW, lr ~2e-3) | 0.7101 | 0.3987 | 0.8426 | 0.6312 |
| exp3_fine_tuning | rede inteira, AdamW, lr0=1e-4 | 0.6816 | 0.3759 | 0.8455 | 0.5896 |
| exp4_fine_tuning_lr1e3 | rede inteira, AdamW, lr0=1e-3 | 0.7193 | 0.4139 | 0.8652 | 0.6392 |
| exp5_fine_tuning_lr1e3_50ep | rede inteira, AdamW, lr0=1e-3, 50 épocas | **0.7415** | **0.4313** | **0.8777** | **0.6553** |

Fonte: `results/metrics.csv` (inclui data/hora de cada run).

## Achados

### 1. Transfer learning é indispensável para a tarefa

O baseline (COCO sem treino) marca mAP@0.5 ≈ 0.001: o COCO não tem classe de rosto, e a classe mais próxima (`person`, corpo inteiro) quase nunca atinge IoU ≥ 0.5 com caixas de face. Vinte épocas de treino elevam o mAP@0.5 para ~0.7 - o resultado central do projeto.

### 2. Reprodutibilidade verificada na prática

O exp2 foi executado duas vezes (em momentos diferentes do dia) e produziu métricas idênticas até a 4ª casa decimal (mAP50 0.7101, precision 0.8426, recall 0.6312), confirmando que a combinação seed fixa no sorteio do subset + `seed=42` e `deterministic` no treino torna o pipeline reprodutível.

### 3. Feature extraction superou o fine-tuning completo - diagnóstico: subtreino

Contra a expectativa ingênua (mais parâmetros treináveis = melhor resultado), o exp3 ficou abaixo do exp2 em mAP@0.5 (0.682 vs 0.710) e principalmente em recall (0.590 vs 0.631), com precision empatada (~0.84).

Evidências das curvas de treino (`models/*/results.csv`):

- Ambos os experimentos atingiram o melhor mAP de validação **na época 20 (a última)** - nenhum havia parado de melhorar quando o orçamento acabou.
- No exp3, o mAP50 de validação ainda subia ao final (0.6760 → 0.6816 → 0.6830 → 0.6855 nas épocas 17–20).
- A classification loss de validação do exp3 terminou em 0.927 contra 0.753 do exp2: com lr0=1e-4, a cabeça de classificação não teve tempo de convergir.
- As perdas de validação caem monotonicamente nos dois casos: **não há overfitting em nenhum dos experimentos** neste orçamento - o subset pequeno não chegou a ser decorado em 20 épocas.

Interpretação: o exp2 treina as camadas descongeladas com lr ~2e-3 (escolhida pelo otimizador automático), 20× maior que a lr do exp3. Em 20 épocas, mover a rede inteira lentamente rende menos que mover rápido apenas as camadas que precisam se especializar.

## Hipóteses em teste

- **H1 (subtreino):** o exp3 perdeu porque lr0=1e-4 é baixa demais para 20 épocas - com lr maior, o fine-tuning completo alcançaria ou superaria o feature extraction.
- **H2 (regularização/suficiência do backbone):** as features do COCO já são adequadas para faces e o congelamento atua como regularizador no subset pequeno - o fine-tuning completo não teria vantagem mesmo com lr adequada, podendo até degradar as features (esquecimento catastrófico) com lr alta.

## Experimento 4 - fine-tuning completo com lr0=1e-3

Desenho: idêntico ao exp3 em tudo (rede inteira, AdamW, 20 épocas, batch 8, seed 42), mudando **apenas** `lr0` de 1e-4 para 1e-3. Variável única para isolar o efeito da learning rate.

Previsões discriminantes:

- Se **H1** estiver certa: exp4 deve igualar ou superar o exp2 (mAP@0.5 ≥ ~0.71).
- Se **H2** dominar: exp4 deve ficar próximo do exp3 ou abaixo dele (lr maior degradando os pesos pré-treinados sem ganho compensatório).

Resultado (teste): mAP@0.5 **0.7193** | mAP@0.5:0.95 **0.4139** | precision **0.8652** | recall **0.6392** - o melhor experimento em todas as métricas.

### Veredito: H1 confirmada

- A única mudança em relação ao exp3 foi a learning rate (1e-4 → 1e-3), e o fine-tuning completo saltou de último (mAP@0.5 0.682) para primeiro (0.719): o resultado fraco do exp3 era **subtreino**, não limitação da estratégia.
- O exp4 superou o exp2 em todas as métricas, com o maior ganho em mAP@0.5:0.95 (0.414 vs 0.399): especializar o backbone melhora sobretudo a **qualidade de localização** das caixas, não apenas a taxa de acerto.
- Curvas (`models/exp4_fine_tuning_lr1e3/results.csv`): melhor época 19 (mAP50 de validação 0.7364), perdas de validação caindo até o fim - ainda **sem overfitting**; mais épocas possivelmente trariam ganho adicional (trabalho futuro).
- H2 rejeitada na sua versão forte: lr 10× maior não degradou os pesos pré-treinados (sem sinal de esquecimento catastrófico neste orçamento).
- **Conclusão para a aplicação:** o exp4 é o modelo recomendado para a plataforma - maior recall (métrica crítica: rosto não detectado = dado pessoal exposto) e maior precision simultaneamente.

Reprodução: notebook `04_ajuste_lr_e_epocas.ipynb` (executa os experimentos 4 e 5 em sequência).

## Experimento 5 - fine-tuning lr0=1e-3 com 50 épocas

Desenho: idêntico ao exp4 em tudo, mudando **apenas** o orçamento de épocas (20 → 50). Importante: não é o exp4 "continuado" - a learning rate decai linearmente em função do total de épocas, então é um run independente.

Motivação: as curvas de todos os treinos anteriores terminaram com a validação ainda melhorando e sem qualquer sinal de overfitting - o orçamento de 20 épocas pode ser o gargalo.

Previsões:

- Se a validação continuar subindo e estabilizar: o orçamento era o limite; o exp5 vira o modelo final.
- Se a validação atingir pico e cair com a perda de treino ainda melhorando: observamos o início de overfitting em primeira mão (material direto para a discussão pedida na proposta); o `best.pt` protege o resultado.

Resultado (teste): mAP@0.5 **0.7415** | mAP@0.5:0.95 **0.4313** | precision **0.8777** | recall **0.6553** - novo melhor em todas as métricas; **modelo final do projeto**.

### Veredito: o orçamento era o gargalo, e a saturação foi observada

- Confirmou-se a primeira previsão: a validação continuou subindo até saturar. O mAP50 de validação foi de 0.709 (época 20) a um platô de ~0.75–0.76 a partir da época ~40, com melhor época em 49 (0.7593).
- No platô aparece o primeiro sinal de divergência treino/validação: a perda de classificação de treino segue caindo (0.771 na época 40 → 0.680 na 49) enquanto a de validação estabiliza (~0.71) - o início do território de overfitting, ainda **sem degradação** da validação. Treinar além de ~50 épocas tenderia a retorno decrescente com risco crescente.
- Ganho sobre o exp4 (+2.2 p.p. de mAP@0.5, +1.6 p.p. de recall) com o mesmo custo por época - apenas mais orçamento.
- Discussão de overfitting do relatório fica completa com evidência própria: nenhum overfitting até 20 épocas (exps 2–4), saturação com divergência incipiente em ~40–50 (exp5), `best.pt` da validação como proteção.

## Análise do ponto de operação (limiar de confiança)

O mAP resume a qualidade média do detector, mas a implantação exige escolher um **limiar de confiança**. Curvas precision×confiança e recall×confiança do modelo final (exp5, conjunto de teste):

| conf | precision | recall | F1 |
|---|---|---|---|
| 0.05 | 0.480 | 0.759 | 0.588 |
| 0.10 | 0.639 | 0.730 | 0.681 |
| 0.25 | 0.840 | 0.672 | 0.747 |
| 0.50 | 0.956 | 0.586 | 0.727 |
| 0.282 (F1 máximo) | 0.871 | 0.663 | 0.753 |

Leitura para a aplicação de anonimização:

- Os dois tipos de erro têm **custos assimétricos**: rosto não detectado = dado pessoal exposto (grave); região borrada sem rosto = perda estética (leve).
- O ponto de F1 máximo (conf ≈ 0.28) é o ótimo estatístico, mas o ponto **errado** para esta aplicação, pois trata os dois erros como equivalentes.
- Operando em conf 0.05–0.10, o recall sobe de ~0.67 para 0.73–0.76 - cada ponto percentual significa rostos a mais efetivamente anonimizados - ao custo de mais borrões desnecessários.
- Recomendação: limiar de operação **0.10** (recall 0.730 com precision ainda razoável de 0.639), justificada pela assimetria de custos. Registrar na avaliação ética do relatório.

A análise é reproduzível na seção 7 do notebook `04_ajuste_lr_e_epocas.ipynb`.

## Decisões metodológicas

- **Teste derivado da partição de validação oficial do WIDER Face**: o teste oficial não tem anotações públicas; usar a partição de validação (nunca amostrada para treino/validação internos) elimina vazamento de dados.
- **Exclusão de imagens com mais de 80 faces** (`--max-faces 80`): a memória do TaskAlignedAssigner cresce com `batch × nº máximo de faces por imagem × nº de anchors`, e o mosaic multiplica o nº de faces por até 4; cenas de multidão (centenas de faces minúsculas) estouravam VRAM e RAM com batch 16. Excluir (e não truncar - truncar deixaria faces sem anotação, virando supervisão ruidosa) também é coerente com o escopo da aplicação: plataformas cívicas recebem fotos de problemas urbanos, não multidões. Registrar como limitação.
- **`batch=8` e `workers=2`** pelos mesmos limites de memória (12 GB de VRAM compartilhada com o desktop; 16 GB de RAM).
- **`optimizer='AdamW'` explícito nos experimentos com lr0 controlada**: com `optimizer='auto'` (padrão), a Ultralytics ignora o `lr0` informado e escolhe a taxa sozinha - sem fixar o otimizador, o exp3/exp4 não testariam o que dizem testar.
- **Imagens sem nenhuma caixa válida são excluídas** na conversão das anotações; o modelo não vê imagens de fundo no treino (limitação menor a registrar).
