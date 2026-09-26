# IA para Detecção e Vetorização de Linhas de Cana-de-Açúcar

Solução completa em Python para treinar, avaliar e executar uma IA capaz de detectar linhas de plantio de cana-de-açúcar em ortomosaicos georreferenciados e transformá-las em linhas vetoriais `LineString`.

Esta IA é baseada em segmentação semântica multissaída (U-Net ou DeepLabv3+ com codificador ResNet), gerando a máscara da faixa de plantio, a máscara da linha central e um mapa de orientação local ($sin(2\theta)$ e $cos(2\theta)$). O pós-processamento reconstrói as linhas convertendo os pixels do esqueleto em grafos topológicos, realizando poda de espículas, ligação inteligente de falhas e ajuste spline paramétrico local.

---

## 1. Instalação e Configuração

Recomendamos usar **Conda** ou **Mamba** para instalar as dependências de geoprocessamento (`GDAL`, `Rasterio`, `GeoPandas`, `Fiona`) de forma consistente.

```bash
# 1. Criar e ativar o ambiente
conda create -n sugarcane_env python=3.10 -y
conda activate sugarcane_env

# 2. Instalar PyTorch com suporte a CUDA (ajuste a versão conforme sua GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Instalar bibliotecas de geoprocessamento via Conda-Forge
conda install -c conda-forge rasterio geopandas shapely pyproj fiona pyogrio -y

# 4. Instalar as demais dependências do projeto
pip install -r requirements.txt
```

---

## 2. Estrutura do Projeto

* `configs/`: Configurações em formato YAML (`base.yaml`, `unet_resnet34.yaml`, `deeplab_resnet50.yaml`).
* `src/data/`: Leitura de rasters/vetores, reprojeção automática de CRS (estimando UTM local) e geração de alvos (faixas, centro, orientações trigonométricas).
* `src/models/` & `src/losses/`: Redes de segmentação e perdas compostas (BCE, Focal, Dice e clDice topológica).
* `src/postprocessing/`: Limpeza morfológica, esqueleto/thinning, conversão de pixels em grafos (NetworkX), poda de espículas, ligação de falhas (gap bridging) e suavização spline paramétrica.
* `src/geospatial/`: Conversão de coordenadas de pixel para mundo métrico, vetorização e exportação.
* `scripts/`: Scripts executáveis de inspeção, preparação, treino, predição e exportação.

---

## 3. Instruções de Uso

### A. Inspecionar Dados de Entrada
Antes de iniciar, verifique se a região de interesse (ROI), o ortomosaico raster e as linhas de referência se intersectam e quais são seus CRS:
```bash
python scripts/inspect_data.py \
  --orthomosaic data/orthomosaic.tif \
  --roi data/roi.gpkg \
  --lines data/rows.gpkg
```

### B. Preparar o Dataset
Gera os tiles sobrepostos e os alvos de treinamento (NPZ) nas pastas de treino/val/teste divididos espacialmente por blocos:
```bash
python scripts/prepare_dataset.py --config configs/unet_resnet34.yaml
```

### C. Treinar a Rede
Inicia o treinamento com controle de mixed precision (AMP), gradient clipping, monitor clDice e early stopping:
```bash
python scripts/train.py --config configs/unet_resnet34.yaml
```

### D. Validar a Rede
Calcula a acurácia e perdas compostas no conjunto de validação:
```bash
python scripts/validate.py \
  --config configs/unet_resnet34.yaml \
  --checkpoint outputs/checkpoints/best.pt
```

### E. Predizer e Vetorizar Ortomosaicos Completos
Executa a inferência por sliding window com janela de fusão Hann, limpa e reconstrói as linhas vetorizadas georreferenciadas salvando as saídas e camadas de depuração no GeoPackage:
```bash
python scripts/predict.py \
  --config configs/unet_resnet34.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --orthomosaic data/novo_ortomosaico.tif \
  --roi data/nova_regiao.gpkg \
  --output outputs/vectors/linhas_preditas.gpkg
```

---

## 4. Resultados e Diagnósticos

Após a execução da predição, as seguintes saídas estarão disponíveis no diretório configurado:
* `summary_report.html`: Página HTML interativa compilando todas as métricas geométricas e de segmentação.
* `overlays.png`: Comparação visual das linhas previstas versus referências sobre a imagem.
* `probabilities.png`: Mapas de probabilidade de faixa e de linha central estimadas pelo modelo.
