# Instruções para Execução em Outro Computador

Este pacote contém os arquivos complementares necessários para rodar a inferência no outro computador que **não estão versionados no Git**:
1. O **último modelo treinado** (`best_rowgraphnet_16_imagens.pt`) e sua calibração (`.threshold.json`).
2. A configuração de inferência do RowGraphNet (`configs/inference_rowgraphnet.yaml`).
3. O script de execução em lote / fila (`EXECUTAR_FILA.ps1`).
4. O script atualizado de inferência (`EXECUTAR_INFERENCIA.ps1`).

---

## 1. Passo a Passo no Outro Computador

### Passo A: Clonar ou Atualizar o Repositório Git
No outro computador, clone ou atualize o repositório oficial:
```powershell
git clone https://github.com/brunomelucci-eng/LINHAS_DUPLAS.git
cd LINHAS_DUPLAS
```

### Passo B: Extrair este Pacote ZIP
Extraia o conteúdo do `PACOTE_INFERENCIA_ROWGRAPHNET.zip` diretamente na **raiz** da pasta `LINHAS_DUPLAS`.
Os arquivos serão posicionados automaticamente nas subpastas corretas:
- `modelos/best_rowgraphnet_16_imagens.pt`
- `modelos/best_rowgraphnet_16_imagens.pt.threshold.json`
- `configs/inference_rowgraphnet.yaml`
- `EXECUTAR_FILA.ps1`
- `EXECUTAR_INFERENCIA.ps1`

### Passo C: Instalar o Ambiente (se ainda não instalado)
Execute:
```powershell
powershell -ExecutionPolicy Bypass -File .\INSTALAR_WINDOWS.ps1
```

---

## 2. Como Executar as Inferências

### Opção 1: Executar em Fila (Batch)
Edite a lista `$Jobs` no arquivo `EXECUTAR_FILA.ps1` com os caminhos das imagens e talhões do seu computador e execute:
```powershell
powershell -ExecutionPolicy Bypass -File .\EXECUTAR_FILA.ps1
```

### Opção 2: Executar uma Área Individual
```powershell
powershell -ExecutionPolicy Bypass -File .\EXECUTAR_INFERENCIA.ps1 -Imagem "C:\caminho\imagem.gpkg" -Talhao "C:\caminho\talhao.geojson" -Saida "C:\caminho\resultado.gpkg"
```

---

## 3. Especificações do Modelo
- **Modelo:** `best_rowgraphnet_16_imagens.pt` (Epoch 55, Dice 0.767)
- **Arquitetura:** RowGraphNet (mit_b2 + MultiTaskDecoder + ConnectivityRefiner)
- **Limiar de Centro Calibrado:** 0.34
