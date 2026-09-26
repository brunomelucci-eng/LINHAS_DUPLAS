# Pacote portatil - Linhas de cana

Este pacote contém o código de inferência e pós-processamento, a configuração
global adaptativa, as dependências e o último modelo treinado com 12 imagens.
O produto operacional bloqueia linhas que ainda apresentem serrilhado.

## Requisitos

- Windows 10 ou 11 de 64 bits.
- Python 3.12 de 64 bits instalado com a opção `Add Python to PATH`.
- Aproximadamente 8 GB livres para instalação e processamento temporário.
- Para o modo CUDA: GPU NVIDIA e driver atualizado. Não é necessário instalar
  o CUDA Toolkit separadamente; o PyTorch traz as bibliotecas CUDA usadas.

## Instalação

1. Extraia todo o ZIP para uma pasta local, por exemplo `D:\linhas_cana`.
2. Abra o PowerShell nessa pasta.
3. Se o Windows bloquear scripts apenas nesta janela, execute:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

4. Para computador com GPU NVIDIA:

```powershell
.\INSTALAR_WINDOWS.ps1 -Modo CUDA
```

Para computador sem GPU NVIDIA:

```powershell
.\INSTALAR_WINDOWS.ps1 -Modo CPU
```

O modo CPU funciona, mas a inferência de ortomosaicos grandes é muito mais lenta.

## Verificação

O instalador já executa a verificação. Para repeti-la:

```powershell
.\.venv\Scripts\python.exe .\verificar_instalacao.py
```

O resultado deve terminar com `INSTALACAO_OK` e mostrar o threshold calibrado
`0.38`.

## Executar uma inferência completa

```powershell
.\EXECUTAR_INFERENCIA.ps1 `
  -Imagem "D:\dados\imagem.gpkg" `
  -Talhao "D:\dados\talhao.geojson"
```

Para escolher também o alias operacional de saída:

```powershell
.\EXECUTAR_INFERENCIA.ps1 `
  -Imagem "D:\dados\imagem.gpkg" `
  -Talhao "D:\dados\talhao.geojson" `
  -Saida "D:\resultados\linhas_producao.gpkg"
```

O comando executa inferência, limpeza, skeleton, vetorização, refinamento de
linhas duplas, suavização e auditoria final. Não reutilize um caminho de saída
já existente; cada execução cria um nome próprio quando `-Saida` é omitido.

## Arquivos de saída

- O caminho informado em `-Saida` contém somente `predicted_rows`, isto é, o
  produto operacional aprovado e sem serrilhado detectado.
- `outputs\runs\<run_id>\vectors\...__pos_inferencia.gpkg` é o vetor bruto.
- `outputs\runs\<run_id>\vectors\...__pos_processamento.gpkg` contém:
  - `final_lines`: todas as linhas para auditoria;
  - `approved_lines`: linhas aprovadas;
  - `production_lines`: produto operacional sem serrilhado;
  - `manual_review`: linhas que precisam de revisão;
  - demais camadas de reparo e rejeição.
- `outputs\runs\<run_id>\rasters\probability_center.tif` é o mapa de
  probabilidade central usado pelo pós-processamento.

## Reprocessar sem repetir a rede neural

Quando já existirem o vetor bruto e `probability_center.tif`:

```powershell
.\EXECUTAR_POS_PROCESSAMENTO.ps1 `
  -VetoresBrutos "D:\resultado\talhao__pos_inferencia.gpkg" `
  -Probabilidade "D:\resultado\probability_center.tif" `
  -Talhao "D:\dados\talhao.geojson" `
  -Saida "D:\resultado\talhao_reprocessado.gpkg"
```

## Modelo incluído

- `modelos\best_23_08_2026_12_imagens_run_001.pt`
- `modelos\best_23_08_2026_12_imagens_run_001.pt.threshold.json`

Os dois arquivos devem permanecer juntos. Renomear ou separar a calibração faz
a inferência falhar de propósito, evitando usar um threshold não validado.
