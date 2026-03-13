# Protocolo de Execução — App Passos Mágicos
**Modelo Preditivo de Risco de Defasagem**

---

## 1. Visão Geral

Este protocolo descreve os passos necessários para preparar o ambiente, gerar a base de dados e executar o app `app_passos_magicos.py`. O app é uma interface interativa (Streamlit) que carrega automaticamente o arquivo `BASE_PEDE_TRATADA.xlsx` da mesma pasta do script, treina um modelo preditivo (Random Forest) e disponibiliza quatro painéis de análise.

---

## 2. Pré-requisitos

### 2.1 Python
- Versão mínima: **Python 3.10**
- Verificar versão instalada:
  ```bash
  python --version
  ```

### 2.2 Dependências

Instalar todos os pacotes necessários com:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl
```

| Pacote | Finalidade |
|---|---|
| `streamlit` | Interface web do app |
| `pandas` | Manipulação da base de dados |
| `numpy` | Operações numéricas |
| `scikit-learn` | Modelo preditivo (Random Forest) |
| `matplotlib` / `seaborn` | Geração de gráficos |
| `openpyxl` | Leitura de arquivos `.xlsx` |

---

## 3. Estrutura de Arquivos Esperada

Todos os arquivos devem estar na **mesma pasta**:

```
projeto/
├── app_passos_magicos.py       ← script principal do app
├── tratamento_pede.py          ← script de tratamento da base (pré-requisito)
├── BASE_PEDE_TRATADA.xlsx      ← gerado pelo tratamento_pede.py (carregado automaticamente)
└── protocolo.md                ← este documento
```

> **Atenção:** o app busca o arquivo `.xlsx` automaticamente na mesma pasta. Não é necessário fazer upload manual. Se o arquivo não for encontrado, o app exibe uma mensagem de erro com o caminho esperado.

---

## 4. Passo a Passo de Execução

### Passo 1 — Gerar a base tratada

Execute o script de tratamento para gerar o arquivo `BASE_PEDE_TRATADA.xlsx`:

```bash
python tratamento_pede.py
```

- O script deve gerar o arquivo `BASE_PEDE_TRATADA.xlsx` com a aba `BASE_CONSOLIDADA`.
- Confirme que o arquivo foi criado na pasta do projeto antes de prosseguir.

### Passo 2 — Verificar colunas obrigatórias

O app requer as seguintes colunas na base gerada. Qualquer ausência interrompe a execução com mensagem de erro.

**Colunas obrigatórias para o modelo:**

| Coluna | Descrição |
|---|---|
| `fase` | Fase atual do aluno (0 = ALFA, 1–8) |
| `genero_feminino` | Gênero codificado (1 = Feminino, 0 = Masculino) |
| `instituicao_cod` | Instituição de ensino codificada (0–3) |
| `anos_no_programa` | Tempo no programa em anos |
| `iaa` | Índice de Autoavaliação |
| `ieg` | Índice de Engajamento |
| `ips` | Índice Psicossocial |
| `ida` | Índice de Desempenho Acadêmico |
| `ipv` | Índice de Ponto de Virada |
| `nota_matematica` | Nota de Matemática (0–10) |
| `nota_portugues` | Nota de Português (0–10) |
| `nota_ingles` | Nota de Inglês (0 = não avaliado) |
| `media_notas` | Média das notas válidas |
| `media_indicadores` | Média dos indicadores IAA/IEG/IPS/IPV |
| `pedra_ano` | Pedra do ano corrente (1–4, -1 = sem dado) |
| `pedra_2020` | Pedra histórica 2020 (-1 = sem dado) |
| `pedra_2021` | Pedra histórica 2021 (-1 = sem dado) |
| `ian` | Indicador de Adequação de Nível |
| `defasagem` | Defasagem escolar |
| `inde_ano` | INDE do ano de referência |

**Colunas opcionais** (criadas com `NaN` se ausentes): `ra`, `ano_referencia`, `genero`, `ipp`, `pedra_2022`, `pedra_2023`, `inde_2022`, `inde_2023`.

### Passo 3 — Iniciar o app

```bash
streamlit run app_passos_magicos.py
```

O Streamlit abrirá automaticamente o navegador em `http://localhost:8501`. Para acessar manualmente, abra esse endereço.

---

## 5. Funcionamento do App

### Carregamento automático
O app localiza o arquivo `.xlsx` na seguinte ordem de prioridade:
1. `BASE_PEDE_TRATADA.xlsx` (nome padrão)
2. Qualquer outro arquivo `.xlsx` presente na pasta

A barra lateral exibe o nome do arquivo carregado. Caso nenhum arquivo seja encontrado, o app exibe o caminho onde o arquivo deve ser colocado e interrompe a execução.

### Painéis disponíveis

| Painel | Descrição |
|---|---|
| 📊 **Visão Geral** | KPIs gerais, distribuição IAN, evolução INDE, distribuição de risco por fase e pedra |
| 🔍 **Análise por Indicador** | Análise detalhada de cada indicador (IAA, IEG, IPS, IDA, IPV, IPP) com gráficos comparativos |
| 🤖 **Modelo Preditivo** | Desempenho do modelo (AUC-ROC, curva ROC, importância de features, ranking dos alunos em maior risco) |
| 🧑‍🎓 **Predição Individual** | Formulário para inserir dados de um aluno e obter a probabilidade estimada de risco, com comparação à média da base |

### Classificação de Risco

| Faixa de probabilidade | Classificação |
|---|---|
| < 30% | 🟢 Baixo Risco |
| 30% – 59% | 🟡 Risco Moderado |
| ≥ 60% | 🔴 Alto Risco |

---

## 6. Solução de Problemas

| Sintoma | Causa provável | Solução |
|---|---|---|
| App exibe erro de arquivo não encontrado | `.xlsx` ausente ou em pasta errada | Mover `BASE_PEDE_TRATADA.xlsx` para a mesma pasta do script |
| Erro de colunas obrigatórias ausentes | Base gerada incorretamente | Reexecutar `tratamento_pede.py` e verificar saída |
| Aba `BASE_CONSOLIDADA` não encontrada | Arquivo gerado com nome de aba diferente | Reexecutar `tratamento_pede.py`; o app usará a primeira aba disponível com aviso |
| `ModuleNotFoundError` | Dependência não instalada | Executar `pip install <pacote>` conforme seção 2.2 |
| Porta 8501 ocupada | Outro processo usando a porta | Executar com `streamlit run app_passos_magicos.py --server.port 8502` |
| Modelo com AUC baixo | Base com poucos registros ou dados desbalanceados | Verificar qualidade da base; o modelo usa `class_weight="balanced"` automaticamente |

---

## 7. Encerramento

Para encerrar o app, pressione `Ctrl + C` no terminal onde o Streamlit está rodando.

---

*Protocolo gerado para o projeto Datathon Passos Mágicos.*