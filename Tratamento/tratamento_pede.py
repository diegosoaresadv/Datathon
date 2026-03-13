"""
=============================================================================
TRATAMENTO DE DADOS — BASE PEDE 2022 / 2023 / 2024
Passos Mágicos | Datathon
=============================================================================
Como usar:
  1. Coloque este script na mesma pasta que o arquivo Excel original.
  2. Execute: python tratamento_pede.py
  3. O arquivo BASE_PEDE_TRATADA.xlsx será gerado na mesma pasta.
  4. Esse arquivo é o input do app Streamlit.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO — ajuste os caminhos se necessário
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
INPUT_FILE  = BASE_DIR / "BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx"
OUTPUT_FILE = BASE_DIR / "BASE_PEDE_TRATADA.xlsx"


# ---------------------------------------------------------------------------
# UTILITÁRIOS
# ---------------------------------------------------------------------------
def secao(titulo):
    print(f"\n{'='*60}\n{titulo}\n{'='*60}")


def drop_fully_null_cols(df, label):
    """Remove colunas 100% nulas."""
    cols_nulas = [c for c in df.columns if df[c].isna().all()]
    if cols_nulas:
        print(f"  [{label}] Removidas colunas 100% nulas: {cols_nulas}")
    return df.drop(columns=cols_nulas)


def impute_by_group(df, col, group_col="fase", strategy="median"):
    """Imputa coluna usando mediana ou moda por grupo, depois global."""
    if col not in df.columns:
        return df
    if strategy == "median":
        df[col] = df.groupby(group_col)[col].transform(
            lambda x: x.fillna(x.median())
        )
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        df[col] = df.groupby(group_col)[col].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )
        if df[col].isna().any():
            mode_global = df[col].mode()
            if not mode_global.empty:
                df[col] = df[col].fillna(mode_global.iloc[0])
    return df


def extrair_fase(valor):
    """Converte valores como '1A', 'ALFA', 'FASE 3', 9 para int."""
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip().upper()
    if s in ("ALFA", "0"):
        return 0
    if s.startswith("FASE"):
        try:
            return int(s.split()[1])
        except (IndexError, ValueError):
            return np.nan
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            break
    try:
        return int(digits) if digits else np.nan
    except ValueError:
        return np.nan


def normalizar_instituicao(valor):
    """Padroniza os valores da coluna instituição."""
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().lower()
    if "concluiu" in v:
        return "Concluiu 3º EM"
    if any(k in v for k in ["privada", "empresa parceira", "bolsa 100", "apadrinhamento"]):
        return "Privada"
    if any(k in v for k in ["pública", "publica", "escola pública", "rede decisão", "escola jp ii"]):
        return "Pública"
    if any(k in v for k in ["formado", "bolsista universitário"]):
        return "Universitário Formado"
    if "nenhuma" in v:
        return "Nenhuma"
    return str(valor).strip()


def calcular_ipp_reverso(row):
    """
    Infere o IPP pela fórmula reversa do INDE (válida para Fases 0–7):
      INDE = IAN*0.1 + IDA*0.2 + IEG*0.2 + IAA*0.1 + IPS*0.1 + IPP*0.1 + IPV*0.2
      IPP  = (INDE - soma_demais) / 0.1
    """
    fase = row.get("fase")
    inde = row.get("inde_ano")
    campos = [row.get(c) for c in ["ian", "ida", "ieg", "iaa", "ips", "ipv"]]

    if pd.isna(fase) or fase > 7:
        return np.nan
    if any(pd.isna(v) for v in [inde] + campos):
        return np.nan

    ian, ida, ieg, iaa, ips, ipv = campos
    soma_outros = ian * 0.1 + ida * 0.2 + ieg * 0.2 + iaa * 0.1 + ips * 0.1 + ipv * 0.2
    return float(np.clip(round((inde - soma_outros) / 0.1, 1), 0.0, 10.0))


def add_features(df, ano):
    """Engenharia de features para modelagem."""
    # Anos no programa
    df["anos_no_programa"] = (ano - df["ano_ingresso"]).clip(lower=0)

    # Média das notas (exclui zeros — inglês não avaliado)
    notas = [c for c in ["nota_matematica", "nota_portugues", "nota_ingles"] if c in df.columns]
    df["media_notas"] = df[notas].replace(0, np.nan).mean(axis=1)

    # Média dos indicadores socioeducacionais
    indicadores = [c for c in ["iaa", "ieg", "ips", "ipp"] if c in df.columns]
    df["media_indicadores"] = df[indicadores].mean(axis=1)

    # Evolução ordinal da pedra em relação ao ano anterior
    if ano == 2023 and "pedra_2022" in df.columns and "pedra_ano" in df.columns:
        df["evolucao_pedra"] = (
            df["pedra_ano"].replace(-1, np.nan) - df["pedra_2022"].replace(-1, np.nan)
        )
    elif ano == 2024 and "pedra_2023" in df.columns and "pedra_ano" in df.columns:
        df["evolucao_pedra"] = (
            df["pedra_ano"].replace(-1, np.nan) - df["pedra_2023"].replace(-1, np.nan)
        )
    else:
        df["evolucao_pedra"] = np.nan

    # Encoding gênero
    df["genero_feminino"] = (df["genero"] == "Feminino").astype(int)

    # Encoding ordinal instituição
    INST_MAP = {
        "Pública": 0,
        "Privada": 1,
        "Concluiu 3º EM": 2,
        "Universitário Formado": 3,
        "Nenhuma": -1,
    }
    df["instituicao_cod"] = df["instituicao"].map(INST_MAP).fillna(0).astype(int)

    return df


def safe_select(df, cols):
    """Seleciona apenas as colunas existentes no DataFrame."""
    return df[[c for c in cols if c in df.columns]].copy()


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO
# ---------------------------------------------------------------------------
secao("1. CARREGANDO DADOS")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {INPUT_FILE}\n"
        "Verifique se o Excel está na mesma pasta que este script."
    )

sheets   = pd.read_excel(INPUT_FILE, sheet_name=None)
df22_raw = sheets["PEDE2022"].copy()
df23_raw = sheets["PEDE2023"].copy()
df24_raw = sheets["PEDE2024"].copy()

print(f"  PEDE2022: {df22_raw.shape[0]} linhas | {df22_raw.shape[1]} colunas")
print(f"  PEDE2023: {df23_raw.shape[0]} linhas | {df23_raw.shape[1]} colunas")
print(f"  PEDE2024: {df24_raw.shape[0]} linhas | {df24_raw.shape[1]} colunas")


# ---------------------------------------------------------------------------
# 2. REMOÇÃO DE COLUNAS 100% NULAS
# ---------------------------------------------------------------------------
secao("2. REMOVENDO COLUNAS 100% NULAS")

df22 = drop_fully_null_cols(df22_raw, "2022")
df23 = drop_fully_null_cols(df23_raw, "2023")
df24 = drop_fully_null_cols(df24_raw, "2024")


# ---------------------------------------------------------------------------
# 3. PADRONIZAÇÃO DE NOMES
# ---------------------------------------------------------------------------
secao("3. PADRONIZANDO NOMES DE COLUNAS")

RENAME_2022 = {
    "RA": "ra", "Fase": "fase", "Turma": "turma", "Nome": "nome",
    "Ano nasc": "ano_nascimento", "Idade 22": "idade", "Gênero": "genero",
    "Ano ingresso": "ano_ingresso", "Instituição de ensino": "instituicao",
    "Pedra 20": "pedra_2020", "Pedra 21": "pedra_2021", "Pedra 22": "pedra_ano",
    "INDE 22": "inde_ano", "Cg": "cg", "Cf": "cf", "Ct": "ct",
    "Nº Av": "num_avaliadores",
    "Avaliador1": "avaliador_1", "Rec Av1": "rec_avaliador_1",
    "Avaliador2": "avaliador_2", "Rec Av2": "rec_avaliador_2",
    "Avaliador3": "avaliador_3", "Rec Av3": "rec_avaliador_3",
    "Avaliador4": "avaliador_4", "Rec Av4": "rec_avaliador_4",
    "IAA": "iaa", "IEG": "ieg", "IPS": "ips",
    "Rec Psicologia": "rec_psicologia", "IDA": "ida",
    "Matem": "nota_matematica", "Portug": "nota_portugues", "Inglês": "nota_ingles",
    "Indicado": "indicado", "Atingiu PV": "atingiu_pv", "IPV": "ipv",
    "IAN": "ian", "Fase ideal": "fase_ideal", "Defas": "defasagem",
    "Destaque IEG": "destaque_ieg", "Destaque IDA": "destaque_ida",
    "Destaque IPV": "destaque_ipv",
}

RENAME_2023 = {
    "RA": "ra", "Fase": "fase", "INDE 2023": "inde_ano", "Pedra 2023": "pedra_ano",
    "Turma": "turma", "Nome Anonimizado": "nome", "Data de Nasc": "data_nascimento",
    "Idade": "idade", "Gênero": "genero", "Ano ingresso": "ano_ingresso",
    "Instituição de ensino": "instituicao",
    "Pedra 20": "pedra_2020", "Pedra 21": "pedra_2021", "Pedra 22": "pedra_2022",
    "INDE 22": "inde_2022", "INDE 23": "inde_2023",
    "Nº Av": "num_avaliadores",
    "Avaliador1": "avaliador_1", "Avaliador2": "avaliador_2",
    "Avaliador3": "avaliador_3", "Avaliador4": "avaliador_4",
    "IAA": "iaa", "IEG": "ieg", "IPS": "ips", "IPP": "ipp",
    "Rec Psicologia": "rec_psicologia", "IDA": "ida",
    "Mat": "nota_matematica", "Por": "nota_portugues", "Ing": "nota_ingles",
    "Indicado": "indicado", "Atingiu PV": "atingiu_pv", "IPV": "ipv",
    "IAN": "ian", "Fase Ideal": "fase_ideal", "Defasagem": "defasagem",
    "Destaque IEG": "destaque_ieg", "Destaque IDA": "destaque_ida",
    "Destaque IPV": "destaque_ipv", "Destaque IPV.1": "destaque_ipv_2",
}

RENAME_2024 = {
    "RA": "ra", "Fase": "fase", "INDE 2024": "inde_ano", "Pedra 2024": "pedra_ano",
    "Turma": "turma", "Nome Anonimizado": "nome", "Data de Nasc": "data_nascimento",
    "Idade": "idade", "Gênero": "genero", "Ano ingresso": "ano_ingresso",
    "Instituição de ensino": "instituicao",
    "Pedra 20": "pedra_2020", "Pedra 21": "pedra_2021",
    "Pedra 22": "pedra_2022", "Pedra 23": "pedra_2023",
    "INDE 22": "inde_2022", "INDE 23": "inde_2023",
    "Nº Av": "num_avaliadores",
    "Avaliador1": "avaliador_1", "Avaliador2": "avaliador_2",
    "Avaliador3": "avaliador_3", "Avaliador4": "avaliador_4",
    "Avaliador5": "avaliador_5", "Avaliador6": "avaliador_6",
    "IAA": "iaa", "IEG": "ieg", "IPS": "ips", "IPP": "ipp",
    "Rec Psicologia": "rec_psicologia", "IDA": "ida",
    "Mat": "nota_matematica", "Por": "nota_portugues", "Ing": "nota_ingles",
    "Indicado": "indicado", "Atingiu PV": "atingiu_pv", "IPV": "ipv",
    "IAN": "ian", "Fase Ideal": "fase_ideal", "Defasagem": "defasagem",
    "Destaque IEG": "destaque_ieg", "Destaque IDA": "destaque_ida",
    "Destaque IPV": "destaque_ipv",
    "Escola": "escola",
    "Ativo/ Inativo": "status_ativo", "Ativo/ Inativo.1": "status_ativo_2",
}

df22 = df22.rename(columns=RENAME_2022)
df23 = df23.rename(columns=RENAME_2023)
df24 = df24.rename(columns=RENAME_2024)

df22["ano_referencia"] = 2022
df23["ano_referencia"] = 2023
df24["ano_referencia"] = 2024

print("  Colunas padronizadas com sucesso.")


# ---------------------------------------------------------------------------
# 4. CORREÇÃO DE TIPOS
# ---------------------------------------------------------------------------
secao("4. CORRIGINDO TIPOS DE DADOS")

# 4a. INDE 2024: valor 'INCLUIR' → NaN
n_incluir = (df24["inde_ano"] == "INCLUIR").sum()
print(f"  [2024] inde_ano: {n_incluir} 'INCLUIR' substituído(s) por NaN")
df24["inde_ano"] = pd.to_numeric(df24["inde_ano"].replace("INCLUIR", np.nan), errors="coerce")

# 4b. Fase: normalizar para int
df22["fase"] = pd.to_numeric(df22["fase"], errors="coerce")
df23["fase"] = df23["fase"].apply(extrair_fase)
df24["fase"] = df24["fase"].apply(extrair_fase)
print("  [Todas] fase: normalizada para numérico (ALFA=0)")

# 4c. Data de nascimento
df22["data_nascimento"] = pd.to_datetime(
    df22["ano_nascimento"].astype(str) + "-01-01", errors="coerce"
)
df22 = df22.drop(columns=["ano_nascimento"], errors="ignore")
df23["data_nascimento"] = pd.to_datetime(df23["data_nascimento"], errors="coerce")
df24["data_nascimento"] = pd.to_datetime(df24["data_nascimento"], errors="coerce")
print("  [Todas] data_nascimento: padronizada para datetime")

# 4d. Colunas numéricas
NUMERIC_COLS = [
    "inde_ano", "inde_2022", "inde_2023",
    "iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian",
    "nota_matematica", "nota_portugues", "nota_ingles",
    "num_avaliadores", "defasagem", "idade", "ano_ingresso",
]
for df_loop, label in [(df22, "2022"), (df23, "2023"), (df24, "2024")]:
    for col in NUMERIC_COLS:
        if col in df_loop.columns:
            df_loop[col] = pd.to_numeric(df_loop[col], errors="coerce")
print("  [Todas] indicadores e notas: convertidos para float")


# ---------------------------------------------------------------------------
# 4.5 INFERÊNCIA DO IPP 2022 POR ENGENHARIA REVERSA
# ---------------------------------------------------------------------------
secao("4.5 INFERINDO IPP 2022 POR ENGENHARIA REVERSA")

if "ipp" not in df22.columns:
    df22["ipp"] = np.nan

mask_sem_ipp = df22["ipp"].isna()
df22.loc[mask_sem_ipp, "ipp"] = df22[mask_sem_ipp].apply(calcular_ipp_reverso, axis=1)
print(f"  [2022] IPP inferido para {df22['ipp'].notna().sum()} alunos.")
print(f"  [2022] IPP ainda nulo (Fase 8 ou dados insuficientes): {df22['ipp'].isna().sum()}")

for df_loop, label in [(df23, "2023"), (df24, "2024")]:
    if "ipp" in df_loop.columns:
        mask = df_loop["ipp"].isna()
        n_antes = mask.sum()
        if n_antes > 0:
            df_loop.loc[mask, "ipp"] = df_loop[mask].apply(calcular_ipp_reverso, axis=1)
            print(f"  [{label}] {n_antes} IPP nulos → {n_antes - df_loop['ipp'].isna().sum()} inferidos")


# ---------------------------------------------------------------------------
# 5. PADRONIZAÇÃO DE CATEGORIAS
# ---------------------------------------------------------------------------
secao("5. PADRONIZANDO CATEGORIAS")

GENERO_MAP = {
    "menina": "Feminino", "menino": "Masculino",
    "feminino": "Feminino", "masculino": "Masculino",
}
for df_loop in [df22, df23, df24]:
    df_loop["genero"] = df_loop["genero"].str.strip().str.lower().map(GENERO_MAP)
print("  [Todas] genero: padronizado → 'Feminino' / 'Masculino'")

for df_loop in [df22, df23, df24]:
    df_loop["instituicao"] = df_loop["instituicao"].apply(normalizar_instituicao)
print("  [Todas] instituicao: categorias normalizadas")

PEDRA_ORDER = {"Quartzo": 1, "Ágata": 2, "Ametista": 3, "Topázio": 4}
PEDRA_COLS  = ["pedra_ano", "pedra_2020", "pedra_2021", "pedra_2022", "pedra_2023"]
for df_loop in [df22, df23, df24]:
    for col in PEDRA_COLS:
        if col in df_loop.columns:
            df_loop[col] = df_loop[col].map(PEDRA_ORDER)
print("  [Todas] pedra_*: Quartzo=1, Ágata=2, Ametista=3, Topázio=4")


# ---------------------------------------------------------------------------
# 6. TRATAMENTO DE VALORES NULOS
# ---------------------------------------------------------------------------
secao("6. TRATANDO VALORES NULOS")

INDICATOR_COLS = ["iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian"]
NOTE_COLS      = ["nota_matematica", "nota_portugues"]

for df_loop, label in [(df22, "2022"), (df23, "2023"), (df24, "2024")]:
    for col in INDICATOR_COLS + NOTE_COLS:
        if col in df_loop.columns and df_loop[col].isna().any():
            antes = df_loop[col].isna().sum()
            df_loop = impute_by_group(df_loop, col, strategy="median")
            depois = df_loop[col].isna().sum()
            if antes > 0:
                print(f"  [{label}] {col}: {antes} nulos → {depois} após imputação")

    # Inglês: ausência estrutural → 0 (não avaliado)
    if "nota_ingles" in df_loop.columns:
        antes = df_loop["nota_ingles"].isna().sum()
        df_loop["nota_ingles"] = df_loop["nota_ingles"].fillna(0)
        if antes > 0:
            print(f"  [{label}] nota_ingles: {antes} nulos → preenchidos com 0")

    # Pedras históricas: -1 = sem histórico
    for col in PEDRA_COLS:
        if col in df_loop.columns:
            df_loop[col] = df_loop[col].fillna(-1)

    # Idade
    if "idade" in df_loop.columns and df_loop["idade"].isna().any():
        df_loop = impute_by_group(df_loop, "idade", strategy="median")

    # Defasagem: 0 se nulo
    if "defasagem" in df_loop.columns:
        df_loop["defasagem"] = df_loop["defasagem"].fillna(0)

    # INDE histórico: -1 = ausência
    for col in ["inde_2022", "inde_2023"]:
        if col in df_loop.columns:
            df_loop[col] = df_loop[col].fillna(-1)

    # INDE do ano corrente: imputa pela mediana da fase
    if "inde_ano" in df_loop.columns and df_loop["inde_ano"].isna().any():
        antes = df_loop["inde_ano"].isna().sum()
        df_loop = impute_by_group(df_loop, "inde_ano", strategy="median")
        print(f"  [{label}] inde_ano: {antes} nulos → imputados pela mediana da fase")

    # Reatribui (loop cria cópia local)
    if label == "2022":
        df22 = df_loop
    elif label == "2023":
        df23 = df_loop
    else:
        df24 = df_loop


# ---------------------------------------------------------------------------
# 7. ENGENHARIA DE FEATURES
# ---------------------------------------------------------------------------
secao("7. ENGENHARIA DE FEATURES")

df22 = add_features(df22, 2022)
df23 = add_features(df23, 2023)
df24 = add_features(df24, 2024)
print("  Novas colunas: anos_no_programa, media_notas, media_indicadores,")
print("  evolucao_pedra, genero_feminino, instituicao_cod")


# ---------------------------------------------------------------------------
# 8. CONSOLIDAÇÃO
# ---------------------------------------------------------------------------
secao("8. CONSOLIDANDO BASE ÚNICA")

CORE_COLS = [
    "ra", "ano_referencia", "fase", "genero", "genero_feminino",
    "idade", "data_nascimento", "ano_ingresso", "anos_no_programa",
    "instituicao", "instituicao_cod",
    "pedra_ano", "pedra_2020", "pedra_2021", "pedra_2022",
    "inde_ano", "inde_2022",
    "num_avaliadores",
    "iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian",
    "nota_matematica", "nota_portugues", "nota_ingles",
    "media_notas", "media_indicadores", "evolucao_pedra",
    "defasagem",
]

base_22 = safe_select(df22, CORE_COLS)
base_23 = safe_select(df23, CORE_COLS + ["pedra_2023", "inde_2023"])
base_24 = safe_select(df24, CORE_COLS + ["pedra_2023", "inde_2023"])

base_consolidada = pd.concat([base_22, base_23, base_24], ignore_index=True)
print(f"  Base consolidada: {base_consolidada.shape[0]} linhas | {base_consolidada.shape[1]} colunas")

nulos_finais = base_consolidada.isna().sum()
nulos_finais = nulos_finais[nulos_finais > 0]
if len(nulos_finais) > 0:
    print("\n  Nulos remanescentes (ausência estrutural entre anos):")
    for col, n in nulos_finais.items():
        print(f"    {col}: {n} ({n/len(base_consolidada)*100:.1f}%)")
else:
    print("  Sem nulos remanescentes nas colunas core.")


# ---------------------------------------------------------------------------
# 9. EXPORTAÇÃO
# ---------------------------------------------------------------------------
secao("9. EXPORTANDO")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    base_consolidada.to_excel(writer, sheet_name="BASE_CONSOLIDADA", index=False)
    df22.to_excel(writer, sheet_name="PEDE2022_TRATADO", index=False)
    df23.to_excel(writer, sheet_name="PEDE2023_TRATADO", index=False)
    df24.to_excel(writer, sheet_name="PEDE2024_TRATADO", index=False)

print(f"  ✅ Arquivo salvo: {OUTPUT_FILE}")
print(f"     → BASE_CONSOLIDADA : {base_consolidada.shape}")
print(f"     → PEDE2022_TRATADO : {df22.shape}")
print(f"     → PEDE2023_TRATADO : {df23.shape}")
print(f"     → PEDE2024_TRATADO : {df24.shape}")
print("\n✅ Tratamento concluído com sucesso!")
