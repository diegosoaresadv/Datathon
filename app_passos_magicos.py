"""
=============================================================================
APP STREAMLIT — Passos Mágicos | Modelo Preditivo de Risco de Defasagem
=============================================================================
Como usar:
  1. Execute primeiro o script tratamento_pede.py para gerar BASE_PEDE_TRATADA.xlsx
  2. Coloque BASE_PEDE_TRATADA.xlsx na mesma pasta deste script
  3. Rode: streamlit run app_passos_magicos.py
  O arquivo .xlsx será carregado automaticamente, sem necessidade de upload.
=============================================================================
"""

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURAÇÃO DA PÁGINA — deve ser a primeira chamada Streamlit
# =============================================================================
st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# TEMA GLOBAL — alinhado ao layout da apresentação Gamma (azul escuro + moderno)
# =============================================================================
st.markdown("""
<style>
/* ── Paleta base ── */
:root {
    --pm-blue-deep:   #0D1B2A;
    --pm-blue-mid:    #1B4F72;
    --pm-blue-light:  #2E86C1;
    --pm-accent:      #F4A261;
    --pm-white:       #F8F9FA;
    --pm-card-bg:     #132233;
    --pm-border:      rgba(46,134,193,0.35);
}

/* ── Fundo principal ── */
.stApp {
    background: linear-gradient(160deg, #0D1B2A 0%, #102030 60%, #0D1B2A 100%);
    color: #E8EDF2;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0f2236 100%);
    border-right: 1px solid var(--pm-border);
}
[data-testid="stSidebar"] * { color: #CBD5DF !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 { color: #F4A261 !important; }

/* ── Títulos ── */
h1 { color: #F4A261 !important; letter-spacing: -0.5px; }
h2, h3 { color: #7EC8E3 !important; }

/* ── Métricas (st.metric) ── */
[data-testid="stMetric"] {
    background: var(--pm-card-bg);
    border: 1px solid var(--pm-border);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"]  { color: #8AAFC7 !important; font-size: 0.78rem; }
[data-testid="stMetricValue"]  { color: #F4A261 !important; font-weight: 700; }
[data-testid="stMetricDelta"]  { color: #7EC8E3 !important; }

/* ── Dataframe / tabelas ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--pm-border);
    border-radius: 8px;
    overflow: hidden;
}

/* ── Selectbox / radio / slider ── */
[data-testid="stSelectbox"] > div,
[data-testid="stRadio"] > div {
    background: var(--pm-card-bg);
    border-radius: 8px;
    border: 1px solid var(--pm-border);
}

/* ── Botões ── */
.stButton > button {
    background: linear-gradient(135deg, #1B4F72, #2E86C1);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Dividers ── */
hr { border-color: var(--pm-border) !important; }

/* ── Info / Warning / Error boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] { color: #8AAFC7 !important; }

/* ── Apresentação fullscreen container ── */
.pm-presentation-wrapper {
    position: relative;
    width: 100%;
    padding-top: 56.25%;   /* 16:9 */
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid var(--pm-border);
    box-shadow: 0 8px 40px rgba(0,0,0,0.55);
}
.pm-presentation-wrapper iframe {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    border: none;
}

/* ── Fullscreen button ── */
.pm-fs-btn {
    display: inline-block;
    margin-top: 14px;
    padding: 10px 24px;
    background: linear-gradient(135deg, #1B4F72, #2E86C1);
    color: #fff !important;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.92rem;
    transition: opacity .2s;
    cursor: pointer;
}
.pm-fs-btn:hover { opacity: 0.82; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTES
# =============================================================================
CORES = {
    "primaria":   "#1B4F72",
    "secundaria": "#2E86C1",
    "destaque":   "#E74C3C",
    "verde":      "#1E8449",
    "amarelo":    "#F39C12",
    "cinza":      "#7F8C8D",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# Colunas usadas pelo modelo preditivo
FEATURES = [
    "fase", "genero_feminino", "instituicao_cod", "anos_no_programa",
    "iaa", "ieg", "ips", "ida", "ipv",
    "nota_matematica", "nota_portugues", "nota_ingles",
    "media_notas", "media_indicadores",
    "pedra_ano", "pedra_2020", "pedra_2021",
]

# Colunas obrigatórias na base para o app funcionar
COLUNAS_OBRIGATORIAS = FEATURES + ["ian", "defasagem", "inde_ano"]

# Colunas opcionais (usadas em tabelas/gráficos mas não no modelo)
COLUNAS_OPCIONAIS = [
    "ra", "ano_referencia", "genero", "ipp",
    "pedra_2022", "pedra_2023", "inde_2022", "inde_2023",
]

PEDRA_LABEL = {1: "Quartzo", 2: "Ágata", 3: "Ametista", 4: "Topázio"}

INDICADORES_DISP = {
    "IDA — Desempenho Acadêmico": "ida",
    "IEG — Engajamento":          "ieg",
    "IAA — Autoavaliação":        "iaa",
    "IPS — Psicossocial":         "ips",
    "IPV — Ponto de Virada":      "ipv",
    "IPP — Psicopedagógico":      "ipp",
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def classificar_ian(v):
    if pd.isna(v):
        return "Sem dado"
    if v == 10.0:
        return "Adequado"
    if v == 5.0:
        return "Moderado"
    return "Severo"


def nivel_risco(prob):
    if prob < 0.3:
        return "🟢 Baixo Risco"
    if prob < 0.6:
        return "🟡 Risco Moderado"
    return "🔴 Alto Risco"


def label_nivel(prob):
    """Versão sem emoji para uso em tabelas/agrupamentos."""
    if prob < 0.3:
        return "Baixo Risco"
    if prob < 0.6:
        return "Risco Moderado"
    return "Alto Risco"


def fase_label(x):
    try:
        x = int(x)
        return "ALFA" if x == 0 else f"Fase {x}"
    except (ValueError, TypeError):
        return "N/D"


# =============================================================================
# CARREGAMENTO E VALIDAÇÃO DOS DADOS
# =============================================================================

@st.cache_data(show_spinner="Carregando base de dados…")
def carregar_dados(arquivo_bytes: bytes):
    """
    Lê o arquivo Excel tratado a partir de bytes (serializável pelo cache do Streamlit).
    Valida colunas obrigatórias e faz imputações de segurança.
    """
    import io
    buffer = io.BytesIO(arquivo_bytes)

    # ── Leitura ──────────────────────────────────────────────────────────
    try:
        df = pd.read_excel(buffer, sheet_name="BASE_CONSOLIDADA")
    except Exception:
        buffer.seek(0)
        xl = pd.ExcelFile(buffer)
        buffer.seek(0)
        df = pd.read_excel(buffer, sheet_name=xl.sheet_names[0])
        st.warning(
            f"Aba 'BASE_CONSOLIDADA' não encontrada. Usando '{xl.sheet_names[0]}'. "
            "Execute `tratamento_pede.py` para gerar o arquivo correto.",
            icon="⚠️",
        )

    # ── Validação de colunas obrigatórias ─────────────────────────────────
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        st.error(
            f"**Colunas obrigatórias ausentes:** `{faltando}`\n\n"
            "Execute o script `tratamento_pede.py` para gerar a base corretamente."
        )
        st.stop()

    # ── Colunas opcionais: cria com NaN se ausentes ───────────────────────
    for col in COLUNAS_OPCIONAIS:
        if col not in df.columns:
            df[col] = np.nan

    # ── Garantir 'ano_referencia' numérico ────────────────────────────────
    if df["ano_referencia"].isna().all():
        df["ano_referencia"] = 2022

    # ── Pedras históricas: -1 = sem histórico ─────────────────────────────
    for col in ["pedra_ano", "pedra_2020", "pedra_2021"]:
        df[col] = df[col].fillna(-1)

    # ── inde_ano: imputa nulos pela mediana da fase ────────────────────────
    if df["inde_ano"].isna().any():
        df["inde_ano"] = df.groupby("fase")["inde_ano"].transform(
            lambda x: x.fillna(x.median())
        )
        df["inde_ano"] = df["inde_ano"].fillna(df["inde_ano"].median())

    # ── ian e defasagem: preenche nulos com valores neutros ───────────────
    df["ian"] = df["ian"].fillna(10.0)      # 10 = adequado
    df["defasagem"] = df["defasagem"].fillna(0.0)  # 0 = sem defasagem

    return df


# =============================================================================
# TREINAMENTO DO MODELO
# =============================================================================

@st.cache_resource(show_spinner="Treinando modelo…")
def treinar_modelo(_df):
    """
    Treina Random Forest com as FEATURES definidas.
    Retorna: pipeline, X_test, y_test, y_prob_test, auc_score
    """
    df_m = _df.copy()

    # Target: em risco se qualquer critério for verdadeiro
    df_m["em_risco"] = (
        (df_m["defasagem"] < 0)
        | (df_m["ian"] <= 5.0)
        | (df_m["inde_ano"] < 6.5)
    ).astype(int)

    X = df_m[FEATURES].copy()
    y = df_m["em_risco"].copy()

    # Remove linhas onde TODAS as features são NaN
    mask_valido = X.notna().any(axis=1)
    X = X[mask_valido]
    y = y[mask_valido]

    # Split estratificado
    min_class = y.value_counts().min()
    stratify = y if min_class >= 2 else None
    test_size = 0.2 if len(X) >= 50 else 0.1

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob) if len(
        y_test.unique()) == 2 else float("nan")

    return pipe, X_test, y_test, y_prob, auc


# =============================================================================
# LOCALIZAÇÃO AUTOMÁTICA DO ARQUIVO .xlsx
# =============================================================================
_NOME_PADRAO = "BASE_PEDE_TRATADA.xlsx"
_DIR_APP = Path(__file__).parent.resolve()

def _encontrar_xlsx() -> Path | None:
    """
    Procura o arquivo .xlsx na pasta do script.
    Prioridade:
      1. BASE_PEDE_TRATADA.xlsx (nome padrão)
      2. Qualquer outro .xlsx encontrado na mesma pasta
    """
    # 1. Nome padrão
    caminho_padrao = _DIR_APP / _NOME_PADRAO
    if caminho_padrao.exists():
        return caminho_padrao
    # 2. Qualquer .xlsx na pasta
    xlsxs = sorted(_DIR_APP.glob("*.xlsx"))
    return xlsxs[0] if xlsxs else None


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("🎓 Passos Mágicos")
    st.caption("Datathon — Modelo Preditivo de Risco de Defasagem")
    st.divider()

    arquivo_auto = _encontrar_xlsx()

    if arquivo_auto:
        st.success(f"📂 Arquivo carregado automaticamente:\n\n`{arquivo_auto.name}`")
    else:
        st.error(
            f"⚠️ Nenhum arquivo `.xlsx` encontrado em:\n\n`{_DIR_APP}`\n\n"
            f"Coloque `{_NOME_PADRAO}` nessa pasta e reinicie o app."
        )

    st.divider()

    pagina = st.radio(
        "🧭 Navegação",
        [
            "📋 Apresentação",
            "📊 Visão Geral",
            "🔍 Análise por Indicador",
            "🤖 Modelo Preditivo",
            "🧑‍🎓 Predição Individual",
        ],
    )

    st.divider()
    st.caption("Desenvolvido para o Datathon Passos Mágicos")




# =============================================================================
# PÁGINA 0 — APRESENTAÇÃO (não depende de dados)
# =============================================================================
if pagina == "📋 Apresentação":
    st.title("📋 Apresentação — Análise de Dados & Modelo Preditivo")
    st.caption(
        "Visualize a apresentação completa do projeto Passos Mágicos. "
        "Use o botão de tela cheia (⛶) no canto inferior direito do slide para expandir."
    )
    st.markdown("""
    <div class="pm-presentation-wrapper">
        <iframe
            src="https://gamma.app/embed/7k4uocz3yper0qj"
            allow="fullscreen"
            allowfullscreen
            title="Passos Mágicos — Análise de Dados & Modelo Preditivo de Risco">
        </iframe>
    </div>
    <div style="margin-top:16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
        <a class="pm-fs-btn"
           href="https://gamma.app/docs/7k4uocz3yper0qj"
           target="_blank" rel="noopener">
            ⛶ &nbsp;Abrir em Tela Cheia
        </a>
        <span style="color:#8AAFC7; font-size:0.85rem;">
            ou pressione o ícone de tela cheia no canto inferior direito da apresentação
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =============================================================================
# GUARD: sem arquivo → para execução aqui
# =============================================================================
if arquivo_auto is None:
    st.title("🎓 Passos Mágicos — Risco de Defasagem")
    st.error(
        f"📂 **Arquivo não encontrado.**\n\n"
        f"Coloque o arquivo `{_NOME_PADRAO}` na mesma pasta deste script:\n\n"
        f"`{_DIR_APP}`\n\n"
        "Em seguida, recarregue a página.",
        icon="🚫",
    )
    st.stop()

else:
    # =========================================================================
    # CARREGAMENTO + MODELO (executa automaticamente com o arquivo da pasta)
    # =========================================================================
    _bytes = arquivo_auto.read_bytes()
    df = carregar_dados(_bytes)

    pipeline, X_test, y_test, y_prob_test, auc_score = treinar_modelo(df)

    # Probabilidades para toda a base
    X_all = df[FEATURES].copy()
    X_all_filled = X_all.fillna(X_all.median(numeric_only=True))
    df["prob_risco"] = pipeline.predict_proba(X_all_filled)[:, 1]
    df["nivel_risco"] = df["prob_risco"].apply(label_nivel)
    df["ian_classe"] = df["ian"].apply(classificar_ian)

    # =============================================================================
    # PÁGINA 1 — VISÃO GERAL
    # =============================================================================
    if pagina == "📊 Visão Geral":
        st.title("📊 Visão Geral — Passos Mágicos 2022–2024")

        # ── KPIs ─────────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total de Alunos",      f"{len(df):,}")
        c2.metric("INDE Médio",            f"{df['inde_ano'].mean():.2f}")
        c3.metric("IDA Médio",             f"{df['ida'].mean():.2f}")
        c4.metric("% Adequado (IAN)",
                  f"{(df['ian'] == 10).mean()*100:.0f}%")
        c5.metric("% Alto Risco (Modelo)",
                  f"{(df['nivel_risco'] == 'Alto Risco').mean()*100:.1f}%")

        st.divider()
        col1, col2 = st.columns(2)

        # ── Distribuição IAN por Ano ──────────────────────────────────────────
        with col1:
            st.subheader("Distribuição IAN por Ano")
            df_ian = df[
                df["ano_referencia"].notna() & (df["ian_classe"] != "Sem dado")
            ].copy()
            evolucao = (
                df_ian.groupby(["ano_referencia", "ian_classe"])
                .size().reset_index(name="n")
            )
            if not evolucao.empty:
                pivot = evolucao.pivot(
                    index="ano_referencia", columns="ian_classe", values="n"
                ).fillna(0)
                pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
                order_cols = [c for c in ["Adequado", "Moderado",
                                          "Severo"] if c in pivot_pct.columns]
                fig, ax = plt.subplots(figsize=(7, 4))
                pivot_pct[order_cols].plot(
                    kind="bar", stacked=True, ax=ax,
                    color=[CORES["verde"], CORES["amarelo"],
                           CORES["destaque"]][:len(order_cols)],
                    edgecolor="white", width=0.6,
                )
                ax.set_xlabel("Ano de Referência")
                ax.set_ylabel("% Alunos")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.legend(loc="upper right", fontsize=9)
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Sem dados de IAN disponíveis.")

        # ── Crescimento da Base ───────────────────────────────────────────────
        with col2:
            st.subheader("Crescimento da Base de Alunos por Ano")
            alunos_ano = df.groupby("ano_referencia").size().sort_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            bar_colors = [CORES["primaria"],
                          CORES["secundaria"], CORES["verde"]]
            bars = ax.bar(
                alunos_ano.index.astype(int), alunos_ano.values,
                color=bar_colors[:len(alunos_ano)],
                edgecolor="white", width=0.5,
            )
            for bar, v in zip(bars, alunos_ano.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + alunos_ano.max() * 0.02,
                    f"{v:,}", ha="center", fontweight="bold", fontsize=11,
                )
            ax.set_xlabel("Ano")
            ax.set_ylabel("Nº de Alunos")
            ax.set_ylim(0, alunos_ano.max() * 1.25)
            ax.set_xticks(alunos_ano.index.astype(int))
            st.pyplot(fig, use_container_width=True)

        st.divider()
        col3, col4 = st.columns(2)

        # ── INDE Médio por Pedra ──────────────────────────────────────────────
        with col3:
            st.subheader("INDE Médio por Pedra")
            df_pedra = df[df["pedra_ano"].notna() & (
                df["pedra_ano"] != -1)].copy()
            df_pedra["pedra_nome"] = df_pedra["pedra_ano"].map(PEDRA_LABEL)
            ordem = ["Quartzo", "Ágata", "Ametista", "Topázio"]
            medias = (
                df_pedra.groupby("pedra_nome")["inde_ano"]
                .mean().reindex(ordem).dropna()
            )
            if not medias.empty:
                fig, ax = plt.subplots(figsize=(7, 4))
                cores_pedra = [CORES["cinza"], CORES["amarelo"],
                               CORES["secundaria"], CORES["primaria"]]
                bars = ax.bar(
                    medias.index, medias.values,
                    color=cores_pedra[:len(medias)],
                    edgecolor="white", width=0.6,
                )
                for bar, v in zip(bars, medias.values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + 0.05, f"{v:.2f}",
                        ha="center", fontweight="bold",
                    )
                ax.set_ylabel("INDE Médio")
                ax.set_ylim(0, 10)
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Sem dados de pedra disponíveis.")

        # ── Matriz de Correlação ──────────────────────────────────────────────
        with col4:
            st.subheader("Correlação entre Indicadores")
            cols_heat = [c for c in ["inde_ano", "ida", "ieg",
                                     "iaa", "ips", "ipv", "ian"] if c in df.columns]
            heat = df[cols_heat].corr()
            mask = np.triu(np.ones_like(heat, dtype=bool))
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(
                heat, mask=mask, annot=True, fmt=".2f",
                cmap="Blues", ax=ax, linewidths=0.5, annot_kws={"size": 8},
            )
            st.pyplot(fig, use_container_width=True)

        st.divider()

        # ── Risco por Nível (gráfico de pizza) ───────────────────────────────
        st.subheader("Distribuição Geral de Risco")
        contagem_risco = df["nivel_risco"].value_counts()
        ordem_risco = [r for r in ["Baixo Risco", "Risco Moderado",
                                   "Alto Risco"] if r in contagem_risco.index]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            contagem_risco[ordem_risco],
            labels=ordem_risco,
            autopct="%1.1f%%",
            colors=[CORES["verde"], CORES["amarelo"], CORES["destaque"]],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title("Alunos por Nível de Risco")
        col_pie, _ = st.columns([1, 2])
        with col_pie:
            st.pyplot(fig, use_container_width=True)

    # =============================================================================
    # PÁGINA 2 — ANÁLISE POR INDICADOR
    # =============================================================================
    elif pagina == "🔍 Análise por Indicador":
        st.title("🔍 Análise por Indicador")

        # Só exibe indicadores que existem na base carregada
        opcoes = {k: v for k, v in INDICADORES_DISP.items() if v in df.columns}
        if not opcoes:
            st.error("Nenhum indicador encontrado na base.")
            st.stop()

        indicador = st.selectbox("Selecione o indicador", list(opcoes.keys()))
        col = opcoes[indicador]
        df_ind = df[df[col].notna()].copy()

        # KPIs do indicador
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{col.upper()} Médio",   f"{df_ind[col].mean():.2f}")
        c2.metric("Mediana",                 f"{df_ind[col].median():.2f}")
        c3.metric("Desvio Padrão",           f"{df_ind[col].std():.2f}")
        c4.metric("Alunos com dado",         f"{len(df_ind):,}")

        col1, col2 = st.columns(2)

        # ── Distribuição por Ano ──────────────────────────────────────────────
        with col1:
            st.subheader(f"{indicador} — Distribuição por Ano")
            fig, ax = plt.subplots(figsize=(7, 4))
            anos = sorted(df_ind["ano_referencia"].dropna().unique())
            for ano in anos:
                grp = df_ind[df_ind["ano_referencia"] == ano]
                ax.hist(grp[col], bins=20, alpha=0.55,
                        label=str(int(ano)), density=True)
            ax.set_xlabel(col.upper())
            ax.set_ylabel("Densidade")
            ax.legend(title="Ano")
            st.pyplot(fig, use_container_width=True)

        # ── Scatter × INDE ───────────────────────────────────────────────────
        with col2:
            st.subheader(f"{indicador} × INDE")
            df_sc = df_ind[df_ind["inde_ano"].notna()].copy()
            if len(df_sc) >= 5:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.scatter(df_sc[col], df_sc["inde_ano"],
                           alpha=0.18, color=CORES["primaria"], s=12)
                corr = df_sc[[col, "inde_ano"]].corr().iloc[0, 1]
                m, b = np.polyfit(df_sc[col].values,
                                  df_sc["inde_ano"].values, 1)
                xl = np.linspace(df_sc[col].min(), df_sc[col].max(), 100)
                ax.plot(xl, m * xl + b, color=CORES["destaque"], lw=2,
                        label=f"r = {corr:.2f}")
                ax.set_xlabel(col.upper())
                ax.set_ylabel("INDE")
                ax.legend()
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Dados insuficientes para o gráfico de dispersão.")

        # ── Boxplot por Fase ──────────────────────────────────────────────────
        st.subheader(f"{indicador} por Fase")
        df_fase = df_ind[df_ind["fase"].notna() & (df_ind["fase"] <= 8)].copy()
        df_fase["fase_label"] = df_fase["fase"].apply(fase_label)
        ordem_fase = ["ALFA"] + [f"Fase {i}" for i in range(1, 9)]
        ordem_fase = [
            o for o in ordem_fase if o in df_fase["fase_label"].values]
        if ordem_fase:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.boxplot(
                data=df_fase, x="fase_label", y=col,
                order=ordem_fase, palette="Blues", ax=ax, width=0.6,
            )
            ax.set_xlabel("Fase")
            ax.set_ylabel(col.upper())
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Sem dados de fase disponíveis.")

        # ── Evolução da Média por Ano ─────────────────────────────────────────
        st.subheader(f"Evolução da Média de {col.upper()} por Ano")
        evolucao = df_ind.groupby("ano_referencia")[col].mean().reset_index()
        if len(evolucao) >= 2:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(
                evolucao["ano_referencia"].astype(int),
                evolucao[col],
                marker="o", lw=2.5, color=CORES["primaria"],
            )
            for _, row_ev in evolucao.iterrows():
                ax.annotate(
                    f"{row_ev[col]:.2f}",
                    (int(row_ev["ano_referencia"]), row_ev[col]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold",
                )
            ax.set_xlabel("Ano")
            ax.set_ylabel(f"Média {col.upper()}")
            ax.set_xticks(evolucao["ano_referencia"].astype(int))
            st.pyplot(fig, use_container_width=True)

    # =============================================================================
    # PÁGINA 3 — MODELO PREDITIVO
    # =============================================================================
    elif pagina == "🤖 Modelo Preditivo":
        st.title("🤖 Modelo Preditivo de Risco de Defasagem")

        st.info(
            "**Critério de risco (target):** Um aluno é considerado *em risco* se:\n"
            "`defasagem < 0` **ou** `IAN ≤ 5.0` **ou** `INDE < 6.5`",
            icon="ℹ️",
        )

        # ── KPIs do Modelo ────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        auc_label = f"{auc_score:.4f}" if not np.isnan(auc_score) else "N/D"
        em_risco_n = (df["nivel_risco"] == "Alto Risco").sum()
        c1.metric("AUC (Teste)",         auc_label,
                  delta="Random Forest")
        c2.metric("Alunos em Alto Risco", f"{em_risco_n:,}")
        c3.metric("% Alto Risco",         f"{em_risco_n/len(df)*100:.1f}%")
        c4.metric("Total Avaliado",       f"{len(df):,}")

        st.divider()
        col1, col2 = st.columns(2)

        # ── Curva ROC ─────────────────────────────────────────────────────────
        with col1:
            st.subheader("Curva ROC")
            if not np.isnan(auc_score) and len(y_test.unique()) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob_test)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color=CORES["primaria"],
                        lw=2.5, label=f"AUC = {auc_score:.3f}")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatório")
                ax.fill_between(fpr, tpr, alpha=0.08, color=CORES["primaria"])
                ax.set_xlabel("Taxa de Falsos Positivos")
                ax.set_ylabel("Taxa de Verdadeiros Positivos")
                ax.legend(fontsize=10)
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning(
                    "AUC indisponível — apenas uma classe no conjunto de teste.")

        # ── Importância das Features ──────────────────────────────────────────
        with col2:
            st.subheader("Importância das Features (Top 12)")
            imp = pd.Series(
                pipeline["clf"].feature_importances_, index=FEATURES
            ).sort_values(ascending=True).tail(12)
            fig, ax = plt.subplots(figsize=(6, 5))
            imp.plot(kind="barh", ax=ax,
                     color=CORES["primaria"], edgecolor="white")
            ax.set_xlabel("Importância Relativa")
            st.pyplot(fig, use_container_width=True)

        st.divider()
        col3, col4 = st.columns(2)

        # ── Risco por Fase ────────────────────────────────────────────────────
        with col3:
            st.subheader("Risco por Fase")
            df_rf = df[df["fase"].notna() & (df["fase"] <= 8)].copy()
            df_rf["fase_label"] = df_rf["fase"].apply(
                lambda x: "ALFA" if int(x) == 0 else f"F{int(x)}"
            )
            risco_fase = (
                df_rf.groupby(["fase_label", "nivel_risco"])
                .size().unstack(fill_value=0)
            )
            risco_pct = risco_fase.div(risco_fase.sum(axis=1), axis=0) * 100
            cols_plot = [c for c in ["Baixo Risco", "Risco Moderado",
                                     "Alto Risco"] if c in risco_pct.columns]
            fig, ax = plt.subplots(figsize=(7, 4))
            risco_pct[cols_plot].plot(
                kind="bar", stacked=True, ax=ax, edgecolor="white", width=0.7,
                color=[CORES["verde"], CORES["amarelo"],
                       CORES["destaque"]][:len(cols_plot)],
            )
            ax.set_xlabel("Fase")
            ax.set_ylabel("% Alunos")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.legend(loc="upper right", fontsize=8)
            st.pyplot(fig, use_container_width=True)

        # ── Distribuição das Probabilidades ──────────────────────────────────
        with col4:
            st.subheader("Distribuição da Probabilidade de Risco")
            y_arr = np.array(y_test)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(
                y_prob_test[y_arr == 0], bins=25, alpha=0.6,
                color=CORES["verde"], label="Sem Risco", density=True,
            )
            if (y_arr == 1).any():
                ax.hist(
                    y_prob_test[y_arr == 1], bins=25, alpha=0.6,
                    color=CORES["destaque"], label="Em Risco", density=True,
                )
            ax.axvline(0.5, color="black", ls="--",
                       lw=1.5, label="Threshold 0.5")
            ax.set_xlabel("P(Em Risco)")
            ax.set_ylabel("Densidade")
            ax.legend(fontsize=9)
            st.pyplot(fig, use_container_width=True)

        # ── Tabela: Alunos de Maior Risco ─────────────────────────────────────
        st.divider()
        st.subheader("🚨 Alunos com Maior Probabilidade de Risco (Top 20)")
        colunas_tab = [
            c for c in
            ["ra", "ano_referencia", "fase", "genero", "inde_ano",
                "ida", "ieg", "ian", "prob_risco", "nivel_risco"]
            if c in df.columns
        ]
        top_risco = (
            df[colunas_tab]
            .sort_values("prob_risco", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        top_risco["prob_risco"] = top_risco["prob_risco"].apply(
            lambda v: f"{v:.1%}")
        if "fase" in top_risco.columns:
            top_risco["fase"] = top_risco["fase"].apply(
                lambda x: fase_label(x) if pd.notna(x) else "N/D"
            )
        if "ano_referencia" in top_risco.columns:
            top_risco["ano_referencia"] = top_risco["ano_referencia"].apply(
                lambda x: int(x) if pd.notna(x) else "N/D"
            )
        st.dataframe(top_risco, use_container_width=True, hide_index=True)

    # =============================================================================
    # PÁGINA 4 — PREDIÇÃO INDIVIDUAL
    # =============================================================================
    elif pagina == "🧑‍🎓 Predição Individual":
        st.title("🧑‍🎓 Predição Individual de Risco")
        st.caption(
            "Preencha os dados do aluno para estimar a probabilidade de risco de defasagem.")

        with st.form("form_aluno"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Perfil do Aluno**")
                fase_input = st.selectbox(
                    "Fase", [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    format_func=lambda x: "ALFA" if x == 0 else f"Fase {x}",
                )
                genero_input = st.radio(
                    "Gênero", ["Feminino", "Masculino"], horizontal=True)
                anos_prog = st.number_input(
                    "Anos no Programa", min_value=0, max_value=20, value=3)
                inst = st.selectbox(
                    "Instituição de Ensino",
                    ["Pública", "Privada", "Concluiu 3º EM",
                        "Universitário Formado"],
                )
                inst_cod = {"Pública": 0, "Privada": 1,
                            "Concluiu 3º EM": 2, "Universitário Formado": 3}[inst]

            with c2:
                st.markdown("**Indicadores**")
                iaa_v = st.slider("IAA — Autoavaliação",   0.0, 10.0, 7.5, 0.1)
                ieg_v = st.slider("IEG — Engajamento",     0.0, 10.0, 7.5, 0.1)
                ips_v = st.slider("IPS — Psicossocial",    0.0, 10.0, 6.5, 0.1)
                ida_v = st.slider("IDA — Desempenho",      0.0, 10.0, 6.5, 0.1)
                ipv_v = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 7.0, 0.1)

            with c3:
                st.markdown("**Notas e Histórico de Pedras**")
                mat_v = st.slider("Nota Matemática", 0.0, 10.0, 6.5, 0.1)
                por_v = st.slider("Nota Português",  0.0, 10.0, 6.5, 0.1)
                ing_v = st.slider(
                    "Nota Inglês (0 = não avaliado)", 0.0, 10.0, 0.0, 0.1)
                pedra_atual = st.selectbox(
                    "Pedra Atual", [1, 2, 3, 4],
                    format_func=lambda x: PEDRA_LABEL[x],
                )
                pedra_20 = st.selectbox(
                    "Pedra 2020", [-1, 1, 2, 3, 4],
                    format_func=lambda x: "Sem dado" if x == -
                    1 else PEDRA_LABEL[x],
                )
                pedra_21 = st.selectbox(
                    "Pedra 2021", [-1, 1, 2, 3, 4],
                    format_func=lambda x: "Sem dado" if x == -
                    1 else PEDRA_LABEL[x],
                )

            submitted = st.form_submit_button(
                "🔮 Calcular Risco", use_container_width=True)

        if submitted:
            # Derivadas
            notas_validas = [v for v in [mat_v, por_v, ing_v] if v > 0]
            media_n = float(np.mean(notas_validas)) if notas_validas else 0.0
            media_ind = float(np.mean([iaa_v, ieg_v, ips_v, ipv_v]))

            entrada = pd.DataFrame([{
                "fase":               fase_input,
                "genero_feminino":    1 if genero_input == "Feminino" else 0,
                "instituicao_cod":    inst_cod,
                "anos_no_programa":   anos_prog,
                "iaa":                iaa_v,
                "ieg":                ieg_v,
                "ips":                ips_v,
                "ida":                ida_v,
                "ipv":                ipv_v,
                "nota_matematica":    mat_v,
                "nota_portugues":     por_v,
                "nota_ingles":        ing_v,
                "media_notas":        media_n,
                "media_indicadores":  media_ind,
                "pedra_ano":          pedra_atual,
                "pedra_2020":         pedra_20,
                "pedra_2021":         pedra_21,
            }])

            prob = float(pipeline.predict_proba(entrada)[0][1])
            nivel = nivel_risco(prob)

            # ── Card de resultado ─────────────────────────────────────────────
            st.divider()
            _, col_res, _ = st.columns([1, 2, 1])
            with col_res:
                cor = (
                    CORES["verde"] if prob < 0.3 else
                    CORES["amarelo"] if prob < 0.6 else
                    CORES["destaque"]
                )
                st.markdown(
                    f"""
                    <div style='text-align:center; padding:24px; border-radius:14px;
                                background:{cor}18; border: 2.5px solid {cor}'>
                        <h2 style='color:{cor}; margin:0'>{nivel}</h2>
                        <h1 style='color:{cor}; margin:10px 0; font-size:3.5rem'>{prob:.1%}</h1>
                        <p style='color:#555; margin:0; font-size:0.95rem'>
                            Probabilidade estimada de Risco de Defasagem
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Comparação com a base ─────────────────────────────────────────
            st.divider()
            st.subheader("📋 Comparação com a Base de Alunos")

            benchmarks = [
                ("IAA",         "iaa",              iaa_v),
                ("IEG",         "ieg",              ieg_v),
                ("IPS",         "ips",              ips_v),
                ("IDA",         "ida",              ida_v),
                ("IPV",         "ipv",              ipv_v),
                ("Matemática",  "nota_matematica",  mat_v),
                ("Português",   "nota_portugues",   por_v),
                ("Inglês",      "nota_ingles",      ing_v),
            ]

            rows = []
            for label_b, col_b, valor in benchmarks:
                if col_b in df.columns:
                    media_base = df[col_b].mean()
                    diff = valor - media_base
                    situacao = "✅ Acima da média" if diff >= 0 else "⚠️ Abaixo da média"
                    rows.append({
                        "Indicador":       label_b,
                        "Valor do Aluno":  round(valor, 2),
                        "Média da Base":   round(media_base, 2),
                        "Diferença":       f"{diff:+.2f}",
                        "Situação":        situacao,
                    })

            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)

            # ── Gráfico de radar simplificado (barras horizontais) ────────────
            st.subheader("📊 Perfil do Aluno vs. Média da Base")
            ind_radar = [r for r in rows if r["Indicador"]
                         in ["IAA", "IEG", "IPS", "IDA", "IPV"]]
            if ind_radar:
                fig, ax = plt.subplots(figsize=(8, 3))
                labels = [r["Indicador"] for r in ind_radar]
                vals = [r["Valor do Aluno"] for r in ind_radar]
                medias = [r["Média da Base"] for r in ind_radar]
                x = np.arange(len(labels))
                w = 0.35
                ax.bar(x - w/2, vals,   width=w, label="Aluno",
                       color=CORES["primaria"],  alpha=0.85, edgecolor="white")
                ax.bar(x + w/2, medias, width=w, label="Média",
                       color=CORES["cinza"],     alpha=0.65, edgecolor="white")
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_ylim(0, 10)
                ax.set_ylabel("Valor")
                ax.legend()
                st.pyplot(fig, use_container_width=True)

            # ── Recomendação ──────────────────────────────────────────────────
            st.divider()
            if prob >= 0.6:
                st.error(
                    "**⚠️ Alto Risco — Intervenção imediata recomendada.**\n\n"
                    "Este aluno apresenta alta probabilidade de defasagem. "
                    "Recomenda-se acionamento imediato de suporte psicossocial, "
                    "reforço acadêmico e acompanhamento individualizado."
                )
            elif prob >= 0.3:
                st.warning(
                    "**📌 Risco Moderado — Monitoramento reforçado.**\n\n"
                    "Acompanhe este aluno com frequência maior. "
                    "Ações preventivas e engajamento com a família são recomendados."
                )
            else:
                st.success(
                    "**✅ Baixo Risco — Situação estável.**\n\n"
                    "O aluno apresenta baixa probabilidade de defasagem. "
                    "Mantenha o acompanhamento regular."
                )
