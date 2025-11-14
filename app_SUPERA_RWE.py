import streamlit as st
import pandas as pd
import fitz
import re
import pickle
import io

# ===========================================
# 1. Carregar modelo e vectorizer
# ===========================================
with open("modelo_rwe.pkl", "rb") as f:
    clf = pickle.load(f)

with open("vectorizer_rwe.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("SUPERA – Analisador de Consultas Públicas (RWE)")
st.write("Upload do PDF → extração automática → classificação RWE → relatório.")

# ===========================================
# 2. Funções auxiliares
# ===========================================
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def read_pdf_text_blocks(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    doc.close()
    return clean_text("\n".join(text))

def split_contributions(raw_text):
    t = clean_text(raw_text)
    t = re.sub(r"(\d{2}/\d{2}/\d{4})", r"\1\n<<END>>\n", t)
    blocks = [b.strip() for b in t.split("<<END>>") if len(b.strip()) > 40]
    return blocks

def parse_block(block_text):
    """Extrai campos estruturados (igual ao modelo do Colab)."""

    b = " " + clean_text(block_text) + " "

    # Tipo de respondente
    if re.search(r'Familiar|cuidador', b, re.IGNORECASE):
        tipo = "Familiar/cuidador"
    elif re.search(r'Profissional|médic|enfermeir|farmacêutic', b, re.IGNORECASE):
        tipo = "Profissional de saúde"
    elif re.search(r'Paciente\b', b, re.IGNORECASE):
        tipo = "Paciente"
    elif re.search(r'Interessado', b, re.IGNORECASE):
        tipo = "Interessado no tema"
    else:
        tipo = "Outro/Indefinido"

    # Data
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', b)
    data_val = datas[-1] if datas else ""

    def extract(patterns):
        for pat in patterns:
            m = re.search(pat, b, flags=re.IGNORECASE | re.DOTALL)
            if m:
                return clean_text(m.group(1))
        return ""

    opiniao = extract([
        r'1[ªa]?\s*[-–]\s*(.*?)(?=2[ªa]?\s*[-–]|$)',
        r'12\s*[-–]\s*(.*?)(?=22\s*[-–]|$)'
    ])

    experiencia = extract([
        r'2[ªa]?\s*[-–]\s*(.*?)(?=3[ªa]?\s*[-–]|$)',
        r'22\s*[-–]\s*(.*?)(?=32\s*[-–]|$)'
    ])

    evidencias = extract([
        r'4[ªa]?\s*[-–]\s*(.*?)(?=5[ªa]?\s*[-–]|$)',
        r'42\s*[-–]\s*(.*?)(?=52\s*[-–]|$)'
    ])

    economico = extract([
        r'5[ªa]?\s*[-–]\s*(.*?)(?=$)',
        r'52\s*[-–]\s*(.*?)(?=$)'
    ])

    return {
        "Tipo_de_respondente": tipo,
        "Data": data_val,
        "Opiniao": opiniao,
        "Experiencia": experiencia,
        "Evidencias_clinicas": evidencias,
        "Estudos_economicos": economico,
        "Texto_unificado": clean_text(" ".join([opiniao, experiencia, evidencias, economico]))
    }

# ===========================================
# 3. Upload do PDF
# ===========================================
uploaded_pdf = st.file_uploader("Faça upload do PDF da consulta pública", type=["pdf"])

if uploaded_pdf:

    st.info("Processando PDF… aguarde alguns segundos.")

    raw = read_pdf_text_blocks(uploaded_pdf.read())
    blocks = split_contributions(raw)

   parsed_rows = [parse_block(b) for b in blocks]
df_pred = pd.DataFrame(parsed_rows)

    # Garantir que tudo é texto antes da vetorização
    df_pred["Texto_unificado"] = df_pred["Texto_unificado"].fillna("").astype(str)

    # Vetorização
    X_vec = vectorizer.transform(df_pred["Texto_unificado"])
    probs = clf.predict_proba(X_vec)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_pred["RWE_predito"] = preds
    df_pred["Confianca"] = (probs * 100).round(1)

    # Criar nível de RWE baseado na confiança
    def nivel_rwe(p):
        if p >= 75:
            return "Alto"
        elif p >= 50:
            return "Médio"
        else:
            return "Baixo"

    df_pred["Nivel_RWE"] = df_pred["Confianca"].apply(nivel_rwe)

    # ===========================================
    # Resumo
    # ===========================================
    st.success("PDF processado com sucesso!")

    st.metric("Total de contribuições", len(df_pred))
    st.metric("% com RWE", f"{df_pred['RWE_predito'].mean() * 100:.1f}%")

    st.subheader("📊 Distribuição de RWE")
    st.bar_chart(df_pred["RWE_predito"].value_counts())

    st.subheader("📄 Amostra das contribuições classificadas")
    st.dataframe(df_pred.head(20))

    # ======================================================
    # 🔥 TABELA ANALÍTICA (igual ao modelo do Colab)
    # ======================================================
    cols = [
        "Nivel_RWE", "Confianca", "Tipo_de_respondente",
        "Data", "Opiniao", "Experiencia",
        "Evidencias_clinicas", "Estudos_economicos",
        "Texto_unificado"
    ]

    # Garantir colunas
    for c in [
        "Opiniao", 
        "Experiencia", 
        "Evidencias_clinicas", 
        "Estudos_economicos",
        "Tipo_de_respondente",
        "Data",
        "Texto_unificado"
    ]:
        if c not in df_pred.columns:
            df_pred[c] = ""

    df_pp = df_pred[cols].copy()

    df_pp.rename(columns={
        "Nivel_RWE": "Nível RWE",
        "Confianca": "Probabilidade (%)",
        "Tipo_de_respondente": "Respondente",
        "Opiniao": "Opinião",
        "Experiencia": "Experiência",
        "Evidencias_clinicas": "Evidências clínicas",
        "Estudos_economicos": "Econômico",
        "Texto_unificado": "Texto completo"
    }, inplace=True)

    st.subheader("📊 Tabela analítica (pronta para PowerPoint)")
    st.dataframe(df_pp, use_container_width=True)

    # botão: baixar tabela analítica
    buffer_pp = io.BytesIO()
    df_pp.to_excel(buffer_pp, index=False)
    buffer_pp.seek(0)

    st.download_button(
        label="📥 Baixar tabela analítica (Excel)",
        data=buffer_pp,
        file_name="tabela_analitica_SUPERA_RWE.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # botão: baixar relatório completo
    buffer = io.BytesIO()
    df_pred.to_excel(buffer, index=False)
    buffer.seek(0)

    st.subheader("📁 Baixar Excel com análise completa")
    st.download_button(
        label="📥 Baixar relatório completo",
        data=buffer,
        file_name="relatorio_SUPERA_RWE.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
