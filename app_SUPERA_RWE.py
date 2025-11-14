import streamlit as st
import pandas as pd
import fitz
import re
import pickle

# =========================================
# 1. Carregar modelo e vectorizer
# =========================================
clf = pickle.load(open("modelo_rwe.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer_rwe.pkl", "rb"))

# =========================================
# 2. Funções auxiliares
# =========================================
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return clean_text(text)

# =========================================
# 3. Interface Streamlit
# =========================================
st.title("📊 SUPERA – Analisador de Consultas Públicas (RWE)")

uploaded_pdf = st.file_uploader("Faça upload do PDF da Consulta Pública", type=["pdf"])

if uploaded_pdf:
    st.info("Processando PDF...")

    # Extrair texto
    full_text = extract_text_from_pdf(uploaded_pdf)

    # Criar DataFrame simples (versão leve)
    df = pd.DataFrame({"Texto_unificado": [full_text]})
    
    # Vetorizar e classificar
    X_vec = vectorizer.transform(df["Texto_unificado"])
    prob = clf.predict_proba(X_vec)[0][1]
    pred = int(prob >= 0.5)

    # Exibir resultado
    st.subheader("Resultado")
    st.metric("Probabilidade de conter RWE", f"{prob*100:.1f}%")
    st.metric("Classificação", "RWE" if pred==1 else "Não-RWE")

    # Exportar
    output = df.copy()
    output["probabilidade"] = prob
    output["predicao"] = pred

    st.download_button(
        label="📁 Baixar resultado em Excel",
        data=output.to_excel(index=False),
        file_name="resultado_RWE.xlsx"
    )

    st.success("Análise concluída!")
