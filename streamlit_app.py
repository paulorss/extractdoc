# -*- coding: utf-8 -*-
"""
Aplicativo Streamlit para OCR de Documentos (RG, CNH, etc.)
Permite upload de PDF, PNG, JPG, extrai texto com Tesseract OCR,
e utiliza a API do Google Gemini para extrair dados estruturados.
"""

# 1. Imports necessários
import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes # Requer 'poppler-utils' no ambiente
import io
import os
import re
import json # Para processar a resposta da API

# Tenta importar a biblioteca do Google Gemini
try:
    import google.generativeai as genai
except ImportError:
    st.error("Biblioteca 'google-generativeai' não encontrada. Instale-a com 'pip install google-generativeai' e adicione ao requirements.txt.")
    # Adiciona um placeholder para evitar erros posteriores se a importação falhar
    genai = None

# --- Configuração do Tesseract ---
# (Mesma configuração de antes)
# --- Fim da Configuração do Tesseract ---

# 2. Função para realizar OCR (sem alterações)
def perform_ocr(file_bytes, file_type):
    """
    Realiza OCR nos bytes de uma imagem ou PDF.
    (Função original sem modificações)
    """
    # ... (código da função perform_ocr inalterado) ...
    extracted_text = ""
    display_image = None
    images_to_process = []

    try:
        # Processa imagens PNG ou JPEG
        if file_type in ['image/png', 'image/jpeg', 'image/jpg']:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images_to_process.append(image)
            display_image = image

        # Processa arquivos PDF
        elif file_type == 'application/pdf':
            st.info("Convertendo PDF para imagens...")
            pdf_images = convert_from_bytes(file_bytes, dpi=300)
            if pdf_images:
                images_to_process.extend(pdf_images)
                display_image = pdf_images[0]
            else:
                st.warning("Não foi possível extrair imagens do PDF.")
                return None, None
        else:
            st.error(f"Tipo de arquivo não suportado: {file_type}")
            return None, None

        if not images_to_process:
             st.warning("Nenhuma imagem encontrada ou extraída.")
             return display_image, ""

        # Realiza OCR
        st.info(f"Realizando OCR em {len(images_to_process)} imagem(ns)...")
        full_text_list = []
        for i, img in enumerate(images_to_process):
             try:
                 text = pytesseract.image_to_string(img, lang='por')
                 if len(images_to_process) > 1:
                     full_text_list.append(f"--- Página {i+1} ---\n{text}")
                 else:
                     full_text_list.append(text)
             except pytesseract.TesseractError as tess_err:
                 st.error(f"Erro do Tesseract na imagem {i+1}: {tess_err}")
                 full_text_list.append(f"--- Página {i+1}: Erro no OCR ---")
             except Exception as ocr_err:
                 st.error(f"Erro inesperado durante o OCR na imagem {i+1}: {ocr_err}")
                 full_text_list.append(f"--- Página {i+1}: Erro no OCR ---")

        extracted_text = "\n\n".join(full_text_list)
        return display_image, extracted_text.strip()

    # Tratamento de erros (sem alterações)
    except ImportError as import_err:
        if 'pdf2image' in str(import_err) or 'poppler' in str(import_err):
             st.error("Erro: A biblioteca 'pdf2image' ou sua dependência 'poppler' não foi encontrada.")
             st.info("Para rodar localmente, instale o Poppler. Para deploy no Streamlit Cloud, adicione 'poppler-utils' ao seu 'packages.txt'.")
        else:
             st.error(f"Erro de importação: {import_err}")
        return None, None
    except pytesseract.TesseractNotFoundError:
        st.error("Erro Crítico: O executável do Tesseract OCR não foi encontrado.")
        st.info("No Streamlit Cloud, adicione 'tesseract-ocr' e 'tesseract-ocr-por' ao seu 'packages.txt'.")
        return None, None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante o processamento OCR: {e}")
        return display_image if 'display_image' in locals() else None, None


# 3. Função para Análise Estruturada com API Gemini
def analyze_text_with_ai(text, api_key):
    """
    Analisa o texto OCR usando a API do Google Gemini para extrair dados.

    Args:
        text (str): O texto extraído pelo OCR.
        api_key (str): A chave da API do Google Gemini fornecida pelo usuário.

    Returns:
        dict: Um dicionário com os dados estruturados encontrados ou um erro.
    """
    if not genai:
         return {"Erro": "Biblioteca 'google-generativeai' não está instalada ou não pôde ser importada."}

    if not api_key:
        return {"Erro": "Chave da API do Gemini não fornecida."}

    if not text or not isinstance(text, str):
        return {"Erro": "Texto de entrada inválido para análise."}

    # Configura a API Key
    try:
        genai.configure(api_key=api_key)
    except Exception as config_err:
        st.error(f"Erro ao configurar a API Key do Gemini: {config_err}")
        return {"Erro": f"Falha na configuração da API Key: {config_err}"}

    # --- Prompt para o LLM ---
    # (O mesmo prompt de antes, instruindo a retornar JSON)
    prompt = f"""
    Analise o seguinte texto extraído de um documento de identidade brasileiro (como RG ou CNH)
    e retorne as informações estruturadas **estritamente em formato JSON**. Procure pelos seguintes campos:
    - data_nascimento (Formato DD/MM/AAAA)
    - local_nascimento (Cidade e UF, ex: Goiânia-GO)
    - doc_identidade (Número do RG ou CNH)
    - orgao_emissor (Órgão emissor do documento, ex: SSP/GO)
    - cpf (Número do CPF, formato XXX.XXX.XXX-XX)
    - nacionalidade (ex: Brasileira)
    - num_registro (Número de registro do documento, pode ser igual ao doc_identidade ou um campo separado)
    - filiacao_mae (Nome completo da mãe)
    - filiacao_pai (Nome completo do pai)

    Se um campo não for encontrado, retorne null ou omita a chave correspondente no JSON.
    **Responda APENAS com o objeto JSON, sem nenhum texto adicional antes ou depois.**

    Texto para análise:
    ---
    {text}
    ---

    JSON esperado:
    """

    # Configurações de geração (opcional, pode ajustar)
    generation_config = {
        "temperature": 0.2, # Baixa temperatura para respostas mais determinísticas
        "top_p": 1,
        "top_k": 1,
        # "response_mime_type": "application/json", # Funciona melhor com modelos mais recentes (ex: gemini-1.5-pro)
    }

    # Cria o modelo
    model = genai.GenerativeModel(
        model_name="gemini-pro", # Modelo padrão robusto
        generation_config=generation_config
        # safety_settings=... # Pode adicionar configurações de segurança se necessário
        )

    # Chama a API
    try:
        response = model.generate_content(prompt)

        # Tenta extrair e limpar a resposta JSON
        response_text = response.text
        # Remove possíveis blocos de código markdown ```json ... ```
        response_text = re.sub(r'^```json\s*', '', response_text.strip(), flags=re.IGNORECASE)
        response_text = re.sub(r'\s*```$', '', response_text.strip())

        # Tenta decodificar o JSON
        extracted_data = json.loads(response_text)
        return extracted_data

    except json.JSONDecodeError as json_err:
        st.error(f"Erro ao decodificar a resposta JSON da API: {json_err}")
        st.text("Resposta recebida da API:")
        st.code(response.text, language=None) # Mostra a resposta bruta
        return {"Erro": f"Falha ao processar JSON da API. Resposta: {response.text}"}
    except Exception as api_err:
        st.error(f"Erro durante a chamada à API do Gemini: {api_err}")
        # Tenta capturar informações de erro da resposta, se disponíveis
        error_details = getattr(response, 'prompt_feedback', str(api_err))
        return {"Erro": f"Erro na API Gemini: {error_details}"}


# 4. Interface do Aplicativo Streamlit (Atualizada)
st.set_page_config(layout="wide", page_title="OCR e Análise Gemini")

# --- Sidebar ---
st.sidebar.title("Configurações")

# Entrada da API Key
st.sidebar.subheader("Chave da API Google Gemini")
api_key_input = st.sidebar.text_input(
    "Insira sua API Key:",
    type="password",
    help="Sua chave da API do Google Gemini."
)

st.sidebar.warning(
    """
    ⚠️ **Aviso de Segurança:** Inserir a chave aqui é **inseguro** para apps compartilhados.
    Para deploy no Streamlit Cloud, use os **Secrets**!
    """
)
st.sidebar.markdown("[Obtenha uma chave de API aqui](https://aistudio.google.com/app/apikey)")


# Instruções de Deploy Atualizadas
st.sidebar.title("Notas de Deploy (Streamlit Cloud)")
st.sidebar.info(
    """
    Para fazer o deploy deste aplicativo no Streamlit Cloud:

    1.  **Crie um arquivo `requirements.txt`:**
        ```
        streamlit
        pillow
        pytesseract
        pdf2image
        google-generativeai
        ```

    2.  **Crie um arquivo `packages.txt`:**
        ```
        tesseract-ocr
        tesseract-ocr-por
        poppler-utils
        ```

    3.  **Configure a API Key (RECOMENDADO):**
        * Adicione sua chave da API do Google Gemini aos **Secrets** do seu aplicativo no Streamlit Cloud com o nome `GOOGLE_API_KEY`.
        * No código, você pode acessá-la com `st.secrets["GOOGLE_API_KEY"]` em vez de usar o input manual. **Adapte o código se for usar Secrets!**

    4.  Faça o upload dos arquivos para um repositório GitHub e conecte ao Streamlit Cloud.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido com Streamlit, Tesseract OCR e Google Gemini.")

# --- Área Principal ---
st.title("🔍 Aplicativo OCR com Análise via Google Gemini")
st.markdown("Faça upload de um arquivo (`PDF`, `PNG`, `JPG`), extraia o texto e use a API Gemini para obter dados estruturados.")

# Componente de upload de arquivo
uploaded_file = st.file_uploader(
    "Selecione o arquivo do documento",
    type=['pdf', 'png', 'jpg', 'jpeg'],
    help="Arraste e solte ou clique para selecionar."
)

# Variáveis de estado da sessão
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = None
if 'structured_data' not in st.session_state:
    st.session_state.structured_data = None

# Processamento após o upload
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    st.write("---")
    st.write(f"Arquivo carregado: **{file_name}** ({file_type})")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Visualização Prévia")
        # ... (código de visualização inalterado) ...
        display_image_ocr = None
        try:
            if file_type in ['image/png', 'image/jpeg', 'image/jpg']:
                image = Image.open(io.BytesIO(file_bytes))
                st.image(image, caption='Imagem Carregada', use_column_width='auto')
                display_image_ocr = image
            elif file_type == 'application/pdf':
                st.info("Exibindo a primeira página do PDF...")
                preview_images = convert_from_bytes(file_bytes, dpi=150, first_page=1, last_page=1)
                if preview_images:
                    st.image(preview_images[0], caption='Primeira Página do PDF', use_column_width='auto')
                else:
                    st.warning("Não foi possível gerar a visualização do PDF.")
            else:
                st.warning("Visualização não disponível.")
        except ImportError:
             st.error("Erro ao gerar visualização do PDF: 'poppler' não encontrado.")
        except Exception as e:
            st.error(f"Erro ao carregar visualização: {e}")


    with col2:
        st.subheader("Ações")
        # Botão OCR
        if st.button("1. Extrair Texto (OCR)", key="ocr_button"):
            st.session_state.ocr_text = None
            st.session_state.structured_data = None
            with st.spinner('Processando OCR...'):
                _, ocr_result = perform_ocr(file_bytes, file_type)
                if ocr_result is not None:
                     st.session_state.ocr_text = ocr_result
                     st.success("OCR Concluído!")
                else:
                     st.error("Falha no OCR.")

        # Botão Análise Gemini (requer OCR e API Key)
        analysis_possible = st.session_state.ocr_text and api_key_input
        analyze_button_disabled = not analysis_possible

        if st.button("2. Analisar Dados (Gemini API)", key="analyze_button", disabled=analyze_button_disabled):
            if not api_key_input:
                 st.warning("Por favor, insira sua chave da API Gemini na barra lateral.")
            elif not st.session_state.ocr_text:
                 st.warning("Execute o OCR primeiro (Botão 1).")
            else:
                 st.session_state.structured_data = None
                 with st.spinner("Chamando API Gemini para análise..."):
                     # Chama a função de análise com a chave fornecida
                     analysis_result = analyze_text_with_ai(st.session_state.ocr_text, api_key_input)
                     st.session_state.structured_data = analysis_result
                     if "Erro" not in analysis_result:
                         st.success("Análise com Gemini concluída!")
                     else:
                         # Erros já são mostrados dentro da função analyze_text_with_ai
                         st.error("Falha na análise com Gemini. Verifique os erros acima e sua API Key.")
        elif not analyze_button_disabled and not st.session_state.ocr_text:
             st.info("Execute o passo 1 (OCR) primeiro.")
        elif not analyze_button_disabled and not api_key_input:
             st.info("Insira sua API Key na barra lateral para habilitar a análise.")


    # Exibição dos resultados
    st.write("---")

    if st.session_state.ocr_text:
        st.subheader("Texto Extraído via OCR")
        st.text_area("Resultado do OCR:", st.session_state.ocr_text, height=250)

    if st.session_state.structured_data:
        st.subheader("Dados Estruturados (Análise Gemini API)")

        if "Erro" in st.session_state.structured_data:
             st.error(f"Erro na análise: {st.session_state.structured_data['Erro']}")
        else:
            data_map = {
                "data_nascimento": "Data de Nascimento",
                "local_nascimento": "Local de Nascimento",
                "doc_identidade": "Documento de Identidade",
                "orgao_emissor": "Órgão Emissor",
                "cpf": "CPF",
                "nacionalidade": "Nacionalidade",
                "num_registro": "Número de Registro",
                "filiacao_mae": "Filiação (Mãe)",
                "filiacao_pai": "Filiação (Pai)"
            }
            found_count = 0
            for key, value in st.session_state.structured_data.items():
                display_name = data_map.get(key, key.replace("_", " ").title())
                if value: # Só mostra se tiver valor e não for explicitamente null/None
                    st.info(f"**{display_name}:** {value}")
                    found_count += 1

            if found_count == 0:
                 st.warning("A API Gemini não retornou nenhum dos campos esperados com valor preenchido.")

            # Verifica campos que a API pode não ter retornado
            expected_keys = data_map.keys()
            returned_keys = st.session_state.structured_data.keys()
            missing_keys = [data_map[k] for k in expected_keys if k not in returned_keys]
            if missing_keys:
                st.markdown(f"**Campos não retornados pela API:** {', '.join(missing_keys)}")


else:
    st.info("Aguardando o upload de um arquivo PDF, PNG ou JPG.")

st.write("---")
st.caption("Observação: A precisão do OCR e da análise depende da qualidade do documento e da resposta da API Gemini.")

