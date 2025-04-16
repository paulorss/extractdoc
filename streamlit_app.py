# -*- coding: utf-8 -*-
"""
Aplicativo Streamlit para OCR de Documentos (RG, CNH, etc.)
Permite upload de PDF, PNG, JPG, extrai texto com Tesseract OCR,
e utiliza a API do Google Gemini para extrair dados estruturados.
(v1.4 - Adicionado botão de cópia via st.code para dados estruturados)
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
    import google.api_core.exceptions # Para capturar erros específicos da API
except ImportError:
    st.error("Biblioteca 'google-generativeai' não encontrada. Instale-a com 'pip install google-generativeai' e adicione ao requirements.txt.")
    # Adiciona um placeholder para evitar erros posteriores se a importação falhar
    genai = None
    google = None # Placeholder para api_core

# --- Configuração do Tesseract ---
# (Mesma configuração de antes)
# --- Fim da Configuração do Tesseract ---

# 2. Função para realizar OCR (sem alterações significativas)
def perform_ocr(file_bytes, file_type):
    """
    Realiza OCR nos bytes de uma imagem ou PDF.
    (Função original com pequenas melhorias no tratamento de erro)
    """
    # ... (código da função perform_ocr como na v1.3) ...
    extracted_text = ""
    display_image = None
    images_to_process = []

    try:
        # Processa imagens PNG ou JPEG
        if file_type in ['image/png', 'image/jpeg', 'image/jpg']:
            image = Image.open(io.BytesIO(file_bytes))
            # Converte para RGB para evitar potenciais problemas com Tesseract
            if image.mode == 'RGBA':
                 image = image.convert('RGB')
            elif image.mode == 'P': # Paleta
                 image = image.convert('RGB')
            elif image.mode == 'L': # Grayscale
                 image = image.convert('RGB') # Tesseract geralmente prefere RGB ou Grayscale

            images_to_process.append(image)
            display_image = image

        # Processa arquivos PDF
        elif file_type == 'application/pdf':
            st.info("Convertendo PDF para imagens...")
            try:
                # Usa poppler_path se necessário em ambientes específicos (geralmente não no Cloud com packages.txt)
                # poppler_path = None # Defina se necessário
                pdf_images = convert_from_bytes(file_bytes, dpi=300)#, poppler_path=poppler_path)
                if pdf_images:
                    images_to_process.extend(pdf_images)
                    display_image = pdf_images[0]
                else:
                    st.warning("Não foi possível extrair imagens do PDF (pdf2image retornou lista vazia).")
                    return None, None
            except Exception as pdf_err:
                 if 'poppler' in str(pdf_err).lower() or 'unable to get page count' in str(pdf_err).lower():
                      st.error("Erro ao converter PDF: Dependência 'poppler' não encontrada ou não está no PATH.")
                      st.info("Verifique a instalação do 'poppler-utils' (via packages.txt no Streamlit Cloud).")
                 else:
                      st.error(f"Erro inesperado ao converter PDF: {pdf_err}")
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
                 # Configuração Tesseract: --psm 6 assume um bloco de texto uniforme.
                 # Para documentos com layout variado, --psm 3 (Auto Page Segmentation) ou --psm 4 (Assume single column) podem ser melhores. Testar!
                 custom_config = r'--oem 3 --psm 4 -l por' # Tentando PSM 4
                 text = pytesseract.image_to_string(img, config=custom_config)
                 if len(images_to_process) > 1:
                     full_text_list.append(f"--- Página {i+1} ---\n{text}")
                 else:
                     full_text_list.append(text)
             except pytesseract.TesseractNotFoundError:
                  st.error("Erro Crítico: Tesseract não encontrado. Verifique a instalação e o PATH.")
                  return None, None
             except pytesseract.TesseractError as tess_err:
                 st.error(f"Erro do Tesseract na imagem {i+1}: {tess_err}")
                 full_text_list.append(f"--- Página {i+1}: Erro no OCR ---")
             except Exception as ocr_err:
                 st.error(f"Erro inesperado durante o OCR na imagem {i+1}: {ocr_err}")
                 full_text_list.append(f"--- Página {i+1}: Erro no OCR ---")

        extracted_text = "\n\n".join(full_text_list)
        return display_image, extracted_text.strip()

    # Tratamento de erros gerais
    except ImportError as import_err:
        if 'pdf2image' in str(import_err) or 'poppler' in str(import_err):
             st.error("Erro: A biblioteca 'pdf2image' ou sua dependência 'poppler' não foi encontrada.")
             st.info("Para rodar localmente, instale o Poppler. Para deploy no Streamlit Cloud, adicione 'poppler-utils' ao seu 'packages.txt'.")
        else:
             st.error(f"Erro de importação não relacionado ao pdf2image: {import_err}")
        return None, None
    except pytesseract.TesseractNotFoundError:
        st.error("Erro Crítico: O executável do Tesseract OCR não foi encontrado.")
        st.info("Verifique a instalação do Tesseract e se ele está no PATH do sistema.")
        return None, None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante o processamento OCR: {e}")
        return display_image if 'display_image' in locals() else None, None


# 3. Função para Análise Estruturada com API Gemini (Usando configure/GenerativeModel)
def analyze_text_with_ai(text, api_key):
    """
    Analisa o texto OCR usando a API do Google Gemini para extrair dados,
    utilizando genai.configure() e genai.GenerativeModel().
    (Função como na v1.3)
    """
    # ... (código da função analyze_text_with_ai como na v1.3) ...
    if not genai or not google:
        return {"Erro": "Biblioteca 'google-generativeai' não está instalada ou não pôde ser importada."}

    if not api_key:
        return {"Erro": "Chave da API do Gemini não fornecida."}

    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return {"Erro": "Texto de entrada inválido ou muito curto para análise."}

    # --- Configura a API ---
    try:
        genai.configure(api_key=api_key)
    except Exception as config_err:
        st.error(f"Erro ao configurar a API Gemini: {config_err}")
        return {"Erro": f"Falha ao configurar API Gemini: {config_err}"}

    # --- Prompt para o LLM ---
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

    # --- Define o nome do modelo ---
    # Usando um modelo mais recente ou 'gemini-pro' como alternativa estável
    model_name = "gemini-1.5-flash-latest"
    # model_name = "gemini-pro" # Alternativa

    # --- Chama a API usando GenerativeModel ---
    try:
        st.info(f"Enviando solicitação para a API Gemini (modelo: {model_name})...")
        # Cria a instância do modelo AQUI, após configurar a API Key
        model = genai.GenerativeModel(model_name)
        # Adicionando um timeout simples para a chamada da API (ex: 60 segundos)
        # Nota: O timeout real pode depender da implementação da biblioteca
        response = model.generate_content(prompt)#, request_options={'timeout': 60}) # Timeout pode não ser suportado diretamente assim
        st.info("Resposta recebida da API Gemini.")

        # Extrai a resposta como texto
        response_text = ""
        if hasattr(response, 'text'):
             response_text = response.text
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             response_text = response.candidates[0].content.parts[0].text
        else:
             st.warning("Estrutura de resposta da API Gemini inesperada.")
             try:
                 st.json(response)
             except:
                 st.text(str(response))
             return {"Erro": "Estrutura de resposta da API inesperada."}

        # Limpa formatação markdown se presente
        response_text = re.sub(r'^```json\s*', '', response_text.strip(), flags=re.IGNORECASE)
        response_text = re.sub(r'\s*```$', '', response_text.strip())

        # Tenta decodificar o JSON
        extracted_data = json.loads(response_text)
        return extracted_data

    except json.JSONDecodeError as json_err:
        st.error(f"Erro ao decodificar a resposta JSON da API: {json_err}")
        st.text("Resposta recebida da API (pode não ser JSON válido):")
        st.code(response_text if 'response_text' in locals() and response_text else "Nenhuma resposta de texto capturada", language=None)
        return {"Erro": f"Falha ao processar JSON da API. Verifique a resposta acima."}
    except google.api_core.exceptions.GoogleAPIError as api_err:
        st.error(f"Erro na API Google durante a chamada: {api_err}")
        error_details = f"Código: {getattr(api_err, 'code', 'N/A')}, Mensagem: {getattr(api_err, 'message', str(api_err))}"
        if hasattr(api_err, 'details'):
            error_details += f", Detalhes: {api_err.details}"

        if isinstance(api_err, google.api_core.exceptions.NotFound):
            st.warning(f"Erro 404: O modelo '{model_name}' não foi encontrado ou não está acessível com sua API Key.")
        elif isinstance(api_err, google.api_core.exceptions.PermissionDenied):
            st.warning(f"Erro de Permissão (403): Verifique se sua API Key está ativa e tem permissão para usar o modelo '{model_name}'.")
        elif isinstance(api_err, google.api_core.exceptions.InvalidArgument):
            st.warning(f"Erro de Argumento Inválido (400): {api_err.message}. Verifique o prompt ou se o modelo suporta o tipo de conteúdo.")
            if 'model' in str(api_err).lower() and 'does not support' in str(api_err).lower():
                 st.info(f"O modelo '{model_name}' pode não suportar generateContent diretamente desta forma ou com este tipo de prompt. Considere testar com 'gemini-pro'.")
        elif isinstance(api_err, google.api_core.exceptions.ResourceExhausted):
            st.warning(f"Erro de Cota Excedida (429): Você pode ter excedido os limites de uso da API. Tente novamente mais tarde.")
        elif isinstance(api_err, google.api_core.exceptions.DeadlineExceeded):
             st.warning("Erro de Timeout (Deadline Exceeded): A solicitação para a API demorou muito para responder.")


        return {"Erro": f"Erro na API Google: {error_details}"}
    except Exception as generic_api_err:
        st.error(f"Erro genérico durante a chamada à API do Gemini: {generic_api_err}")
        error_details = str(generic_api_err)
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            error_details += f" | Feedback do Prompt: {response.prompt_feedback}"
        return {"Erro": f"Erro na API Gemini: {error_details}"}


# 4. Interface do Aplicativo Streamlit (UI Principal Atualizada)
st.set_page_config(layout="wide", page_title="OCR e Análise Gemini")

# --- Sidebar ---
st.sidebar.title("Configurações")
st.sidebar.subheader("Chave da API Google Gemini")
api_key_input = st.sidebar.text_input(
    "Insira sua API Key:",
    type="password",
    help="Sua chave da API do Google Gemini."



# --- Área Principal ---
st.title("🔍 Aplicativo OCR com Análise via Google Gemini")
st.markdown("Faça upload de um arquivo (`PDF`, `PNG`, `JPG`), extraia o texto e use a API Gemini para obter dados estruturados.")

uploaded_file = st.file_uploader(
    "Selecione o arquivo do documento",
    type=['pdf', 'png', 'jpg', 'jpeg'],
    help="Arraste e solte ou clique para selecionar.",
    key="file_uploader"
)

if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = None
if 'structured_data' not in st.session_state:
    st.session_state.structured_data = None
if 'last_uploaded_filename' not in st.session_state:
     st.session_state.last_uploaded_filename = None

if uploaded_file and uploaded_file.name != st.session_state.get('last_uploaded_filename'):
    st.session_state.ocr_text = None
    st.session_state.structured_data = None
    st.session_state.last_uploaded_filename = uploaded_file.name
    # st.info(f"Novo arquivo detectado: {uploaded_file.name}. Estados anteriores resetados.") # Opcional: Mostrar mensagem de reset


if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    st.write("---")
    st.write(f"Arquivo carregado: **{file_name}** ({file_type})")

    col1, col2 = st.columns([2, 1]) # Coluna da imagem maior

    with col1:
        st.subheader("Visualização Prévia")
        # ... (código de visualização inalterado da v1.3) ...
        try:
            if file_type in ['image/png', 'image/jpeg', 'image/jpg']:
                image = Image.open(io.BytesIO(file_bytes))
                st.image(image, caption='Imagem Carregada', use_column_width='auto')
            elif file_type == 'application/pdf':
                st.info("Exibindo a primeira página do PDF...")
                try:
                    preview_images = convert_from_bytes(file_bytes, dpi=150, first_page=1, last_page=1)
                    if preview_images:
                        st.image(preview_images[0], caption='Primeira Página do PDF', use_column_width='auto')
                    else:
                        st.warning("Não foi possível gerar a visualização do PDF (pdf2image retornou vazio).")
                except Exception as pdf_prev_err:
                     if 'poppler' in str(pdf_prev_err).lower():
                          st.error("Erro na visualização do PDF: 'poppler' não encontrado.")
                     else:
                          st.error(f"Erro ao gerar pré-visualização do PDF: {pdf_prev_err}")
            else:
                st.warning("Visualização não disponível para este tipo de arquivo.")
        except Exception as e:
            st.error(f"Erro inesperado ao carregar visualização: {e}")


    with col2:
        st.subheader("Ações")
        # Botão OCR
        if st.button("1. Extrair Texto (OCR)", key="ocr_button"):
            st.session_state.ocr_text = None
            st.session_state.structured_data = None
            with st.spinner('Processando OCR...'):
                _, ocr_result = perform_ocr(file_bytes, file_type)
                if ocr_result is not None:
                     if ocr_result.strip():
                          st.session_state.ocr_text = ocr_result
                          st.success("OCR Concluído!")
                     else:
                          st.warning("OCR concluído, mas nenhum texto foi detectado.")
                          st.session_state.ocr_text = ""
                else:
                     st.error("Falha no processo de OCR.")

        # Botão Análise Gemini
        analysis_possible = st.session_state.ocr_text is not None and api_key_input
        analyze_button_disabled = not analysis_possible

        if st.button("2. Analisar Dados (Gemini API)", key="analyze_button", disabled=analyze_button_disabled):
            if not api_key_input:
                 st.warning("Por favor, insira sua chave da API Gemini na barra lateral.")
            elif st.session_state.ocr_text is None:
                 st.warning("Execute o OCR primeiro (Botão 1).")
            else:
                 # Caso normal (inclui texto vazio, a função analyze_text_with_ai trata isso)
                 st.session_state.structured_data = None
                 with st.spinner("Chamando API Gemini para análise..."):
                     analysis_result = analyze_text_with_ai(st.session_state.ocr_text, api_key_input)
                     st.session_state.structured_data = analysis_result
                     if isinstance(analysis_result, dict) and "Erro" in analysis_result:
                          st.error("Falha na análise com Gemini. Verifique os erros acima e sua API Key.")
                     else:
                          st.success("Análise com Gemini concluída!")

        # Mensagens de ajuda para o botão desabilitado
        elif analyze_button_disabled:
             if not api_key_input:
                  st.info("Insira sua API Key na barra lateral para habilitar a análise.")
             elif st.session_state.ocr_text is None:
                  st.info("Execute o passo 1 (OCR) primeiro.")


    # Exibição dos resultados
    st.write("---")

    if st.session_state.ocr_text is not None:
        st.subheader("Texto Extraído via OCR")
        st.text_area(
             "Resultado do OCR:",
             st.session_state.ocr_text if st.session_state.ocr_text.strip() else "[Nenhum texto detectado pelo OCR]",
             height=250
        )

    # --- EXIBIÇÃO DOS DADOS ESTRUTURADOS (MODIFICADO) ---
    if st.session_state.structured_data is not None:
        st.subheader("Dados Estruturados (Análise Gemini API)")
        if isinstance(st.session_state.structured_data, dict) and "Erro" in st.session_state.structured_data:
             pass # Erro já tratado visualmente
        elif isinstance(st.session_state.structured_data, dict):
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
            st.write("Valores extraídos (clique no ícone para copiar):") # Adiciona um título
            for key, value in st.session_state.structured_data.items():
                display_name = data_map.get(key, key.replace("_", " ").title())
                # Mostra o valor mesmo que seja null/None ou vazio
                value_str = str(value) if value is not None else ""

                # Exibe o rótulo e o valor em um bloco de código (que geralmente tem botão de cópia)
                st.markdown(f"**{display_name}:**")
                st.code(value_str, language=None) # language=None para texto simples

                if value:
                     found_count += 1

            if found_count == 0 and not ("Erro" in st.session_state.structured_data):
                 st.warning("A API Gemini processou o texto, mas não retornou valores preenchidos para os campos esperados.")

            expected_keys = data_map.keys()
            returned_keys = st.session_state.structured_data.keys()
            missing_keys = [data_map[k] for k in expected_keys if k not in returned_keys]
            if missing_keys:
                st.markdown(f"**Campos não retornados pela API (chave ausente):** {', '.join(missing_keys)}")
        else:
            st.error("A resposta da análise da API não foi um dicionário JSON válido.")
            st.text("Resposta recebida:")
            st.code(str(st.session_state.structured_data), language=None)

else:
    st.info("Aguardando o upload de um arquivo PDF, PNG ou JPG.")

st.write("---")
st.caption("Observação: A precisão do OCR e da análise depende da qualidade do documento e da resposta da API Gemini.")
