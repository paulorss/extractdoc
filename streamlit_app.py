# -*- coding: utf-8 -*-
"""
Aplicativo Streamlit para OCR de Documentos (RG, CNH, etc.)
Permite upload de PDF, PNG, JPG e extrai texto em português.
"""

# 1. Imports necessários
import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes # Requer 'poppler-utils' no ambiente
import io
import os

# --- Configuração do Tesseract (Importante para teste local) ---
# No Streamlit Cloud, geralmente não é necessário se 'tesseract-ocr' e
# 'tesseract-ocr-por' estiverem em packages.txt.
# Descomente e ajuste os caminhos se estiver rodando localmente e o Tesseract
# não estiver no PATH do sistema.
# Exemplo Linux/macOS:
# pytesseract.tesseract_cmd = '/usr/local/bin/tesseract' ou '/usr/bin/tesseract'
# Exemplo Windows:
# pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Para dados de linguagem (se não encontrados automaticamente):
# tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'
# --- Fim da Configuração do Tesseract ---

# 2. Função para realizar OCR
def perform_ocr(file_bytes, file_type):
    """
    Realiza OCR nos bytes de uma imagem ou PDF.

    Args:
        file_bytes (bytes): Os bytes do arquivo carregado.
        file_type (str): O tipo MIME do arquivo (ex: 'image/png', 'application/pdf').

    Returns:
        tuple: Uma tupla contendo (PIL.Image ou None, str ou None).
               A imagem (primeira página do PDF) para exibição e o texto extraído.
               Retorna (None, None) em caso de erro grave.
    """
    extracted_text = ""
    display_image = None
    images_to_process = []

    try:
        # Processa imagens PNG ou JPEG
        if file_type in ['image/png', 'image/jpeg', 'image/jpg']:
            image = Image.open(io.BytesIO(file_bytes))
            # Converte para RGB para garantir compatibilidade com Tesseract
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images_to_process.append(image)
            display_image = image # Define a imagem para exibição

        # Processa arquivos PDF
        elif file_type == 'application/pdf':
            st.info("Convertendo PDF para imagens (pode levar um momento)...")
            # Converte todas as páginas do PDF para imagens PIL
            # Requer que 'poppler-utils' esteja instalado no sistema (via packages.txt no Cloud)
            pdf_images = convert_from_bytes(file_bytes, dpi=300) # DPI mais alto para melhor OCR
            if pdf_images:
                images_to_process.extend(pdf_images)
                display_image = pdf_images[0] # Define a primeira página para exibição
            else:
                st.warning("Não foi possível extrair imagens do PDF.")
                return None, None

        # Tipo de arquivo não suportado
        else:
            st.error(f"Tipo de arquivo não suportado: {file_type}")
            return None, None

        # Verifica se alguma imagem foi carregada/extraída
        if not images_to_process:
             st.warning("Nenhuma imagem encontrada ou extraída do arquivo.")
             return display_image, "" # Retorna imagem de display (se houver) e texto vazio

        # Realiza OCR em cada imagem extraída
        st.info(f"Realizando OCR em {len(images_to_process)} imagem(ns)...")
        full_text_list = []
        for i, img in enumerate(images_to_process):
             # Usa o modelo de linguagem 'por' (Português)
             try:
                 text = pytesseract.image_to_string(img, lang='por')
                 if len(images_to_process) > 1:
                     full_text_list.append(f"--- Página {i+1} ---\n{text}")
                 else:
                     full_text_list.append(text)
             except pytesseract.TesseractError as tess_err:
                 st.error(f"Erro do Tesseract na imagem {i+1}: {tess_err}")
                 st.info("Verifique se o Tesseract e o pacote de linguagem 'por' estão instalados corretamente.")
                 full_text_list.append(f"--- Página {i+1}: Erro no OCR ---")
             except Exception as ocr_err:
                 st.error(f"Erro inesperado durante o OCR na imagem {i+1}: {ocr_err}")
                 full_text_list.append(f"--- Página {i+1}: Erro no OCR ---")


        extracted_text = "\n\n".join(full_text_list)
        return display_image, extracted_text.strip()

    # Tratamento de erros específicos
    except ImportError as import_err:
        if 'pdf2image' in str(import_err) or 'poppler' in str(import_err):
             st.error("Erro: A biblioteca 'pdf2image' ou sua dependência 'poppler' não foi encontrada.")
             st.info("Para rodar localmente, instale o Poppler (veja a documentação do pdf2image). Para deploy no Streamlit Cloud, adicione 'poppler-utils' ao seu arquivo 'packages.txt'.")
        else:
             st.error(f"Erro de importação: {import_err}")
        return None, None
    except pytesseract.TesseractNotFoundError:
        st.error("Erro Crítico: O executável do Tesseract OCR não foi encontrado.")
        st.info("Certifique-se de que o Tesseract está instalado e acessível no PATH do sistema. No Streamlit Cloud, adicione 'tesseract-ocr' e 'tesseract-ocr-por' ao seu 'packages.txt'.")
        return None, None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante o processamento: {e}")
        # Tenta retornar a imagem de display se ela foi carregada antes do erro
        return display_image if 'display_image' in locals() else None, None

# 3. Interface do Aplicativo Streamlit
st.set_page_config(layout="wide", page_title="OCR de Documentos")

# Sidebar com instruções de deploy
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
        ```
        *(Certifique-se de usar versões compatíveis se necessário)*

    2.  **Crie um arquivo `packages.txt`:**
        Este arquivo informa ao Streamlit Cloud quais pacotes do sistema (Debian) instalar.
        ```
        tesseract-ocr
        tesseract-ocr-por
        poppler-utils
        ```

    3.  Faça o upload do seu script Python (`app.py` ou similar), `requirements.txt`, e `packages.txt` para um repositório GitHub.
    4.  Conecte seu repositório ao Streamlit Cloud e faça o deploy.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido com [Streamlit](https://streamlit.io/) e [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).")


# Título principal
st.title("🔍 Aplicativo OCR para Documentos Brasileiros")
st.markdown("Faça upload de um arquivo de imagem (`PNG`, `JPG`) ou `PDF` contendo documentos como RG, CNH, etc., para extrair o texto.")

# Componente de upload de arquivo
uploaded_file = st.file_uploader(
    "Selecione o arquivo",
    type=['pdf', 'png', 'jpg', 'jpeg'],
    help="Arraste e solte ou clique para selecionar o arquivo."
)

# Processamento após o upload
if uploaded_file is not None:
    # Lê os bytes do arquivo para processamento
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    st.write("---")
    st.write(f"Arquivo carregado: **{file_name}** ({file_type})")

    # Layout em duas colunas
    col1, col2 = st.columns(2)

    extracted_text = None # Inicializa a variável de texto extraído

    with col1:
        st.subheader("Visualização Prévia")
        try:
            # Tenta exibir a imagem diretamente ou a primeira página do PDF
            if file_type in ['image/png', 'image/jpeg', 'image/jpg']:
                image = Image.open(io.BytesIO(file_bytes))
                st.image(image, caption='Imagem Carregada', use_column_width='auto')
            elif file_type == 'application/pdf':
                st.info("Exibindo a primeira página do PDF...")
                # Usa DPI menor para preview mais rápido
                preview_images = convert_from_bytes(file_bytes, dpi=150, first_page=1, last_page=1)
                if preview_images:
                    st.image(preview_images[0], caption='Primeira Página do PDF', use_column_width='auto')
                else:
                    st.warning("Não foi possível gerar a visualização do PDF.")
            else:
                st.warning("Visualização não disponível para este tipo de arquivo.")
        except ImportError:
             st.error("Erro ao gerar visualização do PDF: 'poppler' não encontrado. Verifique a instalação (packages.txt).")
        except Exception as e:
            st.error(f"Erro ao carregar visualização: {e}")

    with col2:
        st.subheader("Extração de Texto (OCR)")
        # Botão para iniciar o processo de OCR
        if st.button("✨ Extrair Texto do Documento"):
            with st.spinner('Processando OCR... Isso pode levar alguns segundos, especialmente para PDFs.'):
                # Chama a função principal de OCR
                _, ocr_result = perform_ocr(file_bytes, file_type)
                extracted_text = ocr_result # Armazena o resultado

            if extracted_text is not None and extracted_text:
                st.success("OCR concluído com sucesso!")
            elif extracted_text == "":
                 st.warning("O OCR foi concluído, mas nenhum texto foi detectado na imagem/documento.")
            else:
                # Mensagens de erro já são exibidas dentro de perform_ocr
                st.error("Falha no processo de OCR. Verifique as mensagens de erro acima.")

        # Exibe o texto extraído se o OCR foi executado
        if extracted_text is not None:
            st.text_area(
                "Texto Extraído:",
                extracted_text,
                height=400,
                help="O texto extraído do documento. Pode conter erros dependendo da qualidade da imagem."
            )
        elif 'ocr_result' not in locals(): # Se o botão ainda não foi clicado
             st.info("Clique no botão 'Extrair Texto do Documento' para iniciar o OCR.")


else:
    st.info("Aguardando o upload de um arquivo PDF, PNG ou JPG.")

st.write("---")
st.caption("Observação: A precisão do OCR depende da qualidade e resolução da imagem ou do PDF.")
