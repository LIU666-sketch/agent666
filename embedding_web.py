import os
import fitz  # PyMuPDF
import dashscope
import streamlit as st
from dashscope import TextEmbedding
from dashvector import Client, Doc
import re
from tqdm import tqdm

MAX_INPUT_LENGTH = 2048
MAX_BATCH_SIZE = 25

def list_pdf_files(pdf_folder_path):
    pdf_files = []
    if os.path.isdir(pdf_folder_path):
        for file_name in os.listdir(pdf_folder_path):
            if file_name.lower().endswith(".pdf"):
                pdf_files.append(file_name)
    return pdf_files


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Ensure the page is properly loaded
            text += page.get_text("text")  # Specify 'text' to get plain text
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def clean_text(text):
    """Clean the extracted text."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def split_text(text, max_length=MAX_INPUT_LENGTH):
    """Split text into chunks of max_length."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def prepare_data(path, max_batch_size=MAX_BATCH_SIZE):
    batch_docs = []
    files = [file for file in os.listdir(path) if file.endswith('.pdf')]

    for file in tqdm(files, desc="Processing PDFs"):
        file_path = os.path.join(path, file)
        text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(text)
        chunks = split_text(cleaned_text)
        for chunk in chunks:
            batch_docs.append(chunk)
            if len(batch_docs) == max_batch_size:
                yield batch_docs
                batch_docs = []
    if batch_docs:
        yield batch_docs

def generate_embeddings(news):
    try:
        rsp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v2,
            input=news
        )
        if rsp:
            print(f"API response: {rsp}")
        if rsp and rsp.output and 'embeddings' in rsp.output:
            embeddings = [record['embedding'] for record in rsp.output['embeddings']]
            return embeddings if isinstance(news, list) else embeddings[0]
        else:
            print("Error: No embeddings found in the response.")
            return None
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def vectorize_and_store(api_key, endpoint_api_key, endpoint, pdf_folder_path, collection_name):
    dashscope.api_key = api_key

    # 初始化 dashvector client
    client = Client(
        api_key=endpoint_api_key,
        endpoint=endpoint
    )

    # 创建集合：指定集合名称和向量维度, text_embedding_v2 模型产生的向量统一为 1536 维
    rsp = client.create(collection_name, 1536)
    assert rsp

    # 加载语料
    id = 0
    collection = client.get(collection_name)

    data_batches = list(prepare_data(pdf_folder_path))
    num_batches = len(data_batches)

    progress_bar = st.progress(0)

    for i, news_batch in enumerate(data_batches):
        ids = [id + j for j, _ in enumerate(news_batch)]
        id += len(news_batch)

        vectors = generate_embeddings(news_batch)
        if vectors is None:
            print("Skipping batch due to embedding generation error.")
            continue

        # 写入 dashvector 构建索引
        rsp = collection.upsert(
            [
                Doc(id=str(doc_id), vector=vector, fields={"raw": doc})
                for doc_id, vector, doc in zip(ids, vectors, news_batch)
            ]
        )
        assert rsp

        progress = (i + 1) / num_batches
        progress_bar.progress(progress)

    return "All files processed and uploaded successfully."






