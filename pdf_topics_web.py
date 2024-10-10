import os
import PyPDF2
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import docx
import pandas as pd
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from docx import Document
import json
from transformers import AutoTokenizer, BertTokenizer
import jieba
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 使用中文BERT分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def extract_pdf_topics(pdf_folder_path, num_topics=12):
    pdf_topics = {}
    for file_name in os.listdir(pdf_folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, file_name)
            text = load_pdf(open(pdf_path, 'rb'))
            words = jieba.lcut(text)
            word_counts = Counter(words)
            common_words = word_counts.most_common(num_topics)
            topics = [word for word, _ in common_words if len(word) > 1]  # 过滤掉单字词
            pdf_topics[file_name] = topics
    return pdf_topics

def draw_knowledge_graph(topics, title):
    G = nx.Graph()
    for topic in topics:
        G.add_node(topic)
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            G.add_edge(topics[i], topics[j])
    plt.figure(figsize=(10, 7))
    nx.draw(G, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, font_weight='bold')
    plt.title(title)
    st.pyplot(plt)

def load_document(file):
    if file.type == "application/pdf":
        return load_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return load_docx(file)
    elif file.type == "text/plain":
        return load_txt(file)
    else:
        st.error("Unsupported file type")
        return None

def load_pdf(file):
    output_string = StringIO()
    extract_text_to_fp(file, output_string, laparams=LAParams(), output_type='text', codec='utf-8')
    return output_string.getvalue()

def load_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def load_txt(file):
    return file.getvalue().decode("utf-8")

def split_text_by_structure(text, chunk_size=1000, chunk_overlap=200):
    # 使用正则表达式匹配中文法律文档的常见结构
    pattern = r'(第[一二三四五六七八九十百千]+[章节条]|\d+\.)'
    splits = re.split(pattern, text)
    
    # 重新组合分割后的文本，确保每个chunk都以结构标记开头
    chunks = []
    current_chunk = ""
    for i in range(0, len(splits), 2):
        if i+1 < len(splits):
            section = splits[i] + splits[i+1]
        else:
            section = splits[i]
        
        if len(current_chunk) + len(section) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = section
        else:
            current_chunk += section
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def extract_structure(text):
    lines = text.split('\n')
    structure = []
    current_level = 0
    for line in lines:
        if line.strip():
            # 匹配中文法律文档的常见结构
            if re.match(r'第[一二三四五六七八九十百千]+章', line):
                level = 1
            elif re.match(r'第[一二三四五六七八九十百千]+节', line):
                level = 2
            elif re.match(r'第[一二三四五六七八九十百千]+条', line):
                level = 3
            else:
                level = current_level
            
            content = line.strip()
            structure.append({"level": level, "content": content})
            current_level = level
    return structure

def visualize_text_processing(original_text, chunks, structure):
    st.subheader("文本处理可视化")
    
    # 显示原始文本
    st.write("原始文本")
    st.text_area("", value=original_text[:1000] + "..." if len(original_text) > 1000 else original_text, height=200, disabled=True)
    
    # 显示文本分割结果
    st.write("文本分割结果")
    chunk_lengths = [len(chunk) for chunk in chunks]
    df = pd.DataFrame({"Chunk": range(1, len(chunks) + 1), "Character Count": chunk_lengths})
    st.bar_chart(df.set_index("Chunk"))
    
    # 显示词频统计
    st.write("词频统计")
    words = jieba.lcut(original_text)
    word_freq = pd.Series(words).value_counts().head(20)
    st.bar_chart(word_freq)
    
    # 显示文档结构
    st.write("文档结构")
    st.json(json.dumps(structure, indent=2, ensure_ascii=False))

def process_document(file):
    text = load_document(file)
    if text:
        chunks = split_text_by_structure(text)
        structure = extract_structure(text)
        visualize_text_processing(text, chunks, structure)
        return text, chunks, structure
    return None, None, None

# 主函数
def main():
    st.title("法律文档处理和可视化系统")
    
    uploaded_file = st.file_uploader("选择一个文件", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        text, chunks, structure = process_document(uploaded_file)
        if text:
            st.success("文件处理成功！")
            
            # 显示处理后的文本块
            if st.checkbox("显示处理后的文本块"):
                for i, chunk in enumerate(chunks):
                    st.write(f"块 {i+1}:")
                    st.write(chunk)
                    st.write("---")
            
            # 显示提取的结构
            if st.checkbox("显示提取的文档结构"):
                st.json(structure)

if __name__ == "__main__":
    main()