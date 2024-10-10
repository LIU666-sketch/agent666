import os
import PyPDF2
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def extract_pdf_topics(pdf_folder_path, num_topics=12):
    stop_words = set(stopwords.words('english'))
    pdf_topics = {}

    for file_name in os.listdir(pdf_folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, file_name)
            text = extract_text_from_pdf(pdf_path)
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]
            word_counts = Counter(filtered_words)
            common_words = word_counts.most_common(num_topics)
            topics = [word for word, _ in common_words]
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
