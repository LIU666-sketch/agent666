from dashvector import Client
from embedding_web import generate_embeddings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi

from transformers import AutoTokenizer, AutoModel
import torch
from googletrans import Translator
from concurrent.futures import ThreadPoolExecutor

# 初始化翻译器
translator = Translator()

# 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')

def generate_embeddings_bert(texts):
    """
    使用BERT生成文本的嵌入向量。
    :param texts: 文本列表
    :return: 嵌入向量数组
    """
    # 将文本转换为tokens
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # 提取CLS token的输出作为嵌入
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings


def translate_texts(texts, dest='en', max_workers=5):
    """
    将文本列表翻译成指定语言，并进行错误处理和并发限制。
    :param texts: 文本列表
    :param dest: 目标语言，默认为英语
    :param max_workers: 最大并发数，默认为5
    :return: 翻译后的文本列表
    """

    def safe_translate(text):
        try:
            return translator.translate(text, dest=dest).text
        except Exception as e:
            print(f"Error translating text: {text[:30]}... - {e}")
            return text  # 返回原文本以避免数据丢失

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        translations = list(executor.map(safe_translate, texts))

    return translations


def calculate_cosine_similarities(question_embedding, candidate_news_embeddings):
    """
    计算问题嵌入向量与候选新闻嵌入向量之间的余弦相似度。
    :param question_embedding: 问题的嵌入向量
    :param candidate_news_embeddings: 候选新闻的嵌入向量数组
    :return: 相似度数组
    """
    return cosine_similarity([question_embedding], candidate_news_embeddings).flatten()

def calculate_tfidf_similarities(question_translated, candidate_news_translated):
    """
    计算问题和候选新闻的TF-IDF相似度。
    :param question_translated: 翻译后的问题文本
    :param candidate_news_translated: 翻译后的候选新闻文本列表
    :return: 相似度数组
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(candidate_news_translated + [question_translated])
    question_vector_tfidf = tfidf_matrix[-1]
    candidate_vectors_tfidf = tfidf_matrix[:-1]
    return cosine_similarity(question_vector_tfidf, candidate_vectors_tfidf).flatten()

def calculate_bm25_scores(question_translated, candidate_news_translated):
    """
    计算BM25评分。
    :param question_translated: 翻译后的问题文本
    :param candidate_news_translated: 翻译后的候选新闻文本列表
    :return: BM25评分数组
    """
    tokenized_candidates = [doc.split(" ") for doc in candidate_news_translated]
    bm25 = BM25Okapi(tokenized_candidates)
    return bm25.get_scores(question_translated.split(" "))

def search_relevant_news(question):
    """
    根据问题搜索相关新闻。
    :param question: 问题文本
    :return: 最匹配的新闻文本
    """
    # 初始化 dashvector client
    client = Client(
        api_key='sk-16zRAK4FZsfrM49D25l3dbnhT3T1d604ACD1B348E11EF853FF64DB3CBA72B',
        endpoint='vrs-cn-em93syx0600020.dashvector.cn-hangzhou.aliyuncs.com'
    )

    # 获取刚刚存入的集合
    collection = client.get('news_embeddings')
    assert collection

    # 向量检索：指定 topk = 10
    initial_topk = 3
    rsp = collection.query(generate_embeddings(question), output_fields=['raw'], topk=initial_topk)
    assert rsp

    # 提取初步候选集的新闻内容
    candidate_news = [item.fields['raw'] for item in rsp.output]

    # 将候选新闻和问题翻译成英语
    candidate_news_translated = translate_texts(candidate_news)
    question_translated = translate_texts([question])[0]

    # 使用BERT生成嵌入
    candidate_news_embeddings = generate_embeddings_bert(candidate_news_translated)
    question_embedding = generate_embeddings_bert([question_translated])[0]

    # 计算相似度
    cosine_similarities = calculate_cosine_similarities(question_embedding, candidate_news_embeddings)
    tfidf_similarities = calculate_tfidf_similarities(question_translated, candidate_news_translated)
    bm25_scores = calculate_bm25_scores(question_translated, candidate_news_translated)

    # 综合排序：结合BERT、TF-IDF和BM25的结果
    combined_scores = 0.4 * cosine_similarities + 0.3 * tfidf_similarities + 0.3 * bm25_scores
    best_match_index = np.argmax(combined_scores)
    best_match_news = candidate_news[best_match_index]

    return best_match_news
