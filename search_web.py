from dashvector import Client
from embedding_web import generate_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi

def calculate_tfidf_similarities(question, candidate_news):
    """
    计算问题和候选新闻的TF-IDF相似度。
    :param question: 问题文本
    :param candidate_news: 候选新闻文本列表
    :return: 相似度数组
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(candidate_news + [question])
    question_vector_tfidf = tfidf_matrix[-1]
    candidate_vectors_tfidf = tfidf_matrix[:-1]
    return cosine_similarity(question_vector_tfidf, candidate_vectors_tfidf).flatten()

def calculate_bm25_scores(question, candidate_news):
    """
    计算BM25评分。
    :param question: 问题文本
    :param candidate_news: 候选新闻文本列表
    :return: BM25评分数组
    """
    tokenized_candidates = [doc.split(" ") for doc in candidate_news]
    bm25 = BM25Okapi(tokenized_candidates)
    return bm25.get_scores(question.split(" "))

def search_relevant_news(question, api_key, endpoint_key,collection_name):
    """
    根据问题搜索相关新闻。
    :param question: 问题文本
    :return: 最匹配的新闻文本
    """
    # 初始化 dashvector client
    client = Client(
        api_key=api_key,
        endpoint=endpoint_key
    )

    # 获取刚刚存入的集合
    collection = client.get(collection_name)
    assert collection

    # 向量检索：指定 topk = 3
    initial_topk = 3
    rsp = collection.query(generate_embeddings(question), output_fields=['raw'], topk=initial_topk)
    assert rsp

    # 提取初步候选集的新闻内容
    candidate_news = [item.fields['raw'] for item in rsp.output]
    # 向量相似度
    vector_search_similarities = [item.score for item in rsp.output]  # 直接使用库返回的相似度分数

    # 使用TF-IDF和BM25计算文本相似度
    tfidf_similarities = calculate_tfidf_similarities(question, candidate_news)
    bm25_scores = calculate_bm25_scores(question, candidate_news)

    # 综合排序：结合向量检索、TF-IDF和BM25的结果
    combined_scores = 0.9 * np.array(vector_search_similarities) + 0.05 * np.array(tfidf_similarities) + 0.05 * np.array(bm25_scores)
    best_match_index = np.argmax(combined_scores)
    best_match_news = candidate_news[best_match_index]

    return best_match_news

