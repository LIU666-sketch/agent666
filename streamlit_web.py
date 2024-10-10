import os
import streamlit as st
from http import HTTPStatus
from dashscope import Generation
import dashscope
from embedding_web import vectorize_and_store, convert_text, split_text
import time
from search_web import search_relevant_news
from pdf_topics_web import extract_pdf_topics, draw_knowledge_graph, load_document, visualize_text_processing, process_document
import speech_recognition as sr
import pyttsx3
import threading
from prompts import GovernmentAgentPrompts


# Define multi-round conversation function
def multi_round(messages, option, model, endpoint_api_key, endpoint_api_secret, collection_name):
    if option == "Default":
        response = Generation.call(
            model=model,
            messages=messages,
            result_format='message'
        )
        if response.status_code == HTTPStatus.OK:
            new_message = {
                'role': response.output.choices[0]['message']['role'],
                'content': response.output.choices[0]['message']['content']
            }
            messages.append(new_message)
            return messages, new_message
        else:
            st.error(f"Request id: {response.request_id}, Status code: {response.status_code}, "
                     f"error code: {response.code}, error message: {response.message}")
            return messages[:-1], None
    elif option == "RAG":
        question = messages[-1]["content"]
        best_match_news = search_relevant_news(question, endpoint_api_key, endpoint_api_secret, collection_name)
        response_message = answer_question(question, best_match_news, model)
        new_message = {
            'role': 'assistant',
            'content': response_message
        }
        messages.append(new_message)
        return messages, new_message


def answer_question(question, context, model):
    prompt = f'''请基于```内的内容回答问题。"
    ```
    {context}
    ```
    我的问题是：{question}。
    '''

    rsp = Generation.call(model=model,
                          prompt=prompt)
    if rsp.status_code == HTTPStatus.OK:
        return rsp.output.text.strip()
    else:
        st.error(f"Request id: {rsp.request_id}, Status code: {rsp.status_code}, "
                 f"error code: {rsp.code}, error message: {rsp.message}")
        return "Error"


# Real-time display function
def display_realtime_message(message_content, placeholder, role):
    full_text = ""
    for char in message_content:
        full_text += char
        with placeholder:
            st.chat_message(role).write(full_text)
        time.sleep(0.05)  # Control the display speed


# Sidebar design
def sidebar_configuration():
    st.sidebar.header("资料库管理")

    dashscope_api_key = st.sidebar.text_input("Dashscope API Key", key="chatbot_api_key", type="password")
    dashvector_api_key = st.sidebar.text_input("Dashvector API Key", key="chatbot_endpoint_api_key", type="password")
    dashvector_endpoint = st.sidebar.text_input("Dashvector Endpoint", key="chatbot_endpoint", type="password")
    # 添加链接到侧边栏
    st.sidebar.markdown("[Get a Dashscope API key](https://dashscope.console.aliyun.com/)")
    st.sidebar.markdown("[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")
    st.sidebar.markdown(
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")
    pdf_folder_path = st.sidebar.text_input("资料来源", key="pdf_folder_path")

    pdf_files = list_pdf_files(pdf_folder_path)
    if pdf_files:
        st.sidebar.subheader("资料 目录")
        for pdf_file in pdf_files:
            st.sidebar.markdown(f"- {pdf_file}")

    process_files_button = st.sidebar.button("解析 资料", key="process_button", help="处理选择的PDF文件",
                                             use_container_width=True)

    if process_files_button:
        st.session_state['show_modal'] = True

    if "chat_records" not in st.session_state:
        st.session_state["chat_records"] = []
        st.session_state["chat_titles"] = []

    if st.sidebar.button("新建聊天记录"):
        st.session_state["chat_records"].append([{"role": "assistant", "content": "您好！我有么能帮到您？"}])
        st.session_state["chat_titles"].append("聊天记录 " + str(len(st.session_state["chat_records"])))
        st.session_state["current_chat_index"] = len(st.session_state["chat_records"]) - 1

    for i, (record, title) in enumerate(zip(st.session_state["chat_records"], st.session_state["chat_titles"])):
        if st.sidebar.button(title, key=f"chat_record_{i}"):
            st.session_state["current_chat_index"] = i
            st.session_state["messages"] = record

    # 删除任务类型选择，使用默认值
    task_type = "通用问答"

    return dashscope_api_key, dashvector_api_key, dashvector_endpoint, pdf_folder_path, task_type


def list_pdf_files(pdf_folder_path):
    pdf_files = []
    if os.path.isdir(pdf_folder_path):
        for file_name in os.listdir(pdf_folder_path):
            if file_name.lower().endswith(".pdf"):
                pdf_files.append(file_name)
    return pdf_files


# Initialize conversation history
def initialize_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "您好！我有什么能帮到您？"}]
    if "current_chat_index" not in st.session_state:
        st.session_state["current_chat_index"] = 0
        st.session_state["chat_records"].append(st.session_state["messages"])
        st.session_state["chat_titles"] = ["聊天记录 1"]


# Display conversation history
def display_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.chat_message(msg["role"]).write(msg["content"])
        else:
            col1, col2, col3 = st.columns([1, 1, 3])
            with col3:
                st.chat_message(msg["role"]).write(msg["content"])


# 在文件开头添加以下导入语句
import speech_recognition as sr
import pyttsx3
import threading

# 在 main 函数之前添加以下函数

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("请说话...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="zh-CN")
        return text
    except sr.UnknownValueError:
        st.error("无法识别语音")
        return None
    except sr.RequestError:
        st.error("无法连接到语音识别服务")
        return None

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('voice', 'zh')
    engine.say(text)
    engine.runAndWait()

# Main application logic
def main():
    st.set_page_config(layout="wide")

    st.markdown(
        """
        <style>
        .small-selectbox select {
            width: 150px;
            font-size: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        st.markdown('<div class="small-selectbox">', unsafe_allow_html=True)
        option = st.selectbox("选择功能:", ("Default", "RAG"), key="function_select", index=0,
                              help="选择你需要的功能，Default——正常问答；RAG——知识问答")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="small-selectbox">', unsafe_allow_html=True)
        model = st.selectbox("选择模型:", [
            "qwen-turbo",
            "qwen-turbo-0624", "qwen-turbo-0206", "qwen-plus", "qwen-plus-0624", "qwen-plus-0206",
            "qwen-max", "qwen-max-0428", "qwen-max-0403", "qwen-max-0107", "qwen-max-1201", "qwen-max-longcontext",
            "qwen2-57b-a14b-instruct", "qwen2-72b-instruct", "qwen2-7b-instruct", "qwen2-1.5b-instruct",
            "qwen2-0.5b-instruct", "qwen1.5-110b-chat", "qwen1.5-72b-chat", "qwen1.5-32b-chat",
        ], key="model_select", index=0, help="选择你需要的模型")
        st.markdown('</div>', unsafe_allow_html=True)

    st.title("💬 Local Knowledge Quiz System")
    st.caption("🚀 A Streamlit chatbot powered by Dashscope LLM")

    dashscope_api_key, dashvector_api_key, dashvector_endpoint, pdf_folder_path, task_type = sidebar_configuration()
    initialize_messages()

    chat_placeholder = st.container()

    with chat_placeholder:
        display_messages()

    if 'show_modal' not in st.session_state:
        st.session_state['show_modal'] = False

    if st.session_state['show_modal']:
        st.markdown("### Enter Collection Name")
        collection_name = st.text_input("Collection Name", key="collection_name_input")
        if st.button("Submit Collection Name"):
            st.session_state['collection_name'] = collection_name
            st.session_state['show_modal'] = False
            with st.spinner("正在解析文献..."):
                result, topics = vectorize_and_store_and_extract_topics(dashscope_api_key, dashvector_api_key,
                                                                        dashvector_endpoint, pdf_folder_path,
                                                                        collection_name)
                if "successfully" in result:
                    st.success("文献已解析完毕，可询问关于文献里的知识！")
                    st.session_state.messages = [
                        {"role": "assistant", "content": "您好！文献已解析完毕，可询问关于文献里的知识！"}]
                    st.session_state["topics"] = topics
                else:
                    st.error(result)

    if "topics" in st.session_state:
        # Create a right sidebar for displaying the knowledge graphs
        right_sidebar = st.sidebar.container()
        with right_sidebar:
            st.markdown("### 文献知识图谱")
            for pdf_file, topics in st.session_state["topics"].items():
                st.markdown(f"**{pdf_file}**")
                draw_knowledge_graph(topics, pdf_file)

    # 在 chat_placeholder 之后添加语音输入按钮
    if st.button("语音输入"):
        user_input = speech_to_text()
        if user_input:
            st.text_input("语音输入结果", value=user_input, key="speech_input")

    # 修改现有的输入处理逻辑
    prompt = st.chat_input("请输入你的问题...") or st.session_state.get("speech_input", "")
    if prompt:
        if not dashscope_api_key:
            st.info("Please add your Dashscope API key to continue.")
            st.stop()

        dashscope.api_key = dashscope_api_key

        # 使用提示系统生成完整的提示，使用默认的任务类型
        full_prompt = GovernmentAgentPrompts.generate_response(task_type, prompt)
        
        user_message = {"role": "user", "content": prompt}  # 只存储用户的原始输入
        st.session_state.messages.append(user_message)
        with chat_placeholder:
            col1, col2, col3 = st.columns([1, 1, 9])
            with col3:
                st.chat_message(user_message["role"]).write(prompt)

        response_placeholder = st.empty()

        collection_name = st.session_state.get('collection_name', 'default_collection')
        _, response_message = multi_round([{"role": "user", "content": full_prompt}], option, model,
                                          dashvector_api_key, dashvector_endpoint,
                                          collection_name)

        if response_message:
            if response_message["role"] == "assistant":
                st.session_state.messages.append(response_message)
                display_realtime_message(response_message["content"], response_placeholder, response_message["role"])
                # 添加语音输出
                threading.Thread(target=text_to_speech, args=(response_message["content"],)).start()

        st.session_state.chat_records[st.session_state.current_chat_index] = st.session_state.messages

        if len(st.session_state.messages) > 1 and prompt != "":
            truncated_title = prompt[:16] + "..." if len(prompt) > 16 else prompt
            st.session_state.chat_titles[st.session_state.current_chat_index] = truncated_title

    # 清除语音输入结果
    if "speech_input" in st.session_state:
        del st.session_state.speech_input

    '''# 添加文件上传功能
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        text = load_document(uploaded_file)
        if text:
            st.success("文件加载成功！")
            converted_text = convert_text(text)
            chunks = split_text(converted_text)
            visualize_text_processing(text, converted_text, chunks)'''

    # 在main()函数中添加以下代码
    if st.checkbox("文档处理和可视化"):
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


def vectorize_and_store_and_extract_topics(dashscope_api_key, dashvector_api_key, dashvector_endpoint, pdf_folder_path,
                                           collection_name):
    result = vectorize_and_store(dashscope_api_key, dashvector_api_key, dashvector_endpoint, pdf_folder_path,
                                 collection_name)
    topics = extract_pdf_topics(pdf_folder_path)
    return result, topics


if __name__ == "__main__":
    main()