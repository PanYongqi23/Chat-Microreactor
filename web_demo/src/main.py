import streamlit as st
from db_funcs import *
import sqlite3
from config import *

# 不缓存数据库连接
def get_db_connection():
    """不使用缓存，直接获取数据库连接"""
    conn = sqlite3.connect('..//data//my_database.db')  # 连接到数据库
    return conn

# 自定义CSS样式
def chat_bubble_css():
    st.markdown("""
        <style>
        .user-bubble {
            background-color: #dcf8c6;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 70%;
            text-align: left;
            float: right;
            clear: both;
        }
        .bot-bubble {
            background-color: #f1f0f0;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 70%;
            float: left;
            clear: both;
        }
        </style>
        """, unsafe_allow_html=True)

def main():
    conn = get_db_connection()
    
    # 自定义聊天气泡样式
    chat_bubble_css()

    st.title('Chat-Microreactor')

    # 使用表单来包含输入框、发送按钮和清空历史按钮
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Input your question:", key="input")
        submit_button = st.form_submit_button(label='Send')
        clear_history_button = st.form_submit_button(label='Clear History')

    # 设置显示区，使用 `st.empty()` 占位符来实现更新内容
    user_chat_display = st.empty()
    bot_chat_display = st.empty()

    # 如果用户提交了问题，则处理问题并显示答案
    if submit_button and user_input:
        # 显示用户输入的对话气泡
        user_chat_display.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

        # 在后台处理问题并展示答案
        chat_history = ""
        for chunk in qaa(user_input, conn):  # qaa函数返回的是一个生成器
            chat_history += chunk
            bot_chat_display.markdown(f'<div class="bot-bubble">{chat_history}</div>', unsafe_allow_html=True)

    # 如果用户点击了“Clear History”按钮，则清空历史记录
    if clear_history_button:
        # 假设 history_global 是一个全局变量，用于存储聊天历史
        # history_global.clear()  # 这里需要确保history_global变量存在且正确地存储了聊天历史
        st.success('History Cleared')

if __name__ == "__main__":
    main()
