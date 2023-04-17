# --coding:utf-8--
import json
import os
import re
from typing import List, Union

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import ddg

def extract_text_from_url(url: str):
    try:
        # 发送HTTP GET请求并获取响应
        response = requests.get(url)
        # 将响应内容的字符串表示赋值给变量html
        html = response.text

        # 使用正则表达式排除document.body.outerHTML
        html = re.sub(r'document\.body\.outerHTML\s*=\s*', '', html)

        # 使用BeautifulSoup解析HTML，并获取<body>标签
        soup = BeautifulSoup(html, 'html.parser')
        body = soup.body
        # 获取<body>标签中的文本内容
        text = body.get_text()

        # 去除每行的首尾空格
        lines = (line.strip() for line in text.splitlines())
        # 将每行文本按照多个空格分割成多个短语
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # 将短语用换行符连接成文本，并去除空白的短语
        text = "\n".join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        print(f"Error occurred while extracting text from url: {url}")
        print(f"Exception message: {str(e)}")
        return "No text"

def google_search(query: str, num_results: int = 8) -> str:
    """Return the results of a google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    if not query:
        return json.dumps(search_results)

    results = ddg(query, max_results=num_results)
    if not results:
        return json.dumps(search_results)

    for j in results:
        search_results.append(j)

    return json.dumps(search_results, ensure_ascii=False, indent=4)