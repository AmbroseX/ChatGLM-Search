import json
import os
import platform
import signal
from datetime import datetime
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel

from modules.Search import google_search, extract_text_from_url
from modules.text import summarize_text

# 加载 .env 文件中的环境变量
load_dotenv()
# 读取环境变量
model_path = os.environ.get('CHATGLM_PATH')

os.environ["HTTP_PROXY"] = "127.0.0.1:10808"
os.environ["HTTPS_PROXY"] = "127.0.0.1:10809"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    summary_result = []
    # 对网页内容进行总结
    search_url = []
    summary_with_url = "这里是存放的总结"

    global stop_stream
    print("欢迎使用 ChatGLM-6B 网络搜索问答模型，输入问题即可进行对话，clear 清空对话历史，stop 终止程序，down 导出查询结果")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        if query.strip() == "down":
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # 生成文件名
            filename = f"{current_time}.md"
            with open('./output/'+filename, 'w', encoding='utf-8') as f:
                f.write(summary_with_url)
            continue

        google_result = google_search(query, num_results=3)
        google_result = json.loads(google_result)
        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                 问题:
                {question}
                已知内容:
                {context}
                """

        for result in google_result:
            result['text'] = extract_text_from_url(result['href'])
            search_url.append(result['href'])
            summary = summarize_text(result['text'], query, prompt_template, model, tokenizer, chunk_length=1536, max_length=4096)
            result['summary'] = summary
            print("问题:{}\n 总结:{}\n 来源:{} ".format(query, summary, result['href']))
            summary_result.append(summary)

        search_info = ""
        for i in range(len(summary_result)):
            search_info = search_info + "问题:{}\n,回答:{}".format(query, summary_result[i])

        prompt_template_full = """基于以下已知搜索总结问答信息，用专业语言来回答用户的问题，并整理成要点。
                        不允许在答案中添加编造成分，答案请使用中文。
                         问题:
                        {question}
                        已知内容:
                        {context}
                        """
        text2chatglm = prompt_template_full.format_map({
            'question': query,
            'context': search_info
        })
        summary, history = model.chat(tokenizer, text2chatglm, history=[], max_length=4096)
        torch.cuda.empty_cache()
        summary_out = """问题:{}\n回答:{}""".format(query, summary)

        print(summary_out)

        reference_text = """参考来源:\n
        """
        for result in google_result:
            reference_text = reference_text + "Title:{}\nurl:{}\n".format(result['title'], result['href'])

        print(reference_text)
        summary_with_url = summary_out+'\n'+reference_text


if __name__ == "__main__":
    main()
