# --coding:utf-8--

from typing import Generator

import torch
from tqdm import tqdm


def split_text(text: str, max_length: int = 1024):
    """Split text into chunks of a maximum length

    Args:
        text (str): The text to split
        max_length (int, optional): The maximum length of each chunk. Defaults to 1024.

    Returns:
        List[str]: The list of chunks

    Raises:
        ValueError: If the max_length is less than 1
    """
    if max_length < 1:
        raise ValueError(f"The maximum length must be greater than 0")

    chunks = []
    current_chunk = ''
    for word in text.split():
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += ' ' + word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def summarize_text(text, query,prompt_template, model, tokenizer, chunk_length=1024, max_length=2048):
    """Summarize the text for the given query using the given model and tokenizer.

    Args:
        text (str): The text to summarize.
        query (str): The query string.
        model (torch.nn.Module): The PyTorch model object.
        tokenizer (transformers.PreTrainedTokenizer): The PyTorch tokenizer object.
        max_length (int, optional): The maximum length of each summary. Defaults to 2048.

    Returns:
        str: The summary string.
    """
    chunks = list(split_text(text, max_length=chunk_length))
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        text2chatglm = prompt_template.format_map({
            'question': query,
            'context': '\n'.join(chunk)
        })
        summary, history = model.chat(tokenizer, text2chatglm, history=[], max_length=max_length)
        summaries.append(summary)
        torch.cuda.empty_cache()

    all_text = ''.join(summaries)
    text2chatglm = prompt_template.format_map({
        'question': query,
        'context': all_text
    })
    summary, history = model.chat(tokenizer, text2chatglm, history=[], max_length=max_length)
    torch.cuda.empty_cache()
    return summary