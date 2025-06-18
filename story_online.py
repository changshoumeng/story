"""Generate a novel outline and chapters using lazyllm."""

import os
import json

import lazyllm
from lazyllm.components.formatter import JsonFormatter

# Context management settings
MAX_CONTEXT_TOKENS = 4000  # Approximate token limit for context
CHARS_PER_TOKEN = 3  # Rough estimate: 1 token ≈ 3 Chinese characters
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN

base_url = os.getenv("LAZYLLM_BASE_URL", "https://www.dmxapi.com/v1/")
api_key = os.getenv("LAZYLLM_OPENAI_API_KEY", "")


def generate_outline(topic: str) -> list[dict]:
    """Generate a list of outline entries from the given topic."""
    prompt = (
        "请根据以下主题生成小说大纲，输出为 JSON 列表，每个列表项包含 title 和 describe 字段：\n"
        f"{topic}"
    )
    module = lazyllm.OnlineChatModule(
        source="openai",
        model="gpt-4.1",
        base_url=base_url,
        api_key=api_key,
        stream=False,
        return_trace=True,
    ).formatter(JsonFormatter())
    outlines = module.prompt(prompt)("")
    if isinstance(outlines, str):
        outlines = json.loads(outlines)
    return outlines


def generate_story(outlines: list[dict]) -> str:
    """Generate a full story based on the provided outlines."""
    system_prompt = (
        "你是一位精通小说写作的智能助手。"
        "请收到以下章节标题和描述，以及前文上下文后，用中文扩展为至少2000字的连贯章节内容。"
    )
    chat = lazyllm.OnlineChatModule(
        source="openai",
        model="gpt-4.1",
        base_url=base_url,
        api_key=api_key,
        stream=False,
        return_trace=True,
    ).prompt(system_prompt)

    sections = []
    accumulated = ""
    for outline in outlines:
        if len(accumulated) > MAX_CONTEXT_CHARS:
            accumulated = accumulated[-MAX_CONTEXT_CHARS:]
            print(
                f"Context truncated to {len(accumulated)} characters to manage token limits"
            )

        user_prompt = (
            f"{outline['title']}\n{outline['describe']}\n前文上下文：\n{accumulated}\n"
            "请根据上述信息，用中文扩展为至少2000字的章节内容。"
        )
        print(f"Generating chapter: {outline['title']}")
        print(f"Context length: {len(accumulated)} characters")
        section = chat(user_prompt)
        sections.append(section)
        accumulated += "\n" + section

    return "\n".join([f"{o['title']}\n{s}" for o, s in zip(outlines, sections)])


def main() -> None:
    """Entry point for manual execution."""
    topic = input("请输入小说主题或简介：")
    outlines = generate_outline(topic)
    final_story = generate_story(outlines)
    print(final_story)


if __name__ == "__main__":
    main()
