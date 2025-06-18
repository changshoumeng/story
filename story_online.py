import lazyllm
from lazyllm import pipeline, warp, bind
from lazyllm.components.formatter import JsonFormatter
import os
import json

toc_prompt = """
You are now an intelligent assistant specialized in novel writing. Your task is to carefully analyze the user's input and craft a detailed outline for a novel. The outline must be structured as a list of nested dictionaries. Each dictionary includes:

- `title`: Clearly formatted with Markdown headers to indicate hierarchy (#, ##, ###), signifying chapters, sections, or subsections.
- `describe`: An informative, vivid, and engaging description guiding the writing of that particular section, including narrative elements, character insights, plot developments, and thematic details.

Please generate the corresponding structured outline based on the provided user input.

Example output:
[
    {
        "title": "# Chapter 1: The Mysterious Encounter",
        "describe": "Introduce the main protagonist and establish the initial setting vividly. Include a compelling event that immediately engages the reader and hints at future conflicts."
    },
    {
        "title": "## Section 1.1: Unforeseen Consequences",
        "describe": "Describe the aftermath of the initial event. Detail character emotions, reactions, and initial decisions, laying groundwork for character development and future plot twists."
    },
    {
        "title": "### Subsection 1.1.1: The First Revelation",
        "describe": "Delve deeper into the specific realization or revelation that changes the protagonist's understanding of their situation. Provide sensory details and emotional insights to enrich reader immersion."
    }
]

User input is as follows:
"""  # noqa: E50E

completion_prompt = """
You are now an intelligent assistant specialized in novel writing. Your task is to generate a coherent chapter based on the provided structured parameters.

Instructions:
1. Read the outline (title and describe) to understand the current chapter's requirements
2. Use the context (previous chapters) to ensure narrative continuity and character consistency
3. Create smooth transitions that reference previous events and foreshadow future developments
4. Maintain thematic coherence throughout the narrative
5. Generate rich, engaging content with vivid scenes and deep character development
6. Ensure the output is at least 2000 words in Chinese

Input format (JSON):
{
    "outline": {
        "title": "# Chapter Title",
        "describe": "Chapter description and writing guidance"
    },
    "context": "Previous chapters content for continuity",
    "format": "markdown",
    "style": "engaging_narrative"
}

Output: Generate the chapter content directly without repeating the title, ensuring smooth continuation from the provided context.

Process the following structured input:
"""  # noqa: E50E

# Context management settings
MAX_CONTEXT_TOKENS = 4000  # Approximate token limit for context
CHARS_PER_TOKEN = 3  # Rough estimate: 1 token ≈ 3 Chinese characters
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN

base_url = os.getenv("LAZYLLM_BASE_URL", "https://www.dmxapi.com/v1/")
api_key = os.getenv("LAZYLLM_OPENAI_API_KEY", "")

if __name__ == '__main__':
    # Read user-provided theme for outline generation
    topic = input("请输入小说主题或简介：")
    # Simplified prompt without JSON example braces to avoid placeholder parsing
    script_toc_prompt = f"""请根据以下主题生成小说大纲，输出为 JSON 列表，每个列表项包含 title 和 describe 字段：
+{topic}"""
    # Generate outline list (empty string as dummy input)
    outline_module = lazyllm.OnlineChatModule(
        source="openai", model="gpt-4.1", base_url=base_url,
        api_key=api_key, stream=False, return_trace=True
    ).formatter(JsonFormatter())
    outlines = outline_module.prompt(script_toc_prompt)("")

    # Step 2: sequentially generate story sections with context accumulation
    story_system_prompt = (
        "你是一位精通小说写作的智能助手。", 
        "请收到以下章节标题和描述，以及前文上下文后，用中文扩展为至少2000字的连贯章节内容。"
    )
    # Combine system prompt into single string
    story_system_prompt = ''.join(story_system_prompt)
    story_chat = lazyllm.OnlineChatModule(
        source="openai", model="gpt-4.1", base_url=base_url,
        api_key=api_key, stream=False, return_trace=True
    ).prompt(story_system_prompt)
    story_sections = []
    accumulated_story = ''
    
    for outline in outlines:
        # Apply sliding window context management
        if len(accumulated_story) > MAX_CONTEXT_CHARS:
            # Keep only the most recent context to stay within token limits
            accumulated_story = accumulated_story[-MAX_CONTEXT_CHARS:]
            print(f"Context truncated to {len(accumulated_story)} characters to manage token limits")
        
        # Build chapter prompt as plain text
        user_prompt = f"""{outline['title']}
{outline['describe']}
前文上下文：
{accumulated_story}
请根据上述信息，用中文扩展为至少2000字的章节内容。"""
        print(f"Generating chapter: {outline['title']}")
        print(f"Context length: {len(accumulated_story)} characters")
        # Pass user_prompt as input string to the module
        section = story_chat(user_prompt)
        story_sections.append(section)
        accumulated_story += '\n' + section

    # Step 3: assemble and output final novel text
    final_story = "\n".join([f"{o['title']}\n{s}" for o, s in zip(outlines, story_sections)])
    print(final_story)
