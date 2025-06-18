"""Multi-Agent powered novel generation assistant.

This script builds a LazyLLM pipeline that coordinates several agents to
plan, outline, write, review and polish a long-form novel.
"""

import lazyllm
from lazyllm import pipeline
from lazyllm.components.formatter import JsonFormatter
import os
import json
import logging
from dataclasses import dataclass, field


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

TARGET_WORDS = int(os.getenv("NOVEL_TARGET_WORDS", "50000"))

# ==================== Agent Prompts ====================

story_planning_prompt = """
You are now a senior story strategist and creative consultant. Your task is to understand the user's story concept and develop a comprehensive story setting that serves as the foundation for a 50,000+ word novel.

Please analyze the user input and create a detailed story framework in pure JSON format (do not use ```json``` markers).

Output format requirements:
{
    "story_theme": "Main theme of the story",
    "genre": "Literary genre (e.g., romance, fantasy, mystery)",
    "setting": "Detailed background setting including time, place, and world context",
    "main_characters": [
        {
            "name": "Character name",
            "description": "Detailed character description",
            "role": "Character's role in the story"
        }
    ],
    "world_building": "Comprehensive world-building details",
    "core_conflict": "Central conflict that drives the narrative",
    "target_chapters": "Recommended chapter count (25-30 chapters)"
}

Example output:
{
    "story_theme": "Love transcending time and space",
    "genre": "Science fiction romance",
    "setting": "A futuristic city where time travel is possible but forbidden",
    "main_characters": [
        {
            "name": "Alex Chen",
            "description": "A brilliant physicist who discovers time travel",
            "role": "Protagonist"
        }
    ],
    "world_building": "A world where temporal authorities monitor time streams",
    "core_conflict": "Love versus duty to preserve timeline integrity",
    "target_chapters": 28
}

Requirements:
- Ensure rich and engaging setting details
- Create compelling characters with clear motivations
- Establish conflicts that can sustain a full-length novel
- Output pure JSON string without any markdown formatting
"""

outline_design_prompt = """
You are now a professional outline designer and narrative architect. Your task is to create a detailed chapter-by-chapter outline based on the provided story setting, designed for a 50,000+ word novel with 25-30 chapters.

Input: Story setting and character information in JSON format.

Your task is to design a comprehensive chapter outline where each chapter contains 2,000-3,000 words, ensuring proper pacing and narrative development.

Output format in pure JSON (do not use ```json``` markers):
[
    {
        "chapter_number": 1,
        "title": "Chapter title",
        "summary": "Brief chapter summary",
        "key_events": ["Event 1", "Event 2", "Event 3"],
        "character_development": "How characters develop in this chapter",
        "target_words": 2500
    }
]

Example output:
[
    {
        "chapter_number": 1,
        "title": "The Discovery",
        "summary": "Alex discovers the time travel device in the abandoned laboratory",
        "key_events": [
            "Alex explores the mysterious laboratory",
            "Discovery of the temporal device",
            "First accidental activation"
        ],
        "character_development": "Alex transforms from curious scientist to reluctant time traveler",
        "target_words": 2400
    },
    {
        "chapter_number": 2,
        "title": "First Jump",
        "summary": "Alex's first intentional time travel experience",
        "key_events": [
            "Preparation for time travel",
            "Journey to the past",
            "Unexpected consequences"
        ],
        "character_development": "Alex gains confidence but also realizes the dangers",
        "target_words": 2600
    }
]

Requirements:
- Create 25-30 chapters with logical progression
- Ensure each chapter advances the plot meaningfully
- Balance action, character development, and world-building
- Maintain proper pacing throughout the narrative
- Output pure JSON string without any markdown formatting
"""

content_writing_prompt = """
You are now an accomplished novelist and creative writer. Your task is to write engaging chapter content based on the provided chapter outline and previous story context.

Your writing should be vivid, immersive, and maintain consistency with established characters and plot elements.

Writing requirements:
- Target word count: 2,000-3,000 words per chapter
- Use descriptive and engaging language
- Maintain character consistency throughout
- Ensure smooth narrative flow from previous chapters
- Advance the plot according to the outline
- Write in Chinese for better narrative expression

Input format:
- Chapter outline with key events and character development notes
- Story setting and character information
- Previous chapter content for context

Output: Direct chapter content without title or formatting markers.

Writing style guidelines:
- Use vivid descriptions and sensory details
- Develop realistic dialogue that reveals character
- Balance action with introspection
- Create emotional resonance with readers
- Maintain narrative tension and pacing

Please write the complete chapter content based on the provided information.
"""

quality_review_prompt = """
You are now a professional literary editor and quality assurance specialist. Your task is to review chapter content for quality, consistency, and adherence to the established narrative framework.

Your review should evaluate multiple aspects of the writing and provide actionable feedback.

Review criteria:
1. Adherence to the chapter outline and story framework
2. Character consistency and development
3. Plot coherence and logical progression
4. Language quality and narrative flow
5. Continuity with previous chapters

Output format in pure JSON (do not use ```json``` markers):
{
    "quality_score": 8,
    "consistency_check": "Detailed analysis of consistency issues",
    "suggestions": [
        "Specific improvement suggestion 1",
        "Specific improvement suggestion 2"
    ],
    "approved": true,
    "revised_content": "Improved content if revisions are needed"
}

Example output:
{
    "quality_score": 7,
    "consistency_check": "Character dialogue matches established personality, but pacing could be improved in the middle section",
    "suggestions": [
        "Add more sensory details in the laboratory scene",
        "Strengthen the emotional impact of the discovery moment",
        "Clarify the technical explanation of the device"
    ],
    "approved": true,
    "revised_content": ""
}

Evaluation guidelines:
- Score 1-10 where 10 is exceptional quality
- Focus on constructive feedback
- Approve content that meets minimum quality standards
- Provide revised content only if significant improvements are needed
- Output pure JSON string without any markdown formatting
"""

editing_prompt = """
You are now a senior literary editor specializing in narrative polish and refinement. Your task is to perform final editing and enhancement of chapter content to achieve publication-ready quality.

CRITICAL REQUIREMENT: You must output the COMPLETE polished chapter content. Do not summarize, truncate, or provide excerpts. The output should be the full chapter with improvements applied.

Your editing should focus on:
- Language refinement and flow optimization
- Enhanced readability and engagement
- Consistent writing style throughout
- Grammar and syntax perfection
- Emotional impact amplification

Editing principles:
1. Preserve the author's voice while improving clarity
2. Enhance descriptive passages for greater immersion
3. Strengthen dialogue for authenticity and impact
4. Ensure smooth transitions between scenes
5. Optimize pacing for reader engagement
6. MAINTAIN THE ORIGINAL LENGTH - do not shorten the content

Input: Chapter content requiring final polish

Output: Complete refined and polished chapter content ready for publication (FULL LENGTH)

Quality standards:
- Professional-grade language and style
- Engaging and immersive narrative voice
- Error-free grammar and syntax
- Consistent tone and pacing
- Enhanced emotional resonance
- PRESERVE ORIGINAL CONTENT LENGTH

Please provide the COMPLETE final polished version of the chapter content. Do not provide summaries or excerpts.
"""

# ==================== Context Management ====================

@dataclass
class NovelContext:
    """Maintain novel state and provide context for each chapter."""

    story_setting: dict = field(default_factory=dict)
    characters: dict = field(default_factory=dict)
    outline: list = field(default_factory=list)
    written_chapters: list = field(default_factory=list)
    current_chapter: int = 0
    total_words: int = 0
        
    def update_setting(self, setting):
        """Update story setting and character index."""
        self.story_setting = setting
        self.characters = {char['name']: char for char in setting.get('main_characters', [])}
        
    def update_outline(self, outline):
        """Store chapter outline list."""
        self.outline = outline
        
    def add_chapter(self, chapter_content):
        """Append new chapter and update statistics."""
        self.written_chapters.append(chapter_content)
        self.total_words += check_word_count(chapter_content)
        self.current_chapter += 1
        
    def get_context_for_chapter(self, chapter_num):
        """Return context dict for writing the given chapter."""
        # 获取前3章内容作为上下文
        recent_chapters = self.written_chapters[-3:] if self.written_chapters else []
        context = {
            'story_setting': self.story_setting,
            'characters': self.characters,
            'recent_chapters': recent_chapters,
            'current_chapter_outline': self.outline[chapter_num] if chapter_num < len(self.outline) else {},
            'total_words': self.total_words,
            'progress': f"{chapter_num + 1}/{len(self.outline)}"
        }
        return context

# ==================== Quality Control ====================

def check_word_count(content):
    """改进的字数统计 - 更准确地统计中文字数"""
    # 移除空白字符后统计长度，对中文更准确
    cleaned_content = content.replace(' ', '').replace('\n', '').replace('\t', '')
    return len(cleaned_content)

def save_novel_to_cache(novel_content, story_theme, total_words, total_chapters):
    """Save the entire novel as a markdown file in a local cache directory."""
    import os
    from datetime import datetime
    
    # 创建cache目录
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_theme = "".join(c for c in story_theme if c.isalnum() or c in (' ', '-', '_')).rstrip()[:20]
    filename = f"novel_{safe_theme}_{timestamp}.md"
    filepath = os.path.join(cache_dir, filename)
    
    # 准备markdown内容
    markdown_content = f"""# {story_theme}

**创作时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**总字数**: {total_words:,} 字  
**章节数**: {total_chapters} 章  

---

{novel_content}

---

*本小说由 LazyLLM Multi-Agent 系统创作*
"""
    
    # 保存文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logging.info(f"完整小说已保存到: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"保存小说失败: {e}")
        return None

def log_progress(stage, message, context=None):
    """Write a progress message to logs and optionally include stats."""
    log_msg = f"{stage}: {message}"
    if context:
        log_msg += (
            f" | 当前字数: {context.total_words}"
            f" | 进度: {context.current_chapter}/{len(context.outline)}"
        )
    logging.info(log_msg)
    return log_msg

# ==================== Main Pipeline ====================

def create_novel_pipeline():
    """Build and return the novel creation workflow pipeline."""
    # 环境配置
    base_url = os.getenv("LAZYLLM_BASE_URL", "https://www.dmxapi.com/v1/")
    api_key = os.getenv("LAZYLLM_OPENAI_API_KEY", "")
    
    if not api_key:
        raise ValueError("请设置LAZYLLM_OPENAI_API_KEY环境变量")
    
    # 创建各个Agent
    story_planner = lazyllm.OnlineChatModule(
        source="openai", model="gpt-4", base_url=base_url,
        api_key=api_key, stream=False, return_trace=True
    ).formatter(JsonFormatter()).prompt(story_planning_prompt)
    
    outline_designer = lazyllm.OnlineChatModule(
        source="openai", model="gpt-4", base_url=base_url,
        api_key=api_key, stream=False, return_trace=True
    ).formatter(JsonFormatter()).prompt(outline_design_prompt)
    
    content_writer = lazyllm.OnlineChatModule(
        source="openai", model="gpt-4", base_url=base_url,
        api_key=api_key, stream=False, return_trace=True
    ).prompt(content_writing_prompt)
    
    quality_reviewer = lazyllm.OnlineChatModule(
        source="openai", model="gpt-4", base_url=base_url,
        api_key=api_key, stream=False, return_trace=True
    ).formatter(JsonFormatter()).prompt(quality_review_prompt)
    
    editor = lazyllm.OnlineChatModule(
        source="openai", model="gpt-4", base_url=base_url,
        api_key=api_key, stream=False, return_trace=True
    ).prompt(editing_prompt)

    # 使用 pipeline 串联各 Agent，便于可视化和管理
    with pipeline() as novel_creator:
        novel_creator.planner = story_planner
        novel_creator.outliner = outline_designer
        novel_creator.writer = content_writer
        novel_creator.reviewer = quality_reviewer
        novel_creator.editor = editor
    
    # 小说创作主流程
    def novel_creation_workflow(user_input):
        context = NovelContext()
        log_progress("系统初始化", "Multi-Agent小说创作系统启动")
        
        # 阶段1：故事规划
        log_progress("故事策划", "开始分析用户输入并制定故事设定")
        story_setting = story_planner(user_input)
        context.update_setting(story_setting)
        log_progress("故事策划", f"完成故事设定：{story_setting.get('story_theme', '未知主题')}")
        
        # 阶段2：大纲设计
        log_progress("大纲设计", "开始设计详细章节大纲")
        setting_text = json.dumps(story_setting, ensure_ascii=False)
        outline = outline_designer(setting_text)
        context.update_outline(outline)
        log_progress("大纲设计", f"完成大纲设计，共{len(outline)}章")
        
        # 阶段3：章节创作循环
        final_novel = []
        
        for i, chapter_outline in enumerate(outline):
            log_progress("内容创作", f"开始创作第{i+1}章：{chapter_outline.get('title', '')}")
            
            # 准备章节创作的上下文
            chapter_context = context.get_context_for_chapter(i)
            writing_prompt = f"""
章节大纲：{json.dumps(chapter_outline, ensure_ascii=False)}

故事背景：{json.dumps(chapter_context['story_setting'], ensure_ascii=False)}

前文内容：
{chr(10).join(chapter_context['recent_chapters'][-2:]) if chapter_context['recent_chapters'] else "这是第一章"}

请根据以上信息创作本章内容。
"""
            
            # 内容创作
            chapter_content = content_writer(writing_prompt)
            word_count = check_word_count(chapter_content)
            log_progress("内容创作", f"完成初稿，字数：{word_count}")
            
            # 质量审查
            log_progress("质量审查", "开始质量检查")
            review_prompt = f"""
章节内容：
{chapter_content}

章节大纲：{json.dumps(chapter_outline, ensure_ascii=False)}
故事设定：{json.dumps(chapter_context['story_setting'], ensure_ascii=False)}
前文内容：{chr(10).join(chapter_context['recent_chapters'][-1:]) if chapter_context['recent_chapters'] else "无"}
"""
            review_result = quality_reviewer(review_prompt)
            
            # 根据审查结果决定是否需要修改
            if review_result.get('approved', False):
                final_content = review_result.get('revised_content', chapter_content)
                if final_content.strip():  # 如果有修改内容，使用修改后的
                    logging.info(f"质量审查: 使用修改后内容，原字数: {word_count}, 修改后字数: {check_word_count(final_content)}")
                else:
                    final_content = chapter_content  # 否则使用原内容
                log_progress("质量审查", f"通过审查，质量评分：{review_result.get('quality_score', 'N/A')}")
            else:
                final_content = chapter_content
                log_progress("质量审查", "需要改进，但继续进行")
            
            # 编辑润色
            log_progress("编辑润色", "开始最终润色")
            polished_content = editor(final_content)
            
            # 验证润色后内容长度
            polished_word_count = check_word_count(polished_content)
            original_word_count = check_word_count(final_content)
            
            # 如果润色后内容明显变短，使用原内容
            if polished_word_count < original_word_count * 0.5:
                logging.warning(f"编辑润色: 润色后内容过短({polished_word_count} vs {original_word_count})，使用原内容")
                polished_content = final_content
                polished_word_count = original_word_count
            
            # 添加到上下文和最终小说
            context.add_chapter(polished_content)
            final_novel.append(f"# {chapter_outline.get('title', f'第{i+1}章')}\n\n{polished_content}")
            
            log_progress("章节完成", f"第{i+1}章完成，润色后字数：{polished_word_count}", context)
            if context.total_words >= TARGET_WORDS:
                log_progress("目标达成", f"已达到{TARGET_WORDS}字目标，当前总字数：{context.total_words}")
                break
        
        # 生成最终小说
        complete_novel = "\n\n".join(final_novel)
        log_progress("创作完成", f"小说创作完成！总字数：{context.total_words}，共{context.current_chapter}章")
        
        # 保存到本地cache目录
        story_theme = context.story_setting.get('story_theme', '未知主题')
        cache_file = save_novel_to_cache(complete_novel, story_theme, context.total_words, context.current_chapter)
        
        # 准备返回结果
        result = {
            "success": True,
            "message": {
                "content": f"✅ 小说创作完成！\n\n📖 **{story_theme}**\n\n📊 **统计信息**:\n- 总字数: {context.total_words:,} 字\n- 章节数: {context.current_chapter} 章\n- 文学类型: {context.story_setting.get('genre', '未知')}\n\n💾 **本地保存**: {cache_file or '保存失败'}\n\n---\n\n",
                "log": "",
                "files": []
            },
            "novel": complete_novel,
            "statistics": {
                "total_words": context.total_words,
                "total_chapters": context.current_chapter,
                "story_theme": story_theme,
                "genre": context.story_setting.get('genre', ''),
                "cache_file": cache_file
            }
        }
        
        logging.info(f"返回结果: 成功创作{context.current_chapter}章小说，总计{context.total_words}字")
        return result
    
    novel_creator.run = novel_creation_workflow
    return novel_creator.run

# ==================== Web Interface ====================

if __name__ == '__main__':
    try:
        # 创建小说创作流程
        novel_workflow = create_novel_pipeline()

        logging.info("Web 服务启动，端口范围 23467-23999")

        # 使用WebModule提供Web界面
        lazyllm.WebModule(
            novel_workflow,
            port=range(23467, 24000),
            title="Multi-Agent小说创作系统",
            history=[]
        ).start().wait()

    except Exception as e:
        logging.error(f"系统启动失败: {e}")
        logging.error("请确保已正确设置LAZYLLM_OPENAI_API_KEY环境变量")
