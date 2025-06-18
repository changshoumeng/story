import lazyllm
from lazyllm import pipeline, warp, bind, parallel
from lazyllm.components.formatter import JsonFormatter
import os
import json
import time

# ==================== Agent Prompts ====================

story_planning_prompt = """
你是一位资深的故事策划师。根据用户输入的故事概念，制定详细的故事设定。

请输出JSON格式，包含以下字段：
- story_theme: 故事主题
- genre: 文学类型
- setting: 故事背景设定
- main_characters: 主要人物列表，每个人物包含name, description, role
- world_building: 世界观设定
- core_conflict: 核心冲突
- target_chapters: 建议章节数量（25-30章）

确保设定丰富有趣，为后续创作提供充分的基础。
"""

outline_design_prompt = """
你是一位专业的大纲设计师。基于故事设定，设计详细的章节大纲。

输入信息包含故事设定和人物信息。请设计25-30章的详细大纲，每章2000-3000字。

输出JSON格式，包含章节列表，每章包含：
- chapter_number: 章节编号
- title: 章节标题
- summary: 章节概要
- key_events: 关键事件列表
- character_development: 人物发展要点
- target_words: 目标字数

确保情节发展合理，节奏把控恰当，为5万字长篇小说奠定基础。
"""

content_writing_prompt = """
你是一位优秀的小说作家。根据章节大纲和前文内容，创作具体的章节内容。

写作要求：
1. 字数达到2000-3000字
2. 语言生动，描写细腻
3. 保持人物性格一致
4. 与前文内容连贯
5. 推进情节发展
6. 使用中文创作

请直接输出章节内容，不要包含标题。
"""

quality_review_prompt = """
你是一位专业的文学编辑。审查章节内容的质量和一致性。

审查要点：
1. 内容与大纲的符合度
2. 人物性格和行为的一致性
3. 情节发展的合理性
4. 语言表达的质量
5. 与前文的连贯性

输出JSON格式：
- quality_score: 质量评分（1-10）
- consistency_check: 一致性检查结果
- suggestions: 修改建议列表
- approved: 是否通过审查（true/false）
- revised_content: 修改后的内容（如需要）
"""

editing_prompt = """
你是一位资深文学编辑。对章节内容进行最终润色。

润色要求：
1. 优化语言表达，增强可读性
2. 完善细节描写
3. 统一文风
4. 修正语法错误
5. 增强感染力

请直接输出润色后的章节内容。
"""

# ==================== Context Management ====================

class NovelContext:
    def __init__(self):
        self.story_setting = {}
        self.characters = {}
        self.outline = []
        self.written_chapters = []
        self.current_chapter = 0
        self.total_words = 0
        
    def update_setting(self, setting):
        self.story_setting = setting
        self.characters = {char['name']: char for char in setting.get('main_characters', [])}
        
    def update_outline(self, outline):
        self.outline = outline
        
    def add_chapter(self, chapter_content):
        self.written_chapters.append(chapter_content)
        self.total_words += len(chapter_content)
        self.current_chapter += 1
        
    def get_context_for_chapter(self, chapter_num):
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
    return len(content)

def log_progress(stage, message, context=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {stage}: {message}"
    if context:
        log_msg += f" | 当前字数: {context.total_words} | 进度: {context.current_chapter}/{len(context.outline)}"
    print(log_msg)
    return log_msg

# ==================== Main Pipeline ====================

def create_novel_pipeline():
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
                log_progress("质量审查", f"通过审查，质量评分：{review_result.get('quality_score', 'N/A')}")
            else:
                final_content = chapter_content
                log_progress("质量审查", "需要改进，但继续进行")
            
            # 编辑润色
            log_progress("编辑润色", "开始最终润色")
            polished_content = editor(final_content)
            
            # 添加到上下文和最终小说
            context.add_chapter(polished_content)
            final_novel.append(f"# {chapter_outline.get('title', f'第{i+1}章')}\n\n{polished_content}")
            
            final_word_count = check_word_count(polished_content)
            log_progress("章节完成", f"第{i+1}章完成，润色后字数：{final_word_count}", context)
            
            # 检查是否达到5万字目标
            if context.total_words >= 50000:
                log_progress("目标达成", f"已达到5万字目标，当前总字数：{context.total_words}")
                break
        
        # 生成最终小说
        complete_novel = "\n\n".join(final_novel)
        log_progress("创作完成", f"小说创作完成！总字数：{context.total_words}，共{context.current_chapter}章")
        
        return {
            "novel": complete_novel,
            "statistics": {
                "total_words": context.total_words,
                "total_chapters": context.current_chapter,
                "story_theme": context.story_setting.get('story_theme', ''),
                "genre": context.story_setting.get('genre', '')
            }
        }
    
    return novel_creation_workflow

# ==================== Web Interface ====================

if __name__ == '__main__':
    try:
        # 创建小说创作流程
        novel_workflow = create_novel_pipeline()
        
        # 使用WebModule提供Web界面
        lazyllm.WebModule(
            novel_workflow, 
            port=range(23467, 24000),
            title="Multi-Agent小说创作系统",
            history=[]
        ).start().wait()
        
    except Exception as e:
        print(f"系统启动失败：{e}")
        print("请确保已正确设置LAZYLLM_OPENAI_API_KEY环境变量")
