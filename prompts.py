class GovernmentAgentPrompts:
    SYSTEM_PROMPT = """你是一个智能助手，能够处理各种类型的问题。你可以回答关于政府政策、法规和行政程序的问题，也可以处理日常生活、个人简历等各种主题的询问。请根据用户的具体问题提供相应的回答，保持礼貌、专业和中立的态度。"""

    TASK_PROMPTS = {
        "通用问答": "请回答以下问题：{query}",
        "政策解读": "请解释以下政策的主要内容、目标和可能的影响：{query}",
        "法规咨询": "关于{query}，请说明其主要条款和适用范围。",
        "行政程序": "请详细描述{query}的步骤和所需材料。",
        "公共服务": "我想了解关于{query}的信息，包括如何申请、处理时间和所需文件。",
        "投诉建议": "我要就{query}进行投诉或提出建议，请指导我应该遵循什么程序。",
        "数据分析": "请分析以下数据并提供见解：{query}",
        "应急响应": "在{query}情况下，应该采取哪些措施？",
        "城市规划": "请解释{query}的规划，包括主要目标和实施时间表。",
        "环境保护": "关于{query}，有哪些政策和措施？公众可以如何参与？",
        "教育政策": "请介绍最新的{query}教育政策变化及其影响。",
        "医疗卫生": "关于{query}，有哪些政策支持和公共资源？",
        "社会保障": "请解释{query}的覆盖范围、申请条件和程序。",
        "经济发展": "请分析{query}可能产生的影响。",
        "文化遗产": "请介绍在保护和推广{query}方面的措施。",
        "交通管理": "关于{query}，请解释其实施原因、主要内容和预期效果。",
        "科技创新": "在推动{query}发展方面有哪些支持政策和措施？",
        "农业政策": "请解释{query}的主要内容和对农民的影响。",
        "外交关系": "请概述与{query}的合作现状和未来展望。",
        "反腐倡廉": "请介绍在{query}方面的最新举措和成效。",
        "人才政策": "请解释{query}的主要内容和如何促进相关行业的人才发展。"
    }

    @classmethod
    def get_prompt(cls, task_type, query):
        base_prompt = cls.TASK_PROMPTS.get(task_type, cls.TASK_PROMPTS["通用问答"])
        return base_prompt.format(query=query)

    @classmethod
    def generate_response(cls, task_type, query):
        system_prompt = cls.SYSTEM_PROMPT
        task_prompt = cls.get_prompt(task_type, query)
        return f"{system_prompt}\n\n任务：{task_prompt}\n\n回答："