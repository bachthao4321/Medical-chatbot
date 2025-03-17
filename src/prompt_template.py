from langchain_core.prompts import PromptTemplate


PROMPT_TEMPLATE = """
Bạn là trợ lý y tế AI hữu ích, dùng những thông tin đã được cung cấp để trả lời câu hỏi của người dùng
Nếu thông tin không có, hãy nói bạn chưa có thông tin đầy đủ để trả lời.


Context:
{context}

Question:
{question}

Trả lời trực tiếp , không lòng vòng , trả lời đúng ngôn ngữ của câu hỏi
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])
    return prompt


prompt = set_custom_prompt(PROMPT_TEMPLATE)
