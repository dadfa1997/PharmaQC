import re

class TextCleaner:
    def clean(self, text: str) -> str:
        # Удаление непечатных символов (кроме переносов строк и табуляции)
        text = re.sub(r'[^\S\r\n]+', ' ', text) 
        
        # Удаление множественных пустых строк
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Удаление пробелов в начале и конце каждой строки
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        
        return text.strip()