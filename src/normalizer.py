import unicodedata

class TextNormalizer:
    def __init__(self, normalize_unicode: bool = True):
        self.normalize_unicode = normalize_unicode

    def normalize(self, text: str) -> str:
        if self.normalize_unicode:
            # NFC - каноническая композиция (рекомендуется для NLP)
            text = unicodedata.normalize('NFC', text)
        
        # Унификация переносов строк
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text