"""Исключения сервиса."""


class LLMError(Exception):
    """Базовая ошибка работы с LLM."""


class LLMUnavailable(LLMError):
    """LLM недоступна: таймаут, сеть, rate limit после всех повторов, 5xx."""


class LLMBadResponse(LLMError):
    """LLM ответила, но ответ пустой или не соответствует формату."""
