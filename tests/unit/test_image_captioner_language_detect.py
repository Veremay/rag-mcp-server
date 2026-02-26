
import pytest
from src.ingestion.transform.image_captioner import ImageCaptioner
from unittest.mock import MagicMock

class TestImageCaptionerLanguageDetection:
    @pytest.fixture
    def captioner(self):
        settings = MagicMock()
        settings.ingestion.transform.image_captioner.enabled = True
        settings.ingestion.transform.image_captioner.prompt_path = None
        return ImageCaptioner(settings)

    def test_empty_text_defaults_to_chinese(self, captioner):
        assert captioner._detect_language("") == "zh"
        assert captioner._detect_language(None) == "zh"

    def test_only_image_markdown_defaults_to_chinese(self, captioner):
        text = "![Image](path/to/image.png)"
        assert captioner._detect_language(text) == "zh"

    def test_chinese_content_returns_zh(self, captioner):
        text = "这是一个测试。"
        assert captioner._detect_language(text) == "zh"
        
        text = "Mixed content with 中文 characters."
        assert captioner._detect_language(text) == "zh"

    def test_short_english_defaults_to_chinese(self, captioner):
        # "Table 1" has 6 chars -> 5 letters. Threshold is > 10.
        text = "Table 1" 
        assert captioner._detect_language(text) == "zh"
        
        text = "12345"
        assert captioner._detect_language(text) == "zh"

    def test_long_english_returns_en(self, captioner):
        text = "This is a sufficiently long English sentence to be detected."
        assert captioner._detect_language(text) == "en"
        
        # "Table 1: Sales Data for 2023" -> ~20 english chars
        text = "Table 1: Sales Data for 2023"
        assert captioner._detect_language(text) == "en"

    def test_image_markdown_ignored_in_detection(self, captioner):
        # The image path has many english letters, but should be ignored.
        # "![Image](long/path/with/many/english/words/image.png)"
        # If not ignored, this might trigger 'en'.
        # But since it's stripped, the remaining text is empty -> 'zh'.
        text = "![Image](long/path/with/many/english/words/image.png)"
        assert captioner._detect_language(text) == "zh"
