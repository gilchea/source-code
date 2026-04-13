import unittest
from unittest.mock import MagicMock, patch
from src.models.engine import SLMEngine

class TestSLMEngine(unittest.TestCase):
    @patch('src.models.engine.AutoTokenizer.from_pretrained')
    @patch('src.models.engine.AutoModelForCausalLM.from_pretrained')
    @patch('src.models.engine.AutoConfig.from_pretrained')
    def test_initialization(self, mock_config, mock_model, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<eos>"
        
        engine = SLMEngine("dummy-model", device="cpu", use_4bit=False)
        
        self.assertEqual(engine.model_id, "dummy-model")
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()

    @patch('src.models.engine.AutoTokenizer.from_pretrained')
    @patch('src.models.engine.AutoModelForCausalLM.from_pretrained')
    @patch('src.models.engine.get_peft_model')
    @patch('src.models.engine.AutoConfig.from_pretrained')
    def test_apply_lora(self, mock_config, mock_peft, mock_model, mock_tokenizer):
        engine = SLMEngine("dummy-model", device="cpu", use_4bit=False)
        engine.apply_lora(r=8)
        
        mock_peft.assert_called_once()
        # The config is passed as the second positional argument
        args, kwargs = mock_peft.call_args
        peft_config = args[1]
        self.assertEqual(peft_config.r, 8)

if __name__ == '__main__':
    unittest.main()
