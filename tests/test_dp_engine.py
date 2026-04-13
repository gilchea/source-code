import unittest
import torch
from src.privacy.dp_engine import DPEngine

class TestDPEngine(unittest.TestCase):
    def setUp(self):
        self.weights = {
            "layer1.lora_A": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer1.lora_B": torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        }

    def test_apply_dp_no_noise(self):
        # Clip only
        processed = DPEngine.apply_dp(self.weights, clip_threshold=1.0, noise_multiplier=0.0)
        
        # Check if norm is capped
        total_norm = torch.norm(torch.stack([torch.norm(v.float()) for v in processed.values()]))
        self.assertLessEqual(total_norm, 1.0001) # Small epsilon for float

    def test_apply_dp_with_noise(self):
        processed = DPEngine.apply_dp(self.weights, clip_threshold=1.0, noise_multiplier=0.1)
        # Check if weights are different (due to noise)
        for k in self.weights:
            self.assertFalse(torch.equal(self.weights[k], processed[k]))

    def test_apply_sparsification(self):
        # Top-K ratio 0.5 should keep half weights
        processed = DPEngine.apply_sparsification(self.weights, top_k_ratio=0.5)
        
        for k, v in processed.items():
            num_zeros = (v == 0).sum()
            self.assertEqual(num_zeros, v.numel() // 2)

    def test_apply_quantization(self):
        processed = DPEngine.apply_quantization(self.weights)
        for v in processed.values():
            self.assertEqual(v.dtype, torch.float16)

if __name__ == '__main__':
    unittest.main()
