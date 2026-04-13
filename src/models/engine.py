import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logger = logging.getLogger(__name__)

class SLMEngine:
    """Handles loading Small Language Models and managing LoRA adapters."""
    def __init__(self, model_id: str, device: str = "cuda", use_4bit: bool = True):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- FIX LỖI KEYERROR 'TYPE' TẠI ĐÂY ---
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(config, "rope_scaling"):
            # Cách an toàn nhất: Nếu có rope_scaling mà gây lỗi, ta gán None 
            # để model dùng cấu hình mặc định ban đầu
            config.rope_scaling = None
        # Cấu hình Attention để tránh lỗi Flash Attention
        # attn_imp = "eager" # Dùng eager để ổn định nhất trên Colab/GPU phổ thông
        # ---------------------------------------

        bnb_config = None
        if use_4bit and device != "cpu":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config, # Truyền config đã fix vào đây
            quantization_config=bnb_config,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True,
            attn_implementation='eager' 
        )
        
        if use_4bit and device != "cpu":
            self.model = prepare_model_for_kbit_training(self.model)
            
    def apply_lora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        """Applies LoRA configuration to the model."""
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, config)
        logger.info(f"LoRA applied with r={r}. Trainable parameters:")
        self.model.print_trainable_parameters()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generates text based on a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=False, # Use greedy decoding for SQL usually
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part if the prompt is included
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text
