#!/usr/bin/env python3
import os
import json
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# GESTIÓN DE CONFIGURACIÓN
# ==========================================
@dataclass
class TrainerConfig:
    """Configuración centralizada de hiperparámetros."""
    # --- CAMBIOS REALIZADOS AQUÍ ---
    dataset_path: str = "datos_tutor.jsonl"      # <--- TU NUEVO ARCHIVO
    output_dir: str = "./tutor_final"            # <--- TU NUEVA CARPETA
    # -------------------------------
    
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    cpu_threads: int = 16
    
    # Hiperparámetros
    max_seq_length: int = 256
    learning_rate: float = 3e-4
    batch_size: int = 1
    grad_accumulation: int = 4
    epochs: int = 3
    
    # Configuración LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    system_prompt: str = (
        "Eres un tutor de programación. Solo respondes sobre programación, "
        "algoritmos y código. Si te preguntan otro tema, di que solo ayudas con programación."
    )

# ==========================================
# CLASE PRINCIPAL DE ENTRENAMIENTO
# ==========================================
class CpuLoRATrainer:
    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self._setup_environment()
        self.tokenizer = None
        self.model = None
        self.dataset = None

    def _setup_environment(self):
        torch.set_num_threads(self.cfg.cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(self.cfg.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.cfg.cpu_threads)
        print(f"[INIT] Entorno configurado: {self.cfg.cpu_threads} Threads CPU")

    def load_dataset(self):
        print(f"[DATA] Cargando dataset desde: {self.cfg.dataset_path}")
        try:
            with open(self.cfg.dataset_path, "r", encoding="utf-8") as f:
                raw_data = [json.loads(line) for line in f if line.strip()]
            self.dataset = Dataset.from_list(raw_data)
            print(f"       >>> Registros cargados: {len(raw_data)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo {self.cfg.dataset_path}")

    def load_model_and_tokenizer(self):
        print(f"[MODEL] Cargando arquitectura: {self.cfg.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.config.use_cache = False

    def apply_lora_adapters(self):
        print(f"[LORA] Configurando adaptadores (R={self.cfg.lora_r})")
        peft_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=self.cfg.lora_targets,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, peft_config)
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"       >>> Params entrenables: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")

    def process_data(self):
        print("[PREP] Tokenizando dataset...")
        def formatting_func(examples):
            texts = []
            for inst, resp in zip(examples["instruction"], examples["response"]):
                prompt = (
                    f"<|im_start|>system\n{self.cfg.system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{inst}<|im_end|>\n"
                    f"<|im_start|>assistant\n{resp}<|im_end|>"
                )
                texts.append(prompt)
            encodings = self.tokenizer(texts, truncation=True, max_length=self.cfg.max_seq_length, padding="max_length")
            encodings["labels"] = encodings["input_ids"].copy()
            return encodings

        self.tokenized_dataset = self.dataset.map(formatting_func, batched=True, remove_columns=self.dataset.column_names)

    def train(self):
        print(f"[TRAIN] Iniciando entrenamiento ({self.cfg.epochs} épocas)...")
        training_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accumulation,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type="linear",
            warmup_steps=10,
            logging_steps=5,
            save_steps=100,
            save_total_limit=1,
            fp16=False,
            bf16=False,
            optim="adamw_torch",
            report_to="none",
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        trainer.train()
        self._save_artifacts()

    def _save_artifacts(self):
        print(f"[SAVE] Guardando resultados en {self.cfg.output_dir}")
        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        print("\n>>> Proceso finalizado exitosamente.")

def main():
    config = TrainerConfig()
    trainer = CpuLoRATrainer(config)
    trainer.load_dataset()
    trainer.load_model_and_tokenizer()
    trainer.apply_lora_adapters()
    trainer.process_data()
    trainer.train()

if __name__ == "__main__":
    main()