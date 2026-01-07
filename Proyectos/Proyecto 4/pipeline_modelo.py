import subprocess
import sys
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# CONFIGURACIÓN
# ==========================================
@dataclass
class Config:
    base_model_cpu: str = "Qwen/Qwen2.5-0.5B-Instruct"
    base_model_gpu: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # --- CAMBIOS REALIZADOS AQUÍ ---
    lora_dir_v1: Path = Path("./lora-tutor")
    lora_dir_v2: Path = Path("./tutor_final")      # <--- TU NUEVA CARPETA
    # -------------------------------
    
    merged_dir: Path = Path("./merged-model")
    gguf_file: Path = Path("tutor_lora.gguf")
    modelfile_name: Path = Path("Modelfile.tutor_final")
    ollama_name: str = "tutorias-progra-final"

# ==========================================
# UTILIDADES DE CONSOLA
# ==========================================
class Logger:
    @staticmethod
    def header(text: str):
        print(f"\n{'-'*60}")
        print(f" {text.upper()}")
        print(f"{'-'*60}")

    @staticmethod
    def info(text: str):
        print(f" [+] {text}")

    @staticmethod
    def step(text: str):
        print(f"\n >>> {text}")

    @staticmethod
    def error(text: str):
        print(f" [!] ERROR: {text}")

def execute_command(cmd: str, cwd: Optional[Path] = None) -> bool:
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.stdout: print(f"     | {result.stdout.strip()}")
        if result.stderr: print(f"     | {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        Logger.error(f"Excepción ejecutando comando: {e}")
        return False

# ==========================================
# LÓGICA DEL PIPELINE
# ==========================================
def get_model_path(cfg: Config) -> tuple[str, Path]:
    if cfg.lora_dir_v2.exists():
        return cfg.base_model_cpu, cfg.lora_dir_v2
    elif cfg.lora_dir_v1.exists():
        return cfg.base_model_gpu, cfg.lora_dir_v1
    else:
        Logger.error(f"No se encontró directorio de adaptadores en {cfg.lora_dir_v2}")
        sys.exit(1)

def merge_models(base_name: str, lora_path: Path, output_path: Path):
    if output_path.exists() and (output_path / "config.json").exists():
        Logger.info("El modelo fusionado ya existe. Saltando paso.")
        return

    Logger.step(f"Cargando tokenizer y modelo base: {base_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        Logger.step("Cargando y aplicando adaptadores LoRA...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        Logger.step("Fusionando pesos (Merge & Unload)...")
        model = model.merge_and_unload()

        Logger.step(f"Guardando modelo en: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)

        del model, base_model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        Logger.info("Fusión completada correctamente.")

    except Exception as e:
        Logger.error(f"Fallo durante la fusión: {e}")
        sys.exit(1)

def convert_to_gguf(model_path: Path, output_file: Path):
    if output_file.exists():
        Logger.info(f"Archivo {output_file} ya existe. Saltando conversión.")
        return

    llama_path = Path("./llama.cpp")
    convert_script = llama_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        Logger.step("Clonando repositorio llama.cpp...")
        if not execute_command("git clone --depth 1 https://github.com/ggerganov/llama.cpp"):
            sys.exit(1)

    Logger.step("Instalando dependencia 'gguf'...")
    execute_command(f"{sys.executable} -m pip install gguf --quiet")

    Logger.step("Iniciando conversión a GGUF (f16)...")
    cmd = f"{sys.executable} {convert_script} {model_path} --outfile {output_file} --outtype f16"
    
    if not execute_command(cmd):
        Logger.error("Falló la conversión a GGUF.")
        sys.exit(1)
        
    Logger.info(f"Archivo generado: {output_file}")

def create_modelfile(filename: Path, gguf_file: Path):
    content = f'''FROM ./{gguf_file}

TEMPLATE """{{{{- if .System }}}}
<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """Eres un TUTOR DE PROGRAMACIÓN especializado. Tu ÚNICO propósito es enseñar programación.
REGLAS ESTRICTAS:
1. SOLO responde preguntas sobre programación, algoritmos, estructuras de datos y código
2. Si te preguntan sobre otro tema, responde que solo ayudas con programación.
3. Siempre da ejemplos de código.
Responde siempre en español."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
'''
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        Logger.info(f"Modelfile generado: {filename}")
    except IOError as e:
        Logger.error(f"No se pudo escribir el Modelfile: {e}")
        sys.exit(1)

def register_ollama_model(model_name: str, modelfile: Path):
    Logger.step(f"Registrando modelo '{model_name}' en Ollama...")
    if execute_command(f"ollama create {model_name} -f {modelfile}"):
        Logger.info("Modelo registrado exitosamente.")
    else:
        Logger.error("No se pudo crear el modelo en Ollama.")
        sys.exit(1)

def main():
    cfg = Config()
    Logger.header("Iniciando Pipeline: LoRA -> GGUF -> Ollama")

    base_model, lora_path = get_model_path(cfg)
    Logger.info(f"Modelo Base: {base_model}")
    Logger.info(f"Adaptadores: {lora_path}")

    Logger.header("Fase 1: Fusión de Modelos")
    merge_models(base_model, lora_path, cfg.merged_dir)

    Logger.header("Fase 2: Conversión GGUF")
    convert_to_gguf(cfg.merged_dir, cfg.gguf_file)

    Logger.header("Fase 3: Configuración Ollama")
    create_modelfile(cfg.modelfile_name, cfg.gguf_file)
    register_ollama_model(cfg.ollama_name, cfg.modelfile_name)

    Logger.header("Proceso Finalizado")
    print(f" El modelo '{cfg.ollama_name}' está listo para usar.\n")

if __name__ == "__main__":
    main()