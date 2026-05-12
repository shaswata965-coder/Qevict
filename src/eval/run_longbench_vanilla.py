import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo root to path to import src.* modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine import engine
from src.data import data_loader
import src.sticky_config as config

def main():
    print(f"Loading Pure Vanilla LLaMA (No sticky cache logic) from {config.MODEL_PATH}...")
    
    # Load base HF LLaMA 3.2 1B
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig as HFLlamaConfig
        with open(os.path.join(config.MODEL_PATH, "config.json"), "r") as f:
            v_config_dict = json.load(f)
        rope_scaling_config = v_config_dict.get("rope_scaling", None)
        if "rope_scaling" in v_config_dict:
            del v_config_dict["rope_scaling"]
        v_config = HFLlamaConfig(**v_config_dict)

        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH, 
            config=v_config,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

        # --- MONKEY PATCH LLAMA 3 ROPE ---
        if rope_scaling_config is not None:
            rope_type = rope_scaling_config.get("type", rope_scaling_config.get("rope_type", ""))
            if rope_type == "llama3":
                print("Monkey-patching HuggingFace 4.35 model with custom Llama 3 RoPE...")
                from src.models.sticky_llama_attention import Llama3RotaryEmbedding
                dim = v_config.hidden_size // v_config.num_attention_heads
                max_pos = v_config.max_position_embeddings
                base = getattr(v_config, "rope_theta", 500000.0)
                factor = rope_scaling_config.get("factor", 8.0)
                low_freq = rope_scaling_config.get("low_freq_factor", 1.0)
                high_freq = rope_scaling_config.get("high_freq_factor", 4.0)
                orig_max_pos = rope_scaling_config.get("original_max_position_embeddings", getattr(config, "ORIGINAL_MAX_POSITION_EMBEDDINGS", 8192))
                for layer in model.model.layers:
                    layer.self_attn.rotary_emb = Llama3RotaryEmbedding(
                        dim=dim, max_position_embeddings=max_pos, base=base,
                        scaling_factor=factor, low_freq_factor=low_freq,
                        high_freq_factor=high_freq, original_max_position_embeddings=orig_max_pos
                    ).to(device=layer.self_attn.q_proj.weight.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Model and Tokenizer loaded successfully.")

    # Load LongBench datasets
    data_dir = getattr(config, "DATA_DIR", "1LongBenchData")
    print(f"Loading datasets from {data_dir}...")
    datasets = data_loader.load_datasets(data_dir)

    all_results = {}
    
    for task_name, dataset in datasets.items():
        print(f"\nEvaluating dataset: {task_name}")
        for seed in config.SEEDS:
            res = engine.evaluate_dataset(
                name=task_name,
                dataset=dataset,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                device="cuda"
            )
            
            # Aggregate or store the result
            if task_name not in all_results:
                all_results[task_name] = []
            all_results[task_name].append(res)
            
            # Print intermediate sample size completion
            print(f"Completed {task_name} (Seed {seed}): Evaluated {res['sample_size']} instances.")

    # Save outputs as NPZ
    import numpy as np
    output_file = "long_bench_pure_vanilla_metrics.npz"
    np.savez_compressed(output_file, data=np.array([json.dumps(all_results)]))
    
    print(f"\nAll evaluations complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
