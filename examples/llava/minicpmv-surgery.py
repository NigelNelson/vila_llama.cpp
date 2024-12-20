import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Literal, TypedDict, Dict

from llava.model import *

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to MiniCPM-V or VILA model")
args = ap.parse_args()

ModelType = Literal["vila", "minicpmv"]

class ProjectorConfig(TypedDict):
    prefix: str
    output_name: str

MODEL_CONFIGS: Dict[ModelType, ProjectorConfig] = {
    "vila": {
        "mm_prefix": "mm_projector",
        "output_projector_name": "vila.projector",
        "clip_prefix": "vision_tower.vision_tower.vision_model.",
    },
    "minicpmv": {
        "mm_prefix": "resampler",
        "output_projector_name": "minicpmv.projector",
        "clip_prefix": "vpm.",
    }
}

model_type: ModelType = "vila" if "vila" in args.model else "minicpmv"
config = MODEL_CONFIGS[model_type]

# find the model part that includes the the multimodal projector weights
model = AutoModel.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
checkpoint = model.state_dict()

# get a list of mm tensor names
mm_tensors = [k for k, v in checkpoint.items() if k.startswith(config["mm_prefix"])]

# store these tensors in a new dictionary and torch.save them
projector = {name: checkpoint[name].float() for name in mm_tensors}
torch.save(projector, f"{args.model}/{config['output_projector_name']}")

clip_tensors = [k for k, v in checkpoint.items() if (k.startswith(config["clip_prefix"]) and "head" not in k)]
if len(clip_tensors) > 0:
    clip = {name.replace(config["clip_prefix"], ""): checkpoint[name].float() for name in clip_tensors}
    torch.save(clip, f"{args.model}/{model_type}.clip")

    # added tokens should be removed to be able to convert Mistral models
    if os.path.exists(f"{args.model}/added_tokens.json"):
        with open(f"{args.model}/added_tokens.json", "w") as f:
            f.write("{}\n")

# config = model.llm.config
# if model_type == "minicpmv":
#     config.auto_map = {
#         "AutoConfig": "configuration_minicpm.MiniCPMConfig",
#         "AutoModel": "modeling_minicpm.MiniCPMModel",
#         "AutoModelForCausalLM": "modeling_minicpm.MiniCPMForCausalLM",
#         "AutoModelForSeq2SeqLM": "modeling_minicpm.MiniCPMForCausalLM",
#         "AutoModelForSequenceClassification": "modeling_minicpm.MiniCPMForSequenceClassification"
#     }
# else:
#     config.auto_map = {
#         "AutoConfig": "configuration_llama.LlamaConfig",
#         "AutoModel": "modeling_llama.LlamaModel",
#         "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM",
#         "AutoModelForSeq2SeqLM": "modeling_llama.LlamaForCausalLM",
#         "AutoModelForSequenceClassification": "modeling_llama.LlamaForSequenceClassification"
#     }

if model_type == "minicpmv":
    model.llm.save_pretrained(f"{args.model}/model")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.save_pretrained(f"{args.model}/model")

print("Done!")
print(f"Now you can convert {args.model if model_type == 'minicpmv' else args.model + '/llm'} to a regular LLaMA GGUF file.")
print(f"Also, use {args.model}/{model_type}.projector to prepare a {model_type}-encoder.gguf file.")


# ╰─ PYTHONPATH=${PYTHONPATH}:/media/m2/holoscan-dev/holoscan-ml/multi-modality-research/VILA python 
