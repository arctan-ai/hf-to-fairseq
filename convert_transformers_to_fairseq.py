import os
import argparse

import torch
from transformers import HubertModel, HubertConfig

# # ---- paths ----
# hf_dir = r"D:\models\distilhubert"  # folder with config.json + pytorch_model.bin
# out_fairseq = r"D:\models\fairseq_distilhubert.pt"


def convert_to_fairseq(hf_dir, out_fairseq_path):
    # 1) load HF model
    config = HubertConfig.from_pretrained(hf_dir)
    hf_model = HubertModel.from_pretrained(hf_dir, config=config)
    hf_sd = hf_model.state_dict()

    # 2) your mapping (HF_key -> fairseq_key in your script)
    mapping_hf_to_fair = {
        "masked_spec_embed": "encoder.mask_emb",
        "encoder.layer_norm.bias": "encoder.layer_norm.bias",
        "encoder.layer_norm.weight": "encoder.layer_norm.weight",
        "encoder.pos_conv_embed.conv.bias": "encoder.pos_conv.0.bias",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0": "encoder.pos_conv.0.weight_g",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1": "encoder.pos_conv.0.weight_v",
        "feature_projection.layer_norm.bias": "encoder.layer_norm.bias",
        "feature_projection.layer_norm.weight": "encoder.layer_norm.weight",
        "feature_projection.projection.bias": "encoder.post_extract_proj.bias",
        "feature_projection.projection.weight": "encoder.post_extract_proj.weight",
    }
    
    for layer in range(12):
        for j in ["q", "k", "v"]:
            mapping_hf_to_fair[
                f"encoder.layers.{layer}.attention.{j}_proj.weight"
            ] = f"encoder.layers.{layer}.self_attn.{j}_proj.weight"
            mapping_hf_to_fair[
                f"encoder.layers.{layer}.attention.{j}_proj.bias"
            ] = f"encoder.layers.{layer}.self_attn.{j}_proj.bias"

        mapping_hf_to_fair[
            f"encoder.layers.{layer}.final_layer_norm.bias"
        ] = f"encoder.layers.{layer}.final_layer_norm.bias"
        mapping_hf_to_fair[
            f"encoder.layers.{layer}.final_layer_norm.weight"
        ] = f"encoder.layers.{layer}.final_layer_norm.weight"

        mapping_hf_to_fair[
            f"encoder.layers.{layer}.layer_norm.bias"
        ] = f"encoder.layers.{layer}.self_attn_layer_norm.bias"
        mapping_hf_to_fair[
            f"encoder.layers.{layer}.layer_norm.weight"
        ] = f"encoder.layers.{layer}.self_attn_layer_norm.weight"

        mapping_hf_to_fair[
            f"encoder.layers.{layer}.attention.out_proj.bias"
        ] = f"encoder.layers.{layer}.self_attn.out_proj.bias"
        mapping_hf_to_fair[
            f"encoder.layers.{layer}.attention.out_proj.weight"
        ] = f"encoder.layers.{layer}.self_attn.out_proj.weight"

        mapping_hf_to_fair[
            f"encoder.layers.{layer}.feed_forward.intermediate_dense.bias"
        ] = f"encoder.layers.{layer}.fc1.bias"
        mapping_hf_to_fair[
            f"encoder.layers.{layer}.feed_forward.intermediate_dense.weight"
        ] = f"encoder.layers.{layer}.fc1.weight"

        mapping_hf_to_fair[
            f"encoder.layers.{layer}.feed_forward.output_dense.bias"
        ] = f"encoder.layers.{layer}.fc2.bias"
        mapping_hf_to_fair[
            f"encoder.layers.{layer}.feed_forward.output_dense.weight"
        ] = f"encoder.layers.{layer}.fc2.weight"

    for layer in range(7):
        mapping_hf_to_fair[
            f"feature_extractor.conv_layers.{layer}.conv.weight"
        ] = f"encoder.feature_extractor.conv_layers.{layer}.0.weight"

        if layer == 0:
            mapping_hf_to_fair[
                f"feature_extractor.conv_layers.{layer}.layer_norm.weight"
            ] = f"encoder.feature_extractor.conv_layers.{layer}.2.weight"
            mapping_hf_to_fair[
                f"feature_extractor.conv_layers.{layer}.layer_norm.bias"
            ] = f"encoder.feature_extractor.conv_layers.{layer}.2.bias"

    # 3) build fairseq-style state_dict by reversing your direction
    fair_sd = {}
    missing_in_hf = []

    print(f"keys in the hf_checkpoint : ")
    
    for hf_k in hf_sd.keys():
        print(hf_k)

    for hf_k, fair_k in mapping_hf_to_fair.items():
        print(f"hf key: {hf_k}\t fairseq key: {fair_k}")
        if hf_k in hf_sd:
            fair_sd[fair_k] = hf_sd[hf_k].clone()
        else:
            missing_in_hf.append(hf_k)

    # 4) save a fairseq checkpoint skeleton
    ckpt = {
        "model": fair_sd,
        # keep HF config for reference; fairseq normally expects hydra cfg here
        "cfg": {
            "model": config.to_dict()
        },
        "task_state": {},
        "optimizer_history": [],
        "extra_state": {},
    }
    torch.save(ckpt, out_fairseq_path)

    print("Wrote:", out_fairseq_path)
    if missing_in_hf:
        print("HF keys missing for mapping:")
        for value in missing_in_hf:
            print(f"{value}")

def test_load_fairseq(fairseq_ckpt_path):
    from fairseq.checkpoint_utils import load_model_ensemble_and_task

    try:
        models, cfg, task = load_model_ensemble_and_task([fairseq_ckpt_path])
        model = models[0].eval()

    except Exception as e:
        print(f"Unable to load the converted model due to exception - {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_hf_dir', type=str, default='./hubert')
    parser.add_argument('--output_fairseq_path', type=str, default="./hubert.pt")
    args = parser.parse_args()

    convert_to_fairseq(args.input_hf_dir, args.output_fairseq_path)

    test_load_fairseq(args.output_fairseq_path)

    print("Successfully converted the checkpoint to fairseq.")