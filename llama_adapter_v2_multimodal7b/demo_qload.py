import llama
import torch
import json
import argparse
import os.path as osp

from PIL import Image
from typing import List, Dict
from llama.llama_adapter import LLaMA_adapter

device = "cuda" if torch.cuda.is_available() else "cpu"

def load(name, llama_dir, llama_type="7B", device=device, 
         max_batch_size: int=1, quant_bits="no"):
   
    if osp.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found"), None

    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    # llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = osp.join(llama_dir, llama_type)
    llama_tokenzier_path = osp.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})
    print(model_cfg)

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=512, max_batch_size=max_batch_size,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        w_new_gate=model_cfg.get('w_lora', False), # for compatibility
        quant_bits=quant_bits,
        phase="finetune")

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device), model.clip_transform


def batch_inference(model, preprocess, device, imgs: List[str], personas: List[str]):
    assert len(imgs)==len(personas), f"Length inequal, get imgs={len(imgs)} and personas={len(personas)}"
    model.eval()
    # build prompt lists
    p_base = llama.format_prompt("Write a comment of this image in the context of a given personality trait: <persona>.")
    prompts = [p_base.replace("<persona>", persona) for persona in personas]

    # build imgs
    img_pil = [Image.open(im).convert("RGB") for im in imgs]
    imgs_ = torch.stack([preprocess(im) for im in img_pil], dim=0).to(device)

    result = model.generate(img_pil, prompts)[0]

    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--llama_dir", type=str, required=True)
    parser.add_argument("--inference_src", type=str, required=True)
    args=parser.parse_args()

    '''your code here'''
    llama_dir = args.llama_dir
    # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    model, preprocess = load(args.checkpoint, llama_dir, device=device, quant_bits="8bit")
    model.eval()

    # prompt = llama.format_prompt("Write a comment of this image in the context of a given personality trait: <persona>.")
    # prompt = prompt.replace("<persona>", args.persona)

    # img = Image.open(args.img).convert("RGB")
    # img = preprocess(img).unsqueeze(0).to(device)

    # result = model.generate(img, [prompt])[0]

    # load inference_src
    with open(args.inference_src) as fp:
        inf_src = json.load(fp)
    
    # get image root: $(pwd)/data/PCap/yfcc_images
    cur_path = osp.split(osp.abspath(__file__))[0]
    img_root = osp.join(cur_path, "data", "PCap", "yfcc_images")

    result = batch_inference(model, preprocess, device, )
    print(result)
