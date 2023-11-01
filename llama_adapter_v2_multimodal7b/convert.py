import os
import argparse
from llama.llama_adapter import LLaMA_adapter
import util.misc as misc
import util.extract_adapter_from_checkpoint as extract

if __name__=="__main__":
    '''your code here'''
    parser = argparse.ArgumentParser("LLaMA_Adapter Checkpoint saver")
    parser.add_argument("--llama_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--type", type=str, choices=["BIAS", "LORA"], default="BIAS")
    args = parser.parse_args()

    llama_dir = args.llama_dir
    lora = (args.type == "LORA")
    bias = (args.type == "BIAS")
    llama_type = '7B'
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path, 
                          w_lora=lora, w_new_gate=lora, w_bias=bias)

    print("Load checkpoint from {}".format(args.checkpoint))
    misc.load_model(model, args.checkpoint)
    model.eval()

    output_filename = os.path.join(args.output_dir, "adapter_{}_{}.pth".format(args.type, llama_type))
    extract.save(model, output_filename, args.type)

    print("Saved model to {}".format(output_filename))