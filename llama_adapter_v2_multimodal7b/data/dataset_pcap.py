import yaml, torch, copy, random
from PIL import Image
from pathlib import Path
from llama import Tokenizer

from data.dataset import transform_train
from data.personality_captions import PCapDataset, Dataset
from data.personality_captions.utils import pre_captions

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def format_prompt(instruction: str, input_str:str = None):
    ## TODO: test add personality ahead, or add rephrase
    if input_str is not None:
        full_prompt = PROMPT_DICT["prompt_input"].format_map(
            {"instruction": instruction, "input": input_str}
        )
    else:
        full_prompt = PROMPT_DICT["prompt_no_input"].format_map(
            {"instruction": instruction}
        )
    
    return full_prompt


class PCapFinetuneDataset(Dataset):
    def __init__(self, 
                 config_path:str, 
                 max_words:int=30, 
                 transform=transform_train, 
                 tokenizer_path=None, **kwargs):
        
        print("Loading PCapFinetuneDataset from {}".format(config_path))
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        self.persona_token = self.config["persona_token"]

        # read paths from config
        pcap_root = Path(self.config["pcap_root_dir"])
        vis_path = (pcap_root / "yfcc_images").resolve()
        train_json = (pcap_root / "personality_captions/train.json").resolve()
        
        # load prompt lists
        prompt_txt = self.config["prompt_txt"]
        with open(prompt_txt, 'r') as f:
            self.prompt_list = f.read().splitlines()
            print("Prompt library items: {}".format(len(self.prompt_list)))

        self.ann = PCapDataset(str(vis_path), str(train_json), max_words)
        print(f"\ntotal length: {len(self.ann)}")
        self.transform = transform
        self.max_words = max_words
        self.max_words_cap = self.config["max_word_caption"]

        print("\nInit tokenizer...")
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        
    def __len__(self):
        return len(self.ann)

    def persona_to_prompt(self, persona:str):
        prompt = random.choice(self.prompt_list)
        assert self.persona_token in prompt

        prompt = prompt.replace(self.persona_token, persona)

        return prompt

    def __getitem__(self, index):
        data_item = self.ann[index]
        filename = data_item['images']
        question = self.persona_to_prompt(data_item['personality'])
        answer = data_item['comment']
    
        image = Image.open(filename).convert("RGB")
        image = self.transform(image)
        format_instruction = question
        format_input = None

        # TODO: format prompt
        input1 = format_prompt(format_instruction, format_input)
        input2 = input1 + answer

        # debug
        print(input1)
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        
        # mask prompts in labels
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()

        return input2, labels, input2_mask, image


