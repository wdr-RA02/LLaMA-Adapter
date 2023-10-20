from datasets import load_dataset
from torch.utils.data import Dataset

from .utils import img_hash_to_addr, collate_test_set, pre_captions

class Personality_Captions(Dataset):
    def __init__(self, vis_root, split_jsonfile,
                 max_len:int=30, **kwargs):
        super().__init__()

        # dataset: load from datasets.load_dataset
        self.annotation=load_dataset("json", data_files=split_jsonfile, split="train")
        # merge additional column into "comment"
        if "additional_comments" in self.annotation.column_names:
            self.annotation=self.annotation.map(collate_test_set, batch_size=128)
        
        # ann_paths <=> config["img_path"]
        self.img_addr=vis_root
        self.img_name_fmt="{}.jpg"

        # others
        self.max_len=max_len


    def __getitem__(self, index):
        sample = self.annotation[index]
        item = img_hash_to_addr(sample, self.img_addr, self.img_name_fmt)
        item["comment"] = pre_captions(item["comment"], self.max_len)
        '''
        item = 
        {
            "images": "....",
            "personality": "<persona>",
            "comment": "..." or ["", *5]
        }
        '''

        return item

    def __len__(self):
        return len(self.annotation)
    
