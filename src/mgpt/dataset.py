import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

class AstroDataset(Dataset):
    """Dataset class for the AstroGPT model.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, txt_data, 
                 tokenizer_name='gpt2', 
                 max_length=32,
                 stride=1,
                 batch_size=4):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_length = max_length
        self.data = txt_data
        self.stride = stride
        self.batch_size = batch_size
        self.token_ids = self.tokenizer.encode(self.data)
        self.input_ids, self.target_ids = self.create_chunks()
        
    def create_chunks(self):
        inputs = []
        targets = []
        for i in range(0, len(self.token_ids) - self.max_length, self.stride):
            input_chunk = self.token_ids[i:i+self.max_length]
            target_chunk = self.token_ids[i+1:i+self.max_length+1]
            inputs.append(torch.tensor(input_chunk))
            targets.append(torch.tensor(target_chunk))
        return inputs, targets
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    def create_dataloader(self, shuffle=True, drop_last=True,
                          num_workers=0):
        """Return a PyTorch DataLoader for this dataset."""
        dataloader = DataLoader(self, batch_size=self.batch_size,
                                shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers)
        return dataloader
    
    def data_decoder(self, token_ids):
        """Decodes token ids back to text words.
        """
        decoded_texts = self.tokenizer.decode(token_ids)
        return decoded_texts

    
