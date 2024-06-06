import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader

import numpy as np
import more_itertools

## Implemented based on 
# https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies
class SmartBatchingBillingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        #For encoding, we PAD both SOS and EOS. For decoding, we only pad SOS.
        #THe model is required to predict EOS and stop on its own.
        
        #Make sure that padding is not negative (ie the sentance is too long)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")
            
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                # torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype = torch.int64)
            ],
            dim =  0,
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                # torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0,
        )
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                # torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64),
            ],
            dim = 0,
        )
        
        # assert encoder_input.size(0) == self.seq_len
        # assert decoder_input.size(0) == self.seq_len
        # assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            # "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), 
            # encoder mask: (1, 1, seq_len) -> Has 1 when there is text and 0 when there is pad (no text)
            
            # "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            # (1, seq_len) and (1, seq_len, seq_len)
            # Will get 0 for all pads. And 0 for earlier text.
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
            # "encoder_str_length": len(enc_input_tokens),
            # "decoder_str_length": len(dec_input_tokens) 
            }
    
    def get_dataloader(self, batch_size, max_len):
        self.sampler = SmartBatchingSampler(
            encoder_inputs=[self[i]["encoder_input"] for i in range(len(self))],
            batch_size=batch_size
        )
        collate_fn = SmartBatchingCollate(
            # decoder_inputs = [self[i]["decoder_input"] for i in range(len(self))],
            # labels=[self[i]["label"] for i in range(len(self))],
            # src_text = [self[i]["src_text"] for i in range(len(self))]
            # tgt_text = [self[i]["tgt_text"] for i in range(len(self))]
            max_length=max_len,
            pad_token_id=self.pad_token
        )
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            pin_memory=True
        )
        return dataloader

class SmartBatchingSampler(Sampler):
    def __init__(self, encoder_inputs, batch_size):
        super(SmartBatchingSampler, self).__init__(encoder_inputs)
        self.len = len(encoder_inputs)
        sample_lengths = [len(seq) for seq in encoder_inputs]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None
    
    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len
    
    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds
    
class SmartBatchingCollate:
    def __init__(self, max_length, pad_token_id):
        # self._decoder_inputs = decoder_inputs
        # self._labels = labels
        self._max_length = max_length
        self._pad_token_id = pad_token_id
        
    def __call__(self, batch):
        # encoder_inputs_np, decoder_inputs_np, labels_np, src_text, tgt_text = list(zip(*batch))
        
        encoder_inputs_np = []
        decoder_inputs_np = []
        labels_np = []
        src_text = []
        tgt_text = []
        
        for b in batch:
            encoder_inputs_np.append(b["encoder_input"])
            decoder_inputs_np.append(b["decoder_input"])
            labels_np.append(b["label"])
            src_text.append(b["src_text"])
            tgt_text.append(b["tgt_text"])

        encoder_inputs, encoder_mask = self.pad_sequence(
            encoder_inputs_np,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        decoder_inputs, decoder_mask = self.pad_sequence(
            decoder_inputs_np,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        labels, _ = self.pad_sequence(
            labels_np,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )
        
        # if self._targets is not None:
        #     output = input_ids, attention_mask, torch.tensor(targets)
        # else:
        #     output = input_ids, attention_mask
        return {
            "encoder_input":encoder_inputs,
            "decoder_input":decoder_inputs,
            "encoder_mask":encoder_mask,
            "decoder_mask":decoder_mask,
            "label":labels,
            "src_text":src_text,
            "tgt_text":tgt_text
        }

    
    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])
            
            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)
            
            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)
            
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        
        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences, attention_masks

def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int)
    #This will get the upper traingle values
    return mask == 0
    
    
    
