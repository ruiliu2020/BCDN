import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
import json

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class MABSADataset(Dataset):
    def __init__(self, fname, tokenizer, imagefeat_dir, num_roi_boxes=100):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        all_token_num = 0
        all_sent_num = 0
        max_sen = 0
        
        for i in range(0, len(lines), 4):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            img_id = lines[i + 3].strip()
            
            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            
            all_token_num += np.sum(text_raw_indices != 0)
            max_sen = max(np.sum(text_raw_indices != 0),max_sen)

            polarity = int(polarity) + 1

            src_mask = [0] + [1] * (np.sum(text_raw_indices != 0)) + [0] * (tokenizer.max_seq_len - (np.sum(text_raw_indices != 0)) - 1)
            src_mask = src_mask[:tokenizer.max_seq_len]
            src_mask = np.asarray(src_mask, dtype='int64')

            text_left_indices = tokenizer.text_to_sequence(text_left)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            left_context_with_aspect_len = np.sum(text_left_with_aspect_indices != 0)
            aspect_mask = [0] + [0] * left_context_len + [1] * aspect_len + [0] * (tokenizer.max_seq_len - left_context_with_aspect_len - 1)
            aspect_mask = aspect_mask[:tokenizer.max_seq_len]
            aspect_mask = np.asarray(aspect_mask, dtype='int64')

            if left_context_len + 1 > tokenizer.max_seq_len:
                aspect_mask[0] = 1
        
            second = aspect
            second_raw_indices = tokenizer.text_to_sequence(second.lower())
            second_len = np.sum(second_raw_indices != 0)

            all_text = '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect +' [SEP]'
            text_bert_indices = tokenizer.text_to_sequence(all_text)

            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (second_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)       

            bert_attention_mask = np.asarray([1] * (np.sum(text_raw_indices != 0) + 2 + second_len + 1))
            bert_attention_mask =  pad_and_truncate(bert_attention_mask, tokenizer.max_seq_len)  

            img_feat = np.load(imagefeat_dir+'/'+str(img_id)+'.npy')
            image = np.load(imagefeat_dir+'_full/'+str(img_id)+'.npy')
        
            if img_feat.shape[0]<num_roi_boxes:
                img_feat = np.append(img_feat,np.zeros((num_roi_boxes-img_feat.shape[0],img_feat.shape[1])),axis=0)
            elif img_feat.shape[0]>num_roi_boxes:
                img_feat = img_feat[:num_roi_boxes,:]
            all_sent_num += 1

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'bert_attention_mask': bert_attention_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
                'src_mask': src_mask,
                'img_feat': img_feat,
                'full_img_feat': image
            }
            all_data.append(data)
        self.data = all_data
        print(max_sen,all_token_num/all_sent_num)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
