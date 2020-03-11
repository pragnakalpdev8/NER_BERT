"""BERT NER Inference."""

from __future__ import absolute_import, division, print_function

import json
import os

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from nltk import word_tokenize
# from transformers import (BertConfig, BertForTokenClassification,
                                #   BertTokenizer)
from pytorch_transformers import (BertForTokenClassification, BertTokenizer)


class BertNer(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        return logits

class Ner:

    def __init__(self,model_dir: str):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        model = BertNer.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        # print("valid positions from text o/p:=>", valid_positions)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # print("input ids with berttokenizer:=>", input_ids)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions

    def predict_entity(self, B_lab, I_lab, words, labels, entity_list):
        temp=[]
        entity=[]

        for word, (label, confidence), B_l, I_l in zip(words, labels, B_lab, I_lab):

            if ((label==B_l) or (label==I_l)) and label!='O':
                if label==B_l:
                    entity.append(temp)
                    temp=[]
                    temp.append(label)
                    
                temp.append(word)

        entity.append(temp)
        # print(entity)

        entity_name_label = []
        for entity_name in entity[1:]:
            for ent_key, ent_value in entity_list.items():
                if (ent_key==entity_name[0]):
                    entity_name_label.append(' '.join(entity_name[1:]) + ": " + ent_value)
        
        return entity_name_label

    def predict(self, text: str):
        input_ids,input_mask,segment_ids,valid_ids = self.preprocess(text)
        # print("valid ids:=>", segment_ids)
        input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        # print("logit values:=>", logits)
        logits = F.softmax(logits,dim=2)
        # print("logit values:=>", logits[0])
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]
        # print("logits label value list:=>", logits_label)

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()
        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        words = word_tokenize(text)

        entity_list = {'B-PER':'Person', 'B-LOC':'Location', 'B-ORG':'Organization', 'B-MISC':'Miscelleneous Entity'}
        
        B_labels=[]
        I_labels=[]
        for label, confidence in labels:
            if (label[:1]=='B'):
                B_labels.append(label)
                I_labels.append('O')
            elif (label[:1]=='I'):
                I_labels.append(label)
                B_labels.append('O')
            else:
                B_labels.append('O')
                I_labels.append('O')

        assert len(labels) == len(words) == len(I_labels) == len(B_labels)

        output = self.predict_entity(B_labels, I_labels, words, labels, entity_list)

        # output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]
        return output


