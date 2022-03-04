from torch.utils.data import Dataset
import os,torch,json
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class GPT2Dataset(Dataset):
    def __init__(self,tokenizer,max_len,title_max_len,data_dir,data_set_name,path_file=None,is_overwrite=False):
        self.tokenizer=tokenizer

        self.abs_id=self.tokenizer.convert_tokens_to_ids("[Abst]")
        self.title_id=self.tokenizer.convert_tokens_to_ids("[Title]")
        self.space_id=self.tokenizer.convert_tokens_to_ids("[Space]")
        self.max_len=max_len
        self.title_max_len=title_max_len

        cached_feature_file=os.path.join(data_dir,"cached_{}_{}".format(data_set_name,max_len))
        if os.path.exists(cached_feature_file) and not is_overwrite:
            self.data_set=torch.load(cached_feature_file)["data_set"]
        else:
            self.data_set=self.load_data(path_file)
            torch.save({"data_set":self.data_set},cached_feature_file)
        
    def load_data(self,path_file):
        self.data_set=[]
        with open(path_file,'r',encoding='utf-8') as fh:
            data=json.load(fh)
            for idx,sample in enumerate(tqdm(data,desc='iter',disable=False)):
                input_ids,token_type_ids=self.convert_features(sample)
                self.data_set.append({"input_ids":input_ids,"token_type_ids":token_type_ids})
        return self.data_set   #返回的是整个训练集或者测试集的东西
    '''
    dataset:  
    [
        {
            "input_ids":input_ids,
            "token_type_ids":token_type_ids})
        }
        {
            "input_ids":input_ids,
            "token_type_ids":token_type_ids})
        }
        ....
    ]
    '''


    def convert_features(self,sample:dict):
        '''
        sample: 一个字典，包含新闻的正文和新闻的标题，格式为{"abst": abst, "title": title}
        '''
        input_ids=[]
        token_type_ids=[]
        abst_tokens=self.tokenizer.tokenize(sample['abst'])
        title_tokens=self.tokenizer.tokenize(sample['title'].replace(" ","[Space]"))
        if len(title_tokens)>self.title_max_len:
            title_tokens=title_tokens[:self.title_max_len]
        if len(abst_tokens)>self.max_len-len(title_tokens)-3:
            abst_tokens=abst_tokens[:self.max_len-len(title_tokens)-3]
        
        input_ids.append(self.tokenizer.cls_token_id)
        token_type_ids.append(self.abs_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(abst_tokens))
        token_type_ids.extend([self.abs_id]*len(abst_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.abs_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(title_tokens))
        token_type_ids.extend([self.title_id] * len(title_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.title_id)
        # 判断input_ids与token_type_ids长度是否一致
        assert len(input_ids) == len(token_type_ids)
        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= self.max_len
        return input_ids, token_type_ids  #返回的是一个样本的东西
    
    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance




def collate_func(batch_data):
    from torch.nn.utils.rnn import pad_sequence
    batch_size=len(batch_data)
    if batch_size==0:
        return {}
    input_ids_list, token_type_ids_list=[],[]
    for instance in batch_data:
        input_ids_temp=instance["input_ids"]
        token_type_ids_temp=instance["token_type_ids"]
        input_ids_list.append(torch.tensor(input_ids_temp,dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp,dtype=torch.long))

    return {
            "input_ids":pad_sequence(input_ids_list,batch_first=True,padding_value=0),
            "token_type_ids":pad_sequence(token_type_ids_list,batch_first=True,padding_value=0)
        }

    '''
        这个input_ids: 
        [
            tensor[1abstokenids+titletokenids]  #这是一个样本
            tensor[1abstokenids+titletokenids]
            ...
        ]
    '''