from torch.utils.data import Dataset
import os,torch,json
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class GPT2Dataset(Dataset):
    def __init__(self,tokenizer,max_len,title_max_len,data_dir,data_set_name,path_file=None,is_overwrite=False):
        self.tokenizer=tokenizer
        self.max_len=max_len

        # 从缓存读取已经处理好的数据，或者将处理好的数据存到缓存中
        cached_feature_file=os.path.join(data_dir,"cached_{}_{}".format(data_set_name,max_len))
        if os.path.exists(cached_feature_file) and not is_overwrite:
            self.data_set=torch.load(cached_feature_file)["data_set"]
        else:
            self.data_set=self.load_data(path_file)
            torch.save({"data_set":self.data_set},cached_feature_file)
        
    def load_data(self,path_file):
        self.data_set=[]
        with open(path_file,'r',encoding='utf-8') as f:
            data=json.load(f)
            for idx,sample in enumerate(tqdm(data,desc='iter',disable=False)):   #sample是一个字典，有两个键title,content
                # 切分成token、将token转为ids、截断和补全、返回input_ids,token_type_ids,attention_masks(padding部分为0)
                encoded=self.tokenizer.encode_plus(
                    sample['content'].replace('\n',''),sample['title'],
                    max_length=self.max_len,
                    truncation='only_first',
                    padding='max_length',
                    return_token_type_ids=True,
                    return_tensors='pt',
                )
                input_ids=encoded['input_ids'].reshape(self.max_len,)
                token_type_ids=encoded['token_type_ids'].reshape(self.max_len,)
                
                # 判断input_ids与token_type_ids长度是否一致
                assert len(input_ids) == len(token_type_ids)

                # 判断input_ids长度是否小于等于最大长度
                assert len(input_ids)  <= self.max_len
                self.data_set.append({"input_ids":input_ids,"token_type_ids":token_type_ids})
        return self.data_set   #返回的是整个训练集或者测试集的东西
    '''
                data_set:  
                [
                    {
                        "input_ids":input_ids(tensor),
                        "token_type_ids":token_type_ids(tensor)
                    }
                    ....
                ]
    '''
    
    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]  #返回一个字典
        '''
        {
            "input_ids":input_ids,
            "token_type_ids":token_type_ids})
        }
        '''

