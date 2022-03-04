def clean_title(title:str):
    return title

def clean_abs(abs:str):
    return abs

def clean_data(sample:tuple):
    (abs,title)=sample
    sample=dict()
    sample['title']=clean_title(title.strip())
    sample['abs']=clean_abs(abs.strip())
    return sample

def build_data(data_path,train_save_path,test_save_path):
    import json
    import random
    data=[]
    with open(data_path,'r',encoding='utf-8') as f:
        for line in f:
            d=json.loads(line)
            data.append({'abst':d['abst'],'title':d['title']})

    data_set=set()
    for d in data:
        if d['abst'] in data_set or len(d['abst'])<50 or len(d['title'])<2:
            continue
        else:
            data_set.add(d['abst'])
    random.shuffle(data)
    train_data=data[:3001]
    test_data=data[3001:]
    fin = open(train_save_path, "w", encoding="utf-8")
    fin.write(json.dumps(train_data, indent=4, ensure_ascii=False))
    fin.close()
    fin = open(test_save_path, "w", encoding="utf-8")
    fin.write(json.dumps(test_data, indent=4, ensure_ascii=False))
    fin.close()

if __name__ == '__main__':
    data_path='gpt_news\csl_title_public\csl_data.json'
    train_save_path_dir = "gpt_news/data_dir/train_data.json"
    test_save_path_dir = "gpt_news/data_dir/test_data.json"
    build_data(data_path,train_save_path_dir, test_save_path_dir)
    