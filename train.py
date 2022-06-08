import torch
import os
import random
import numpy as np
import argparse
import logging
from transformers import GPT2Config,GPT2LMHeadModel,BertTokenizer,AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from model import  MyGPT2LMHeadModel
from data_set import GPT2Dataset
from rouge import Rouge

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train(model,device,train_data,test_data,args):
    tb_write=SummaryWriter()
    if args.gradient_accumulation_steps<1:
        raise ValueError("Gradient accumulation参数")
    train_batch_size=int(args.train_batch_size/args.gradient_accumulation_steps)
    train_sampler=RandomSampler(train_data)
    train_data_loader=DataLoader(train_data,sampler=train_sampler,
                                batch_size=train_batch_size)
    total_steps=int(len(train_data_loader)*args.num_train_epochs/args.gradient_accumulation_steps)
    logger.info("总训步数{}".format(total_steps))
    model.to(device)
    param_optimizer=list(model.named_parameters())
    no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer=AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    torch.cuda.empty_cache()
    model.train()
    title_id=1   #..........................................
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step=0
    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].reshape(train_batch_size,args.n_ctx).to(device)
            token_type_ids = batch["token_type_ids"].reshape(train_batch_size,args.n_ctx).to(device)
            # 获取训练结果
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            loss = outputs[0]
            tr_loss += loss.item()
            # 将损失值放到Iter中，方便观察
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # 如果步数整除logging_steps，则记录学习率和训练集损失值
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_write.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_write.add_scalar("train_loss", (tr_loss-logging_loss) /
                                        (args.logging_steps*args.gradient_accumulation_steps), global_step)
                    logging_loss = tr_loss
                # 如果步数整除eval_steps，则进行模型测试，记录测试集的损失
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, device, test_data, args)
                    tb_write.add_scalar("test_loss", eval_loss, global_step)
                    model.train()

        # 每个epoch进行完，则保存模型
        if iepoch%10==0:
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()

def evaluate(model, device, test_data, args):
    """
    对测试数据集进行模型测试
    Args:
        model: 模型
        device: 设备信息
        test_data: 测试数据类
        args: 训练参数配置信息
    Returns:
    """
    tokenizer=BertTokenizer.from_pretrained(args.vocab_path)
    length=40
    # 构造测试集的DataLoader
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    title_id = 1            #....................

    repetition_penalty=1.2
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    model.eval()

    rougescore1=[]
    rougescore2=[]
    rougescorel=[]
    with torch.no_grad():
        for idx in range(len(test_data)):
            token_type_id = test_data[idx]["token_type_ids"].reshape(args.max_len,)
            input_id=test_data[idx]["input_ids"].reshape(args.max_len,)
            for j in range(args.max_len):
                if token_type_id[j]==1:
                    break
            for k in range(j,args.max_len):
                if token_type_id[k]==0:
                    break
            label_id=input_id[j:k].contiguous()
            input_id=input_id[:j].contiguous()

            generated_ids=[]
            for _ in range(length):
                # 获取预测结果
                outputs = model.forward(input_ids=input_id)
                # 获取预测文本
                next_token_logits = outputs[0][-1,:]
                already_token_ids =set([ids for ids in generated_ids])
                for token_id in already_token_ids:
                    next_token_logits[token_id] /= repetition_penalty
                
                next_token_logits[unk_id] = -float("Inf")
                filter_logits = top_k_top_p_filtering(next_token_logits)
                next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)   #multinomial对张量的每一行进行num_samples次取样
                if next_tokens==sep_id:
                    break
                generated_ids.append(next_tokens.item())
                input_ids = torch.cat((input_id, next_tokens), dim=-1)
                if len(input_id)>=1024:
                    input_id=input_id[-1023:]  
            predict="".join(tokenizer.convert_ids_to_tokens(generated_ids)).replace("##", "").replace("[SEP]", " ").replace("[UNK]", "") 
            label="".join(tokenizer.convert_ids_to_tokens(label_id)).replace("##", "").replace("[UNK]", "").replace("[SEP]", " ")  
            
            rouge = Rouge()
            rouge_score = rouge.get_scores(predict, label)
            rougescore1.append(rouge_score[0]["rouge-1"]['f'])
            rougescore2.append(rouge_score[0]["rouge-2"]['f'])
            rougescorel.append(rouge_score[0]["rouge-l"]['f'])
    return (sum(rouge_score1)/len(rouge_score1),sum(rouge_score2)/len(rouge_score2),sum(rouge_scorel)/len(rouge_scorel))

def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--config_path', default='model/config.json', type=str, help='模型参数配置信息')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--train_file_path', default='data_dir/train.json', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--test_file_path', default='data_dir/dev.json', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--pretrained_model_path', default='model', type=str, help='预训练的GPT2模型的路径')
    parser.add_argument('--data_dir', default='data_dir', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=4, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=8, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='模型训练时的学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')
    parser.add_argument('--eval_steps', default=4000, type=int, help='训练时，多少步进行一次测试')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=2020, help='随机种子')
    parser.add_argument('--max_len', type=int, default=1024, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--title_max_len', type=int, default=32, help='生成标题的最大长度，要比max_len小')
    return parser.parse_args()

def main():
    # 设置模型训练参数
    args = set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # 设置随机种子，方便模型复现
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # 加载模型的config
    model_config = GPT2Config.from_json_file(args.config_path)
    # 实例化GPT2LMHeadModel模型，这里我们没有加载预训练好的模型，而是直接从头开始训练。
    # 为什么从头开始训练？我们采用的是小模型，只有6层，并且词表也做了修改，没有找到合适的预训练模型。（其实是，穷人，卡不行。）
    # 判断是否使用预训练好的GPT2模型
    if args.pretrained_model_path:
        model = MyGPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
    else:
        # 如果没有指定的预训练模型，则初始化模型
        model = MyGPT2LMHeadModel(config=model_config)
    # model = GPT2LMHeadModel(config=model_config)
    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    # 将[space]作为一个分割整体，例如："我爱[Space]中国。"，使用原始tokenizer分词结果为"['我', '爱', '[', 'Space', ']', '中', '国', '。']";
    # 增加分割符号后的结果为"['我', '爱', '[Space]', '中', '国', '。']"
    tokenizer.add_tokens("[Space]", special_tokens=True)
    # 创建模型的输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 加载训练数据和测试数据
    train_data = GPT2Dataset(tokenizer, args.max_len, args.title_max_len, args.data_dir, "train", args.train_file_path)
    test_data = GPT2Dataset(tokenizer, args.max_len, args.title_max_len, args.data_dir, "test", args.test_file_path)
    # 开始训练
    train(model, device, train_data, test_data, args)



def top_k_top_p_filtering(logits, filter_value=-float("Inf")):
    
    top_k=5
    top_p=0.95
    assert logits.dim() == 1   # logits的维度为1，size:[vocab_size]

    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]    #< 后面表示的是topk里面最小的值
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)      #[0.3,0.35,0.4,..,0.95,0.951,0.952,...1]
        sorted_indices_to_remove = cumulative_probs > top_p                            #删除累积概率高于top_p的标记，[false,....,true,true,true,true,true]
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() #不加省略号也可以，加是为了当生成多个结果时
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]    
        logits[indices_to_remove] = filter_value
    return logits

if __name__ == '__main__':
    main()
