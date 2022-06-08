from transformers import GPT2PreTrainedModel,GPT2Model
import  torch.nn as nn 

class MyGPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # self.lm_head为将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)投影为
        # 词典大小维度(config.vocab_size)的输出层, 此时hidden_states张量的形状将会由(batch_size, 1, n_embed)投影变为
        # lm_logits张量的(batch_size, 1, vocab_size). 呸！[batch_size, 1, 1024, 21128]
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.init_weights()
    
    def forward(self, input_ids=None, past_key_values=None, token_type_ids=None, labels=None, title_id=None):
        """
        前向函数，计算GPT2预测结果值
        Args:
            input_ids: 输入序列在词表中的索引序列，size:[batch_size, sequence_length]
            past_key_values: 包含由模型预先计算好的隐藏状态，一般使用在预测阶段，用于加速顺序解码，防止重复计算前面计算过的token
            token_type_ids: 用于区分输入序列中content和title的分隔符序列，size:[batch_size, sequence_length]
            labels: 标签序列，size:[batch_size, sequence_length]，一般情况下，与input_ids相同
            title_id: title部分分隔符的id
        Returns:

        """
        # 获取GPT2模型的输出结果
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, token_type_ids=token_type_ids)
        # 获取GPT2模型的最后一层的隐层节点状态，size:[batch_size, sequence_length, config.n_embd]
        hidden_states = transformer_outputs[0]
        # 预测隐层节点状态中的每一个token的下一个token，size:[batch_size, sequence_length, config.vocab_size]
        lm_logits = self.lm_head(hidden_states)
        # 拼接输出结果
        outputs = (lm_logits,) + transformer_outputs[1:]
        # 如果labels不为None时，计算损失值loss，并拼接到输出结果中
        if labels is not None:
            # 获取mask值，如果token_type_ids中等于title_id的部分需要计算loss，标记为1；否则为0。
            # size:[batch_size, sequence_length]
            mask = (token_type_ids == title_id).long()     #我这里token_type_ids=[0000,111],title_id是1 ,所以这一步实际上mask=token_type_ids,即不需要这一步
            # 获取新的标签，size:[batch_size, 1, sequence_length]
            labels = labels * mask            #labels是content的部分都变成0了，
            # 对预测结果和标签进行偏移操作
            # GPT2的生成机制为通过前面的token，预测下一个token；并且labels与input_ids相同，
            # 因此input_ids中的第一个token的预测结果，实际上是标签中的第二个token，以此类推，最终仅计算sequence_length-1个token的loss
            shift_logits = lm_logits[..., :-1, :].contiguous()  #contiguous()断开赋值的两变量之间的关系。shift_logits改变不影响lm_logits
            # print(shift_logits)
            shift_labels = labels[..., 1:].contiguous()     #[batch,1,1023]

            # 定义损失函数CrossEntropyLoss，并且设置忽略计算loss的索引，以及返回loss的形式
            # 忽略shift_labels中为0的loss，也就是仅计算title部分的损失值
            # 对loss的计算方式设为sum，由于我们仅计算了itle部分的损失值，如果使用mean，会使loss变小（实际除的是sequence_length-1，不是title部分的真实长度）
            loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")    #ignore_index=0忽略labels=0的类的计算
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # 获取title部分的真实长度，并计算真实loss
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


