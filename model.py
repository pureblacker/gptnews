from transformers import GPT2PreTrainedModel,GPT2Model
import  torch.nn as nn 

class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.transformers = GPT2Model(config)
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.init_weights()
    
    def forward(self,input_ids=None,past=None,token_type_ids=None,labels=None,title_id=None):
        transformer_outputs = self.transformers(input_ids, token_type_ids=token_type_ids)
        hidden_states=transformer_outputs[0]
        lm_logits=self.lm_head(hidden_states)
        outputs=(lm_logits,)+transformer_outputs[1:]
        if labels is not None:
            if title_id is None or token_type_ids is None:
                raise Exception("xxxx")
            mask=(token_type_ids==title_id).long()    
            labels=labels*mask
            shift_logits=lm_logits[...,:-1,:].contiguous()
            shift_labels=labels[...,1:].contiguous()
            loss_fct=nn.CrossEntropyLoss(ignore_index=0,reduction='sum')
            loss=loss_fct(shift_logits.view(-1,shift_logits.size(-1)),shift_labels.view(-1))
            num=shift_labels.ne(0).long().sum().item()
            loss=loss/num
            outputs=(loss,)+outputs
        return outputs #(loss),lm_logits,presents,(all hidden_states) (attention)
