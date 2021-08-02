import torch
from transformers import BertModel

class BERTClass(torch.nn.Module):

    def __init__(self,n_classes,model_name):

        self.n_classes = n_classes
        self.model_name = model_name

        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, n_classes)
    
    def forward(self, ids, mask, token_type_ids):
        output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)['pooler_output']
        print(output_1)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

