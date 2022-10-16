import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH,return_dict=False)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 6)
        )

    def forward(self, ids, mask, token_type_ids):
        #TODO possible chances of error, check if BERT output size is comparable/correct
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        out = self.classifier(o2)
        return out
