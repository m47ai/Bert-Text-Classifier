
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):

  def __init__(self, titles, targets, tokenizer, max_len):

    self.titles = titles
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):

    return len(self.titles)

  def __getitem__(self, item):

    title = str(self.titles[item])

    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      title,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt'
      
    )
    return {
      'review_text': title,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long),
      'token_type_ids': encoding['token_type_ids'].flatten()
    }


# class CustomDataset(Dataset):

#     def __init__(self, dataframe, tokenizer, max_len):
#         self.tokenizer = tokenizer
#         self.data = dataframe
#         self.comment_text = dataframe.text
#         self.targets = dataframe.one_hot
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.comment_text)

#     def __getitem__(self, index):
#         comment_text = str(self.comment_text[index])
#         comment_text = " ".join(comment_text.split())

#         inputs = self.tokenizer.encode_plus(
#             comment_text,
#             None,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             pad_to_max_length=True,
#             return_token_type_ids=True
#         )
#         ids = inputs['input_ids']
#         mask = inputs['attention_mask']
#         token_type_ids = inputs["token_type_ids"]


#         return {
#             'ids': torch.tensor(ids, dtype=torch.long),
#             'mask': torch.tensor(mask, dtype=torch.long),
#             'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
#             'targets': torch.tensor(self.targets[index], dtype=torch.float)
#         }