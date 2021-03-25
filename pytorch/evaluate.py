import os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets import Test_Dataset
from pytorch_utils import ToTensor, get_interpolation_mode
from SpectraProject.utils.evaluate_utils import plot_confusion_matrix


def Evaluate(model, weights_path, samples_list_dir, data_root, interpolation_mode):
  IntMode = interpolation_mode
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  with open(samples_list_dir, 'r') as data_txt:
    data_lines = data_txt.readlines()
  samples_list = [elemen.strip().split('\t') for elemen in data_lines]
  samples_list = [(os.path.join(data_root, item), st, en, lbl) for item, st, en, lbl in samples_list]

  #creates dataset
  test_dset = Test_Dataset(samples_list, 
                          transform=transforms.Compose([ToTensor(), 
                                                        transforms.Resize((200,240), interpolation=IntMode.NEAREST)]))
  test_dloader = DataLoader(test_dset, batch_size=50)
  
  model.to(device)
  model.load_state_dict(torch.load(weights_path, map_location=device))
  model.eval()
  gbl_preds, gbl_labels, acc = model_eval(model, test_dloader, device)

  confussion_matrix = torch.zeros(2,2, dtype=torch.int32)
  stacked = torch.stack((gbl_preds, gbl_labels), dim=1)
  for elemen in stacked:
    tpred, lpred = elemen.tolist()
    confussion_matrix[tpred, lpred] += 1

  plt.figure(figsize=(10,10))
  plot_confusion_matrix(confussion_matrix, ['no flare', 'flare'])


def model_eval(local_model, set_loader, device):
  eval_accuracy_sum = 0.0
  items_num = 0

  global_preds = torch.tensor([], dtype=torch.int).to(device)
  global_labels = torch.tensor([], dtype=torch.int).to(device)
  with torch.no_grad():
    for _, batch in enumerate(set_loader):
      data, labels = batch[0].to(device), batch[1].to(device)
      model_out = local_model(data)
      preds = softmax_preds(model_out)

      global_preds = torch.cat((global_preds, preds))
      global_labels = torch.cat((global_labels, labels))

      eval_accuracy_sum += sum(preds == labels).item()
      items_num+=set_loader.batch_size

      print(items_num, ' total accuracy: ',eval_accuracy_sum/items_num)

  final_acc = eval_accuracy_sum/items_num
  return global_preds, global_labels, final_acc


def softmax_preds(output):
  return F.softmax(output, dim=1).max(dim=1)[1]


