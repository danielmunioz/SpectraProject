import torch.nn.functional as F

def model_eval(local_model, set_loader, device):
  eval_accuracy_sum = 0.0
  items_num = 0
  
  with torch.no_grad():
    for _, batch in enumerate(set_loader):
      data, labels = batch[0].to(device), batch[1].to(device)
      model_out = local_model(data)
      preds = softmax_preds(model_out)

      eval_accuracy_sum += sum(preds == labels).item()
      items_num+=set_loader.batch_size

  return eval_accuracy_sum/items_num
  
def softmax_preds(output):
  return F.softmax(output, dim=1).max(dim=1)[1]