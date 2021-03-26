import torch
import torch.nn.functional as F
from PIL import  Image
from torchvision import transforms


def get_interpolation_mode(type):
  """exctract interpolation mode from 'torchvision' or 'PIL'
  """
  return transforms.InterpolationMode if type=='torchvision' else Image if type == 'PIL'else None


class ToTensor(object):
  #custom transform
  def __call__(self, data):
    return torch.from_numpy(data).unsqueeze(0)


def softmax_preds(output):
  return F.softmax(output, dim=1).max(dim=1)[1]