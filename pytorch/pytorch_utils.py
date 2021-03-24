import torch
from PIL import  Image
from torchvision import transforms

def get_interpolationMode(type):
  return transforms.InterpolationMode if type=='torchvision' else Image if type == 'PIL'else none

class ToTensor(object):
  #custom transform
  def __call__(self, data):
    return torch.from_numpy(data).unsqueeze(0)