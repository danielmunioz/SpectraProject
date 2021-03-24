import numpy as np
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader

class H5_dataset(Dataset):
  """Extracts data from hdf5 datasets
  flare dataset: https://www.kaggle.com/muniozdaniel0/flares-set-240x210
  non_flare dataset: https://www.kaggle.com/muniozdaniel0/non-flares-set-200x240-repaired
  """
  def __init__(self, hdf5_flares, hdf5_non_flares, split = 'train', transform = None, clip = True):
    self.flare_data = hdf5_flares
    self.nonflare_data = hdf5_non_flares
    self.transform = transform
    self.clip = clip
    pre_keys = [(1, key) for key in list(hdf5_flares.keys())] + [(0, key) for key in list(hdf5_non_flares.keys())]

    if split is not None:
      self.keys = self._split(pre_keys, split)
    else:
      self.keys = pre_keys

  def __len__(self):
    return len(self.keys)

  def __getitem__(self, idx):
    label, local_key = self.keys[idx]
    if label == 1:
      local_data = self.flare_data[local_key][:,:].astype(np.float32)
    else:
      local_data = self.nonflare_data[local_key][:,:].astype(np.float32)

    if self.clip:
      local_data = np.clip(local_data, a_min=0.0, a_max=None)
    if self.transform:
      local_data = self.transform(local_data)
    
    return local_data, label
  
  def _split(self, keys_list, split_mode):
    keys_list.sort()  #fixed orded
    random.seed(230) #fixed seed
    random.shuffle(keys_list) #therefore fixed shuffle

    split_1 = int(0.8*len(keys_list))
    split_2 = int(0.9*len(keys_list))
    if split_mode == 'train':
      return keys_list[:split_1]
    elif split_mode =='dev':
      return keys_list[split_1:split_2]
    elif split_mode =='test':
      return keys_list[split_2:]
    else:
      raise Exception('need a valid split mode')


class Test_Dataset(Dataset):
  """Designed for testing given a list of samples of format 'item_dir, start, end, label'
  """
  def __init__(self, samples_list, transform=None, clip=True):
    self.local_list = samples_list
    self.transform = transform
    self.clip = True
  
  def __len__(self):
    return len(self.local_list)

  def __getitem__(self, idx):
    item, start, end, label = self.local_list[idx]
    data = fits.open(item)[0].data[:, int(start):int(end)].astype(np.float32)
    if self.clip:
      data = np.clip(data, a_min=0, a_max=None)
    if self.transform:
      data = self.transform(data)
    return data, int(label)


class Field_Dataset(Dataset):
  """Designed for inference
  """
  def __init__(self, files_dir, window_length, transform = None, clip=True):
    self.data = self._create_samples(files_dir, window_length)
    self.transform = transform
    self.clip = clip

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item, start, end = self.data[idx]
    data = fits.open(item)[0].data[:, start:end].astype(np.float32)

    if self.clip:
      data = np.clip(data, a_min=0.0, a_max=None)
    if self.transform:
      data = self.transform(data)

    return data, item
  
  def _create_samples(self, files_dir, window_length):
    if files_dir[-1] !='/':
      files_dir = files_dir + '/'

    non_root = files_dir
    non_flare_list = [non_root + elemen for elemen in os.listdir(files_dir)]

    big_list = []

    for elemen in non_flare_list:
      local_start = 0
      local_list= [ ]
      local_shape = fits.open(elemen)[0].data.shape
      total_pieces = math.floor(local_shape[1]/window_length)

      for local in range(total_pieces):
        local_end = local_start+window_length

        print(elemen, local_start, local_end)
        local_list.append((elemen, local_start, local_end))
        
        local_start = local_end
    
      big_list = big_list + local_list
    return big_list