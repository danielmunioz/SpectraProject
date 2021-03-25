import math
import numpy as np
from astropy.io import fits
from sunpy.time import parse_time
from radiospectra.sources.callisto import query

#background subtraction methods
def auto_find_background(data, amount=0.05):
  tmp = (data - np.average(data, 1).reshape(data.shape[0], 1))
  sdevs = np.asarray(np.std(tmp, 0))
  cand = sorted(range(data.shape[1]), key=lambda y: sdevs[y])
  return cand[:max(1, int(amount * len(cand)))]

def auto_const_bg(data):
  realcand = auto_find_background(data)
  bg = np.average(data[:, realcand], 1)
  return bg.reshape(data.shape[0], 1)

def subtract_bg(data):
  """Perform constant background subtraction"""
  return data - auto_const_bg(data)

def subtract_downloader(local_url, download_dir):
  """Download and perform constant background subtraction"""
  base_name = os.path.basename(local_url)
  local_fits = fits.open(local_url)
  local_fits[0].data = subtract_bg(local_fits[0].data)
  local_fits.writeto(download_dir+base_name)
  return base_name


def read_txt(dir_txt):
  #extracts contents from a txt file 
  #format: date\ttime_start\ttime_end\tfile_dir or date\ttime_interval\tftype\tfile_dir
  with open(dir_txt, 'r') as data_txt:
    data_lines = data_txt.readlines()
  return [elemen.strip().split('\t') for elemen in data_lines]


def mask_zeros(file_array, new_lenght):
  mask = np.zeros((file_array.shape[0], new_lenght))
  mask[:, :file_array.shape[1]] = file_array
  return mask


def get_delt(header):
  """Extracts t_delt from fits file"""
  #swapped = 'time' not in header['CTYPE1'].lower()
  if 'time' not in header['CTYPE1'].lower():
    t_delt = header['CDELT2']
  else:
    t_delt = header['CDELT1']
  return t_delt #seconds/pixels


def download_subtracted_fits(samples_list, download_dir):
  """Downloads and applies background subtraction to a list of files with format 'date\ttime\tftype\tinstruments'
  """
  exist = []
  samples = []
  for elemen in samples_list:
    date, time, ftype, instruments = elemen
    str_start, str_end = time.split('-')
    instruments = instruments.replace(' ','').split(',')

    start_time, end_time = datetime.datetime.strptime(date + str_start, '%Y%m%d%H:%M'), datetime.datetime.strptime(date + str_end, '%Y%m%d%H:%M')
    urls = list(query(start_time, end_time, instruments=instruments))

    for lurl in urls:
      local_name = os.path.basename(lurl)
      if local_name not in exist:
        local_item = subtract_downloader(lurl, download_dir)
        exist.append(local_item)
      else:
        print('-----------------repeated ' + local_name)
        local_item = local_name
      print(date, str_start, str_end, ftype, download_dir+local_item)
      samples.append((date, str_start, str_end, ftype, download_dir+local_item))

  return samples


def download_month(instrument, year, month, download_dir, start_time = '05:30:00', end_time = '18:00:00', ):
  """Given a month and year, downloads all the available data for an Instrument of the callisto network
  """
  str_start = year + '-' + month + '-1 ' + start_time
  str_end = year + '-' + month + '-1 ' + end_time

  start_time, end_time = parse_time(str_start).datetime, parse_time(str_end).datetime
  month_len = calendar.monthlen(start_time.year, start_time.month)

  for elemen in range(month_len):

    urls_list = list(query(start_time, end_time, [instrument]))
    _ = [subtract_downloader(local_url).writeto(download_dir + os.path.basename(local_url)) for local_url in urls_list]
    print(start_time, end_time, len(urls_list))

    start_time = start_time + datetime.timedelta(days=1)
    end_time = end_time + datetime.timedelta(days=1)


def create_inference_samples(files_dir, window_length):
  """Creates samples of size window_length on the x-axis for inference
  """
  dir_list = [files_dir + elemen for elemen in os.listdir(files_dir)]
  big_list = []

  for elemen in dir_list:
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


def window_divide(file_dir, item_start, item_end, window_length, temp_root, temp_list=[]):
  """Divides a fits file by window_length based on a time window
  """
  item = file_dir
  local_list = []
  fits_file = fits.open(item)
  f_start, f_end = datetime.datetime.strptime(item_start, '%H:%M'), datetime.datetime.strptime(item_end, '%H:%M')
  str_obs, str_end = fits_file[0].header['TIME-OBS'], fits_file[0].header['TIME-END']

  if str_end == '23:59:60':
    raise Exception('cannot recognize TIME-END = 23:59:60')

  if str_end[-2:] == '60':
    str_end = (datetime.datetime.strptime(str_end[:5], '%H:%M') + datetime.timedelta(minutes=1)).strftime('%H:%M:%S')
  try:
    obs, end = datetime.datetime.strptime(str_obs[:8], '%H:%M:%S'), datetime.datetime.strptime(str_end, '%H:%M:%S')
  except ValueError:
    str_end = str_end.replace('24', '23')
    obs, end = datetime.datetime.strptime(str_obs[:8], '%H:%M:%S'), datetime.datetime.strptime(str_end, '%H:%M:%S') + datetime.timedelta(hours=1)

  if f_start == f_end:
    #sets f_end to the end of the minute
    f_end += datetime.timedelta(seconds=59)
    
  #checking patching shapes
  if fits_file[0].data.shape[1] > 3600 and fits_file[0].data.shape[1] < 7200: 
    target_shape = math.ceil(fits_file[0].data.shape[1]/window_length)*window_length
    new_data = mask_zeros(fits_file[0].data, target_shape)
    fits_file[0].data = new_data
    temp_file = temp_root + os.path.basename(item)
    if temp_file not in temp_list:
      fits_file.writeto(temp_file)
      temp_list.append(temp_file)
    item = temp_file

  if fits_file[0].data.shape[1] < 3600: 
    target_shape = math.ceil(fits_file[0].data.shape[1]/window_length)*window_length
    new_data = mask_zeros(fits_file[0].data, target_shape) 
    fits_file[0].data = new_data
    temp_file = temp_root + os.path.basename(item)
    if temp_file not in temp_list:
      fits_file.writeto(temp_file)
      temp_list.append(temp_file)
    item = temp_file

  
  #Case 1 window_length divide
  if obs >= f_start and end <= f_end:
    obj_num = fits_file[0].data.shape[1]/window_length
    
    if not obj_num.is_integer():
      raise Exception('Number of pieces is not an Integer: ' + str(obj_num))

    start_here = 0
    for elemen in range(int(obj_num)):
      local_list.append((item, start_here, start_here + window_length, 1))
      print((item, start_here, start_here + window_length, 1))
      start_here += window_length

  #Case 2
  if obs > f_start and end > f_end:
    t_delt = get_delt(fits_file[0].header)
    pxl_start, pxl_end = 0, int((f_end-obs).seconds/t_delt)
    local_shape = pxl_end - pxl_start
    obj_num = local_shape/window_length

    if not obj_num.is_integer():
      obj_num = math.ceil(obj_num)
    else:
      obj_num = int(obj_num)

    available_pieces = fits_file[0].shape[1]/window_length
    if obj_num>available_pieces:
      raise Exception('Not enough available pieces for window_length: ' + str(available_pieces) + ' Needed: ' + str(obj_num))

    start_here = 0
    for elemen in range(obj_num):
      end_here = start_here + window_length
      local_list.append((item, start_here, end_here, 1))
      print((item, start_here, end_here, 1))
      start_here = end_here
    
    #negative calculations
    non_start = end_here
    for _ in range(math.floor((fits_file[0].data.shape[1] - non_start)/window_length)):
      local_list.append((item, non_start, non_start+window_length, 0))
      print((item, non_start, non_start+window_length, 0))
      non_start+=window_length

  #Case 3
  if obs < f_start and end < f_end:
    t_delt = get_delt(fits_file[0].header)
    pxl_start, pxl_end = int((f_start-obs).seconds/t_delt), int((end-obs).seconds/t_delt)
    local_shape = pxl_end - pxl_start
    obj_num = local_shape/window_length

    if not obj_num.is_integer():
      obj_num = math.ceil(obj_num)
    else:
      obj_num = int(obj_num)
    
    available_pieces = fits_file[0].shape[1]/window_length
    if obj_num>available_pieces:
      raise Exception('Not enough available pieces for window_length: ' + str(available_pieces) + ' Needed: ' + str(obj_num))

    excess = (obj_num*window_length) - local_shape
    pxl_start = pxl_start-excess

    #we use the zero paddings only if necessesary
    #overall solution would be to set pxl_end = fits_file[0].data.shape[1] from the beggining
    #so  pxl_start, pxl_end = int((f_start-obs).seconds/t_delt), fits_file[0].shape[1]
    if pxl_start<0:
      pxl_start=0

    start_here = pxl_start
    for elemen in range(obj_num):
      end_here = start_here + window_length
      local_list.append((item, start_here, end_here, 1))
      print((item, start_here, end_here, 1))
      start_here = end_here

    #negative calculations
    non_start = 0
    for _ in range(math.floor(pxl_start/window_length)):
      local_list.append((item, non_start, non_start+window_length, 0))
      print((item, non_start, non_start+window_length, 0))
      non_start+=window_length

  #Case 4
  if obs < f_start and end > f_end:
    t_delt = get_delt(fits_file[0].header)
    pxl_start, pxl_end = int((f_start - obs).seconds/t_delt), int((f_end - obs).seconds/t_delt)
    local_shape = pxl_end - pxl_start
    obj_num = math.ceil(local_shape/window_length)
    
    available_pieces = fits_file[0].shape[1]/window_length
    if obj_num>available_pieces:
      raise Exception('Not enough available pieces for window_length: ' + str(available_pieces) + ' Needed: ' + str(obj_num))

    excess = (obj_num*window_length)-local_shape
    
    if pxl_end + excess > fits_file[0].data.shape[1]:
      excess = (pxl_end + excess) - fits_file[0].data.shape[1]
      pxl_start = pxl_start - excess
      pxl_end = fits_file[0].data.shape[1]
    
    start_here = pxl_start
    for elemen in range(obj_num):
      end_here = start_here + window_length
      local_list.append((item, start_here, end_here, 1))
      print((item, start_here, end_here, 1))
      start_here = end_here
    
    #negative calculations
    left_start = 0
    right_start = end_here

    left_range = math.floor(pxl_start/window_length)
    for _ in range(left_range):
      local_list.append((item, left_start, left_start+window_length, 0))
      print((item, left_start, left_start+window_length, 0))
      left_start+=window_length

    right_range = math.floor((fits_file[0].data.shape[1] - right_start)/window_length)
    for _ in range(right_range):
      local_list.append((item, right_start, right_start+window_length, 0))
      print((item, right_start, right_start+window_length, 0))
      right_start+=window_length
  
  return local_list, temp_list

#samples creator
def samples_test_creator(samples_list, window_length, temp_root):
  """Given a list of samples, calculates mini-samples for testing
  """
  temp_list = []
  items_list = []
  for elemen in samples_list:
    local_list, temp_list = window_divide(elemen[4], elemen[1], elemen[2], window_length, temp_root, temp_list)
    items_list+=local_list
  return items_list