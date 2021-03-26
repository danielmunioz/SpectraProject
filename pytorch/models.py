import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock5x5(nn.Module):
  def __init__(self, in_ch, out_ch, pool_size=2, pool_type='avg'):
    super().__init__()
    self.conv_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                            kernel_size=5, stride=1, padding=2,
                            bias=False)
    self.bn_1 = nn.BatchNorm2d(out_ch)

    self.pool_size = pool_size
    self.pool_type = pool_type

    self.init_layers()

  def forward(self, x):
    x = F.relu(self.bn_1(self.conv_1(x)))
    if self.pool_type == 'avg':
      x = F.avg_pool2d(x, kernel_size=self.pool_size)
    elif self.pool_type == 'max':
      x = F.max_pool2d(x, kernel_size=self.pool_size)
    else:
      raise Exception('need either avg or max pool_type')
    return x

  def init_layers(self):
    nn.init.xavier_uniform_(self.conv_1.weight)


class ConvBlock(nn.Module):
  def __init__(self, in_ch, out_ch, pool_size=2, pool_type='avg'):
    super().__init__()
    self.conv_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                            kernel_size=3, stride=1, padding=1,
                            bias=False)
    self.conv_2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                            kernel_size=3, stride=1, padding=1,
                            bias=False)
    self.bn_1 = nn.BatchNorm2d(out_ch)
    self.bn_2 = nn.BatchNorm2d(out_ch)

    self.pool_size = pool_size
    self.pool_type = pool_type

    self.init_layers()

  def forward(self, x):
    x = F.relu(self.bn_1(self.conv_1(x)))
    x = F.relu(self.bn_2(self.conv_2(x)))
    if self.pool_type == 'avg':
      x = F.avg_pool2d(x, kernel_size=self.pool_size)
    elif self.pool_type == 'max':
      x = F.max_pool2d(x, kernel_size=self.pool_size)
    else:
      raise Exception('need either avg or max pool_type')
    return x

  def init_layers(self):
    nn.init.xavier_uniform_(self.conv_1.weight)
    nn.init.xavier_uniform_(self.conv_2.weight)


class Base_CNN4Smaller16x32(nn.Module):
    def __init__(self, classes_num):
        super().__init__()
        self.conv_block_1 = ConvBlock5x5(1, 16, pool_size=2, pool_type='avg')
        self.drop_1 = nn.Dropout(0.2)
        self.conv_block_2 = ConvBlock5x5(16, 32, pool_size=2, pool_type='avg')
        self.drop_2 = nn.Dropout(0.2)

        self.drop_global = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32, 32, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.fc_out = nn.Linear(32, classes_num, bias=True)
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.drop_1(x)
        x = self.conv_block_2(x)
        x = self.drop_2(x)
        
        #global pooling
        x = torch.mean(x, dim=3)
        x1,_ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1+x2
        
        x = self.drop_global(x)
        x = F.relu(self.fc1(x))

        x = self.drop_out(x)
        output = self.fc_out(x)
        
        return output
    
    def init_layers(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        

class Base_CNN4Smaller(nn.Module):
    def __init__(self, classes_num):
        super().__init__()
        self.conv_block_1 = ConvBlock5x5(1, 32, pool_size=2, pool_type='avg')
        self.drop_1 = nn.Dropout(0.2)
        self.conv_block_2 = ConvBlock5x5(32, 64, pool_size=2, pool_type='avg')
        self.drop_2 = nn.Dropout(0.2)

        self.drop_global = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 64, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, classes_num, bias=True)
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.drop_1(x)
        x = self.conv_block_2(x)
        x = self.drop_2(x)
        
        x = torch.mean(x, dim=3)
        x1,_ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1+x2
        
        x = self.drop_global(x)
        x = F.relu(self.fc1(x))

        x = self.drop_out(x)
        output = self.fc_out(x)
        
        return output
    
    def init_layers(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)


class Base_CNN4(nn.Module):
    def __init__(self, classes_num):
        super().__init__()
        self.conv_block_1 = ConvBlock5x5(1, 64, pool_size=2, pool_type='avg')
        self.drop_1 = nn.Dropout(0.2)
        self.conv_block_2 = ConvBlock5x5(64, 128, pool_size=2, pool_type='avg')
        self.drop_2 = nn.Dropout(0.2)

        self.drop_global = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 128, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, classes_num, bias=True)
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.drop_1(x)
        x = self.conv_block_2(x)
        x = self.drop_2(x)
        
        x = torch.mean(x, dim=3)
        x1,_ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1+x2
        
        x = self.drop_global(x)
        x = F.relu(self.fc1(x))

        x = self.drop_out(x)
        output = self.fc_out(x)
        
        return output
    
    def init_layers(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)


class Base_CNN5(nn.Module):
    def __init__(self, classes_num):
        super().__init__()
        self.conv_block_1 = ConvBlock5x5(1, 64, pool_size=2, pool_type='avg')
        self.drop_1 = nn.Dropout(0.2)
        self.conv_block_2 = ConvBlock5x5(64, 128, pool_size=2, pool_type='avg')
        self.drop_2 = nn.Dropout(0.2)
        self.conv_block_3 = ConvBlock5x5(128, 256, pool_size=2, pool_type='avg')
        self.drop_3 = nn.Dropout(0.2)

        self.drop_global = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 256, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, classes_num, bias=True)
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.drop_1(x)
        x = self.conv_block_2(x)
        x = self.drop_2(x)
        x = self.conv_block_3(x)
        x = self.drop_3(x)
        
        x = torch.mean(x, dim=3)
        x1,_ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1+x2
        
        x = self.drop_global(x)
        x = F.relu(self.fc1(x))

        x = self.drop_out(x)
        output = self.fc_out(x)
        
        return output
    
    def init_layers(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)


class Base_CNN6(nn.Module):
    def __init__(self, classes_num):
        super().__init__()
        self.conv_block_1 = ConvBlock5x5(1, 64, pool_size=2, pool_type='avg')
        self.drop_1 = nn.Dropout(0.2)
        self.conv_block_2 = ConvBlock5x5(64, 128, pool_size=2, pool_type='avg')
        self.drop_2 = nn.Dropout(0.2)
        self.conv_block_3 = ConvBlock5x5(128, 256, pool_size=2, pool_type='avg')
        self.drop_3 = nn.Dropout(0.2)
        self.conv_block_4 = ConvBlock5x5(256, 512, pool_size=2, pool_type='avg')
        self.drop_4 = nn.Dropout(0.2)

        self.drop_global = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.fc_out = nn.Linear(512, classes_num, bias=True)

        self.init_layers()

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.drop_1(x)
        x = self.conv_block_2(x)
        x = self.drop_2(x)
        x = self.conv_block_3(x)
        x = self.drop_3(x)
        x = self.conv_block_4(x)
        x = self.drop_4(x)
        
        x = torch.mean(x, dim=3)
        x1,_ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1+x2

        x = self.drop_global(x)
        x = F.relu(self.fc1(x))

        x = self.drop_out(x)
        output = self.fc_out(x)

        return output

    def init_layers(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)


class Base_CNN10(nn.Module):
    def __init__(self, classes_num):
        super().__init__()
        self.conv_block_1 = ConvBlock(1, 64, pool_size=2, pool_type='avg')
        self.drop_1 = nn.Dropout(0.2)
        self.conv_block_2 = ConvBlock(64, 128, pool_size=2, pool_type='avg')
        self.drop_2 = nn.Dropout(0.2)
        self.conv_block_3 = ConvBlock(128, 256, pool_size=2, pool_type='avg')
        self.drop_3 = nn.Dropout(0.2)
        self.conv_block_4 = ConvBlock(256, 512, pool_size=2, pool_type='avg')
        self.drop_4 = nn.Dropout(0.2)

        self.drop_global = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.fc_out = nn.Linear(512, classes_num, bias=True)

        self.init_layers()

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.drop_1(x)
        x = self.conv_block_2(x)
        x = self.drop_2(x)
        x = self.conv_block_3(x)
        x = self.drop_3(x)
        x = self.conv_block_4(x)
        x = self.drop_4(x)
        
        x = torch.mean(x, dim=3)
        x1,_ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1+x2

        x = self.drop_global(x)
        x = F.relu(self.fc1(x))

        x = self.drop_out(x)
        output = self.fc_out(x)

        return output

    def init_layers(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)


class Base_CNN14(nn.Module):
    def __init__(self, classes_num):
        super().__init__()
        self.conv_block_1 = ConvBlock(1, 64, pool_size=2, pool_type='avg')
        self.drop_1 = nn.Dropout(0.2)
        self.conv_block_2 = ConvBlock(64, 128, pool_size=2, pool_type='avg')
        self.drop_2 = nn.Dropout(0.2)
        self.conv_block_3 = ConvBlock(128, 256, pool_size=2, pool_type='avg')
        self.drop_3 = nn.Dropout(0.2)
        self.conv_block_4 = ConvBlock(256, 512, pool_size=2, pool_type='avg')
        self.drop_4 = nn.Dropout(0.2)
        self.conv_block_5 = ConvBlock(512, 1024, pool_size=2, pool_type='avg')
        self.drop_5 = nn.Dropout(0.2)
        self.conv_block_6 = ConvBlock(1024, 2048, pool_size=1, pool_type='avg')
        self.drop_6 = nn.Dropout(0.2)

        self.drop_global = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.fc_out = nn.Linear(2048, classes_num, bias=True)

        self.init_layers()

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.drop_1(x)
        x = self.conv_block_2(x)
        x = self.drop_2(x)
        x = self.conv_block_3(x)
        x = self.drop_3(x)
        x = self.conv_block_4(x)
        x = self.drop_4(x)
        x = self.conv_block_5(x)
        x = self.drop_5(x)
        x = self.conv_block_6(x)
        x = self.drop_6(x)
        
        x = torch.mean(x, dim=3)
        x1,_ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1+x2
        
        x = self.drop_global(x)
        x = F.relu(self.fc1(x))

        x = self.drop_out(x)
        output = self.fc_out(x)

        return output

    def init_layers(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)