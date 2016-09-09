require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
require 'cudnn'
require 'image'
require 'sys'
torch.setdefaulttensortype('torch.FloatTensor')

-------------------------------------------------------------------
-- Init
boxNum = 20
cuBatch = torch.CudaTensor(boxNum,3,48,48)
labtxt = torch.load('./src/luafile/labeltxt.t7')
local mean_std = torch.load('./src/luafile/norm_lpvs_c49.t7')
mean, std = mean_std[1], mean_std[2]

model = torch.load('./src/luafile/cnn_e10_lpvs_c49_deploy.t7')
print('Torch init finished')
-------------------------------------------------------------------

