require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
require 'stn'
require 'image'
require 'sys'
torch.setdefaulttensortype('torch.FloatTensor')

-------------------------------------------------------------------
-- Init
cuBatch = torch.zeros(20,3,48,48,'torch.CudaTensor')
local mean_std = torch.load('./src/luafile/norm.t7')
mean, std = mean_std[1], mean_std[2]

local norm_kernel = image.gaussian1D(7)
norm = nn.SpatialContrastiveNormalization(3,norm_kernel)

model = torch.load('./src/luafile/stn_test.t7'):cuda()
print('Torch init finished')
-------------------------------------------------------------------

