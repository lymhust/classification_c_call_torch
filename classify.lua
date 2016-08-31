require 'torch'
require 'cutorch'
require 'image'
require 'sys'
torch.setdefaulttensortype('torch.FloatTensor')
cuBatch = torch.CudaTensor()

function classify(img, batch, box, cls)
    print('Lua classify function')
    --[[
    print(#img)
    print(#batch)
    print(#box)
    print(#cls)
    --]]
    sys.tic()
    
    local im = img:permute(3, 1, 2)
    for i = 1, batch:size(1) do
        batch[i] = image.scale(im, 48, 48)
    end
    cuBatch = batch:cuda()
    --image.save('./test.jpg', batch[1])
    
    print('Time: '..(sys.toc()*1000)..'ms')
    return cls
end
