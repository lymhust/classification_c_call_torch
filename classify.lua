function classify(img, batch, box, cls)
    print('Lua classify function')

    sys.tic()
    
    local im = img:permute(3, 1, 2)
    for i = 1, batch:size(1) do
        batch[i] = image.scale(im, 48, 48)
    end
    
    -- Normalize
    batch:add(-mean)
    batch:div(std)
    batch = norm:forward(batch)
    
    -- Forward
    cuBatch:copy(batch)
    local scores = model:forward(cuBatch)
    local _, preds = scores:max(2)
    cls = preds:float()
    
    print('Time: '..(sys.toc()*1000)..'ms')
    
    return cls
end

--classify(torch.Tensor(256,256,3),torch.Tensor(20,3,48,48),torch.Tensor(20,4),torch.Tensor(20))
















