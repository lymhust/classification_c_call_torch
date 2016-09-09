function classify(img, box, cls)
    print('Lua classify function')

    sys.tic()
    cuBatch:zero()
    
    local im = img:permute(3, 1, 2)
    local ind = 0
    
    for i = 1, boxNum do
        if(box[i][1] ~= -1) then
            local left,top,w,h = box[i][1],box[i][2],box[i][3],box[i][4]
            local right = left+w-1
            local bottom = top+h-1
            cuBatch[i] = image.scale(im[{ {},{top,bottom},{left,right} }], 48, 48)    
            ind = ind + 1
        else
            break
        end
    end
    
    if(ind > 0 and ind <= boxNum) then
        -- Normalize
        cuBatch:add(-mean)
        cuBatch:div(std)
    
        -- Forward
        local scores = model:forward(cuBatch[{ {1,ind},{},{},{} }])
        local _, preds = scores:max(2)

        -- Get the result
        cls:copy(preds:int()) -- Copy data and the pointer is not changed
    end
    print('Time: '..(sys.toc()*1000)..'ms')
    
end

--classify(torch.Tensor(256,256,3),torch.IntTensor(20,4),torch.IntTensor(20))
















