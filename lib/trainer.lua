local trainer = {}

function trainer.train(model, criterion, inputs, outputs, testinputs, testoutputs, backend, opts)
  local loss = 0
  local trainIndex = 0
  local epoch = 0
  local iter = 0
  local restartCount = 0
  local decreaseCount = 0
  local decreaseTimes = 0
  local maxIterations
  local maxEpochs
  local gradClipping
  local testloss
  local testpsnr
  local inCropSize = opts.inputCropSize
  local cropSize = opts.outputCropSize or inCropSize
  local optimFunc = opts.optim
  local optimState = opts.optimState
  local initialLearningRate = optimState.learningRate

  if opts.gradClipping and initialLearningRate then
    gradClipping = opts.gradClipping * initialLearningRate
  end
  if opts.maxIterations and opts.maxIterations > 0 then
    maxIterations = opts.maxIterations
  end
  if opts.maxEpochs and opts.maxEpochs > 0 then
    maxEpochs = opts.maxEpochs
  end

  local nchannels = inputs[1]:size(1)

  tools.prepareModelBackend(backend, model, criterion)
  testoutputs = tools.prepareTensorBackend(backend, tools.tabletotensor(testoutputs))
  testinputs = tools.prepareTensorBackend(backend, tools.tabletotensor(testinputs))

  -- allocate minibatch
  local input = tools.prepareTensorBackend(backend, torch.Tensor(opts.batchSize, nchannels, inCropSize, inCropSize))
  local output = tools.prepareTensorBackend(backend, torch.Tensor(opts.batchSize, nchannels, cropSize, cropSize))
  local trainSize = #inputs
  local shuffle = torch.randperm(trainSize)
  local timer = torch.Timer()
  local showTimer = torch.Timer()

  if opts.batchSize > trainSize then
    error('batch size is greater than train set size')
  end

  local params, gradParams = model:getParameters()

  local function feval()
    -- zero grandients
    gradParams:zero()

    -- forward and backward pass
    local predicted = model:forward(input)
    loss = criterion:forward(predicted, output)
    model:backward(input, criterion:backward(predicted, output))

    -- gradient clipping
    if gradClipping then
      local clip = gradClipping/optimState.learningRate
      gradParams:clamp(-clip, clip)
    end

    return loss, gradParams
  end

  local function loadMiniBatch()
    for i=1,opts.batchSize do
      local idx = shuffle[trainIndex+i]
      local inimg, outimg = inputs[idx], outputs[idx]
      local width, height = inimg:size(2), inimg:size(3)
      --TODO: check if image is smaller than crop size
      local x1, y1 = math.random(1,width-inCropSize+1), math.random(1,height-inCropSize+1)
      local x2, y2 = x1+inCropSize-1, y1+inCropSize-1
      local incrop = {{},{x1,x2},{y1,y2}}
      local ratio = cropSize/inCropSize
      local outcrop = incrop
      if ratio > 1 then
        local ox1, oy1 = x1*ratio-1, y1*ratio-1
        local ox2, oy2 = x2*ratio, y2*ratio
        outcrop = {{},{ox1,ox2},{oy1,oy2}}
      end
      input[i] = tools.prepareTensorBackend(backend, inimg[incrop])
      output[i] = tools.prepareTensorBackend(backend, outimg[outcrop])
    end
  end

  if optimState.learningRate and opts.learnRestartRate then
    utils.printf("Train will begin with learning rate %.16f [1/%d]\n",
                optimState.learningRate,
                opts.learnRestartRate)
  end

  -- free any loading unused memory
  collectgarbage()

  while (not maxIterations or iter<maxIterations) and (not maxEpochs or epoch<maxEpochs) do
    local debugPrint = false
    if iter % opts.printIterations == 0 then
      debugPrint = true
    end
    loadMiniBatch()
    if debugPrint then
      local predicted = model:forward(testinputs)
      testpsnr = tools.psnr(predicted, testoutputs)
      testloss = criterion:forward(predicted, testoutputs)
      if opts.visdom and (showTimer:time().real > opts.showInterval or iter == 0) then
        visual.showImageResults('testset', testinputs, testoutputs, predicted)
        showTimer:reset()
      end
    end
    model:training()
    optimFunc(feval, params, optimState)
    model:evaluate()

    trainIndex = trainIndex + opts.batchSize
    if trainIndex + opts.batchSize > trainSize then
      shuffle = torch.randperm(trainSize)
      trainIndex = 0
      epoch = epoch + 1

      if optimState.learningRate then
        if opts.learnDecreaseRate and (epoch % opts.learnDecreaseRate == 0) then
          local restarted = false
          decreaseTimes = decreaseTimes + 1
          decreaseCount = decreaseCount + 1
          if opts.learnRestartRate and decreaseCount % opts.learnRestartRate == 0 then
            decreaseTimes = math.floor(restartCount/opts.learnRestartRepeats)
            restartCount = restartCount + 1
            restarted = true
          else
            optimState.learningRate = initialLearningRate * math.pow(opts.learnDecreaseFactor, decreaseTimes)
          end

          optimState.learningRate = initialLearningRate * math.pow(opts.learnDecreaseFactor, decreaseTimes)

          if restarted then
            utils.printf("Learn rate restarted to %.16f [1/%d] (%d)\n",
                        optimState.learningRate,
                        opts.learnRestartRate,
                        restartCount)
          else
            utils.printf("Learn rate decrease to %.16f [%d/%d]\n",
                        optimState.learningRate,
                        (decreaseCount % opts.learnRestartRate)+1,
                        opts.learnRestartRate)
          end
        end
      end
    end
    iter = iter + 1

    if debugPrint then
      local lr = optimState.lr or optimState.learningRate
      utils.printf('epoch=%06d iter=%06d loss=%.16f testloss=%.16f testpsnr=%.8f lr=%.8f time=%.3fs \n',
                   epoch, iter, loss, testloss, testpsnr, lr, timer:time().real)
    end
  end

  local predicted = model:forward(testinputs)
  testloss = criterion:forward(predicted, testoutputs)
  if opts.visdom then
    visual.showImageResults('testset', testinputs, testoutputs, predicted)
  end
  utils.printf('train ended! epoch=%06d, iter=%06d, testloss=%.16f testpsnr=%.4f time=%.3fs\n',
               epoch, iter, testloss, tools.mse2psnr(testloss), timer:time().real)

end

return trainer