function train(model, criterion, inputs, outputs, testinputs, testoutputs, optimFunc, optimState, opts)
  local params, gradParams = model:getParameters()
  local loss = 0
  local trainIndex = 0
  local epoch = 0
  local iter = 0
  local restartCount = 0
  local decreaseCount = 0
  local decreaseTimes = 0
  local batchSize = opts.batchSize
  local cropSize = opts.cropSize
  local maxIterations
  local maxEpochs
  local gradClipping
  local testloss
  local initialLearningRate = optimState.learningRate
  
  local inCropSize = cropSize
  if settings.pyramidmodel then
    inCropSize = cropSize/2
  end

  if opts.gradClipping and initialLearningRate then
    gradClipping = opts.gradClipping * initialLearningRate
  end
  if opts.maxIterations and opts.maxIterations > 0 then
    maxIterations = opts.maxIterations
  end
  if opts.maxEpochs and opts.maxEpochs > 0 then
    maxEpochs = opts.maxEpochs
  end

  -- allocate minibatch
  local input = tools.prepareTensorBackend(torch.Tensor(batchSize, opts.batchChannels, inCropSize, inCropSize))
  local output = tools.prepareTensorBackend(torch.Tensor(batchSize, opts.batchChannels, cropSize, cropSize))
  local trainSize = #inputs
  local shuffle = torch.randperm(trainSize)
  local timer = torch.Timer()
  local showTimer = torch.Timer()

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
    for i=1,batchSize do
      local idx = shuffle[trainIndex+i]
      local inimg, outimg = inputs[idx], outputs[idx]
      local width, height = inimg:size(2), inimg:size(3)
      --TODO: check if image is smaller than crop size
      local x1, y1 = math.random(1,width-inCropSize+1), math.random(1,height-inCropSize+1)
      local x2, y2 = x1+inCropSize-1, y1+inCropSize-1
      local incrop = {{},{x1,x2},{y1,y2}}
      local outcrop
      if settings.pyramidmodel then
        local ox1, oy1 = x1*2-1, y1*2-1
        local ox2, oy2 = x2*2, y2*2
        outcrop = {{},{ox1,ox2},{oy1,oy2}}
      else
        outcrop = {{},{x1,x2},{y1,y2}}
      end
      input[i] = tools.prepareTensorBackend(inimg[incrop])
      output[i] = tools.prepareTensorBackend(outimg[outcrop])
    end
  end

  if optimState.learningRate then
    utils.printf("Train will begin with learning rate %.16f [1/%d]\n",
                optimState.learningRate,
                opts.learnRestartRate)
  end

  while (not maxIterations or iter<maxIterations) and (not maxEpochs or epoch<maxEpochs) do
    local debugPrint = false
    if iter % opts.printIterations == 0 then
      debugPrint = true
    end
    loadMiniBatch()
    if debugPrint then
      local predicted = model:forward(testinputs)
      testloss = criterion:forward(predicted, testoutputs)
      if settings.visdom and (showTimer:time().real > settings.showinterval or iter == 0) then
        visual.showImageResults('testset', testinputs, testoutputs, predicted)
        showTimer:reset()
      end
    end
    model:training()
    optimFunc(feval, params, optimState)
    model:evaluate()

    trainIndex = trainIndex + batchSize
    if trainIndex + batchSize > trainSize then
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
      utils.printf('epoch=%06d iter=%06d loss=%.16f testloss=%.16f testpsnr=%.8f time=%.3fs \n',
                   epoch, iter, loss, testloss, tools.mse2psnr(testloss), timer:time().real)
    end
  end

  local predicted = model:forward(testinputs)
  testloss = criterion:forward(predicted, testoutputs)
  if settings.visdom then
    visual.showImageResults('testset', testinputs, testoutputs, predicted)
  end
  utils.printf('train ended! epoch=%06d, iter=%06d, testloss=%.16f testpsnr=%.4f time=%.3fs\n',
               epoch, iter, testloss, tools.mse2psnr(testloss), timer:time().real)

end

function saveModel(path, model)
  model:clearState()
  torch.save(path, model)
  print('Model saved!')
end

if settings.save then
  if settings.resume == '' then
    tools.fatalerror('Your need to supply a valid model file to save to!')
  end
  settings.saveto = settings.resume
end

--settings.pyramidmodel = true

local Y = tools.loadImages(settings.trainset, settings.trainsize)
local tY = tools.loadImages(settings.testset, settings.testsize)
local X, tX
if settings.pyramidmodel then
  X = tools.downscaleImages(Y, settings.scalefactor)
  tX = tools.downscaleImages(tY, settings.scalefactor)
else
  X = tools.rescaleImages(Y, settings.scalefactor)
  tX = tools.rescaleImages(tY, settings.scalefactor)
end

local model
if settings.newmodel ~= '' then
  model = require('./models/' .. settings.newmodel)
  if settings.saveto == '' then
    print('New model '.. settings.newmodel .. ' WITHOUT saving!')
  elseif path.exists(settings.saveto) then
    print('New trained model ' .. settings.newmodel .. ' will OVERWRITE ' .. settings.saveto .. '!')
  else
    print('New trained model ' .. settings.newmodel .. ' will be written to ' .. settings.saveto .. '!')
  end
elseif settings.resume ~= '' and path.exists(settings.resume) then
  model = torch.load(settings.resume)
  print('Loaded model ' .. settings.resume)
else
  tools.fatalerror('Your need to supply a valid -newmodel or -resume to train on!')
end
model:evaluate()

local criterion = nn.MSECriterion()
--criterion.sizeAverage = false

tools.prepareModelBackend(model, criterion)
tY = tools.prepareTensorBackend(tools.tabletotensor(tY))
tX = tools.prepareTensorBackend(tools.tabletotensor(tX))

-- auto save on quit
if settings.saveto ~= '' then
  local function term(signum)
    saveModel(settings.saveto, model:float())
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR2, signal.SIG_DFL)
    os.exit(0)
  end
  signal.signal(signal.SIGINT, term)
  signal.signal(signal.SIGTERM, term)
  signal.signal(signal.SIGUSR2, term)
end

print('Training with ' .. settings.optim .. ' optimizer')

local trainOpts = {
  batchSize = settings.batchsize,
  cropSize = settings.cropsize,
  batchChannels = settings.channels,
  maxIterations = settings.maxiterations,
  maxEpochs = settings.maxepochs,
  printIterations = settings.printiterations,
  learnDecreaseRate = 10, -- in epochs
  learnRestartRate = 8,
  learnRestartRepeats = 2,
  learnDecreaseFactor = 0.5,
  gradClipping = 0.01,
}

-- free any loading unused memory
collectgarbage()

if settings.optim == 'adam' then
  train(model, criterion, X, Y, tX, tY, optim.adam, {
    learningRate = 0.0001,
    weightDecay = 0.00001,
  }, trainOpts)
elseif settings.optim == 'adamax' then
  train(model, criterion, X, Y, tX, tY, optim.adamax, {
    learningRate = 0.0001,
    weightDecay = 0.00001,
  }, trainOpts)
elseif settings.optim == 'nag' then
  train(model, criterion, X, Y, tX, tY, optim.nag, {
    learningRate = 0.1,
    weightDecay = 0.00001,
  }, trainOpts)
elseif settings.optim == 'sgd' then
  train(model, criterion, X, Y, tX, tY, optim.sgd, {
    learningRate = 0.01,
    weightDecay = 0.00001,
    momentum = 0.9
  }, trainOpts)
end
-- what about Nadam

if settings.saveto ~= '' then
  saveModel(settings.saveto, model)
end