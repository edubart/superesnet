require '../lib/global'

local settings = {
  trainSet = 'data/sprites',
  trainSize = nil,
  testSet = 'data/sprites_test',
  testSize = nil,
  colorType = 'rgb',
  model = 'vdsr',
  modelFile = 'models/vdsr-2x-16l.t7',
  modelOpts = {
    channels = 3,
    featureMaps = 64,
    depth = 16,
    filterSize = 3,
    batchNormalization = true,
    nonLinear = function() return nn.LeakyReLU(0.2, true) end
  },
  --autosave = true,
  scaleFactor = 2,

  backend = 'cuda',
  backendOpts = {
    seed = 0,
  },

  trainOpts = {
    optim = optim.adamhd,
    optimState = {
      learningRate = 0.0001,
      weightDecay = 0.00001,
    },

    learnDecreaseRate = 10, -- in epochs
    learnRestartRate = 8, -- in decreases times
    learnRestartRepeats = 2,
    learnDecreaseFactor = 0.5,

    gradClipping = 0.01,
    
    batchSize = 8,
    inputCropSize = 64,
    outputCropSize = 64,
    maxEpochs = 0,
    maxIterations = 0,
    printIterations = 1,
    visualizeTest = true,
    visdom = true,
    showInterval = 5,
  }
}

tools.prepareBackend(settings.backend, settings.backendOpts)
local model = tools.loadModel(settings.model, settings.modelOpts, settings.modelFile)
local outputs = tools.loadImages(settings.trainSet, settings.trainSize, settings.colorType)
local testOutputs = tools.loadImages(settings.testSet, settings.testSize, settings.colorType)
local inputs = tools.rescaleImages(outputs, settings.scaleFactor)
local testInputs = tools.rescaleImages(testOutputs, settings.scaleFactor)
local criterion = nn.MSECriterion()
tools.setupAutoSave(settings.modelFile, model, settings.autosave)
trainer.train(model, criterion, inputs, outputs, testInputs, testOutputs, settings.backend, settings.trainOpts)
tools.saveModel(settings.modelFile, model)
