require '../lib/global'

local settings = {
  trainSet = {'data/sprites', 'data/avatars'},
  trainSize = nil,
  testSet = 'data/sprites_test',
  testSize = nil,
  colorType = 'rgb',
  model = 'lapsr',
  modelFile = 'models/lapsr-2x-8l.t7',
  modelOpts = {
    channels = 3,
    featureMaps = 64,
    depth = 8,
    filterSize = 3
  },
  autosave = true,
  scaleFactor = 2,

  backend = 'cuda',
  backendOpts = {
    seed = 0,
  },

  trainOpts = {
    optim = optim.adamhd,
    optimState = {
      learningRate = 0.0001,
      learningLearningRate = 1e-6,
      --weightDecay = 0.0001,
    },

    gradClipping = 0.01,
    
    batchSize = 16,
    inputCropSize = 16,
    outputCropSize = 32,
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
local outputs = tools.loadImages(settings.trainSet, settings.trainSize, settings.colorType, true)
local testOutputs = tools.loadImages(settings.testSet, settings.testSize, settings.colorType)
local inputs = tools.downscaleImages(outputs, settings.scaleFactor)
local testInputs = tools.downscaleImages(testOutputs, settings.scaleFactor)
local criterion = nn.CharbonnierCriterion(1e-3)
tools.setupAutoSave(settings.modelFile, model, settings.autosave)
trainer.train(model, criterion, inputs, outputs, testInputs, testOutputs, settings.backend, settings.trainOpts)
tools.saveModel(settings.modelFile, model)
