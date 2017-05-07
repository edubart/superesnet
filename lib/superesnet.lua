require 'pl'
require 'nn'
require 'optim'
require 'torchx'
require 'image'
signal = require("posix.signal")
tools = require './tools'

local cmd = torch.CmdLine()
cmd:text('superesnet - Super Resolution Deep Convolution Neural Network')
cmd:text()
cmd:text('Upscale Options')
cmd:option("-upscale", '', 'upscale mode, files or directory of images to upscale')
cmd:option("-outsuffix", '_upscaled', 'output upscaled image suffix')
cmd:option("-scalefactor", 2, 'scale factor to resize image')
cmd:option('-model', '', 'trained model to load')
--cmd:option('-color', 'rgb', 'optimization algorithm (grayscale|rgb|hsv|yuv|rgba)')
cmd:option('-upscalebackend', 'cpu', '(cpu|cuda|cudnn)')
cmd:text('Train Options')
cmd:option('-train', false, 'train mode')
cmd:option('-newmodel', '', 'create a new model to train on (vdsr|bnvdsr)')
cmd:option('-resume', '', 'model file to load from')
cmd:option('-save', false, 'save trained model to the same loaded file')
cmd:option('-saveto', '', 'save trained model to the specified model file')
cmd:option("-gpu", -1, 'gpu device id')
cmd:option("-threads", -1, 'number of CPU threads, auto detect by default')
cmd:option('-backend', 'cuda', '(cpu|cuda|cudnn)')
--cmd:option('-tensortype', 'float', '(float|double)')
cmd:option('-seed', 0, 'initial random seed')
cmd:option('-trainset', '', 'folder with images to train on')
cmd:option('-testset', '', 'folder with images to test on')
cmd:option('-trainsize', -1, 'limit number of train images')
cmd:option('-testsize', -1, 'limit number of test images')
--cmd:option('-learningrate', 0.0001, 'learn rate for optimizers')
cmd:option('-maxepochs', 0, 'number of epochs to run')
cmd:option('-maxiterations', 0, 'number of iterations to run')
cmd:option('-batchsize', 24, 'mini batch size')
cmd:option('-cropsize', 32, 'batch crop size')
cmd:option('-scalefactor', 2, 'scale factor to train on')
cmd:option('-printiterations', 1, 'number of iterations to print loss')
--cmd:option('-scalefactors', '', 'list of scale factors to train on separed by comma')
cmd:option('-optim', 'adam', 'optimization algorithm (adam|adamax|sgd)')
cmd:option('-color', 'rgb', 'optimization algorithm (rgb|y)')
cmd:option('-visdom', false, 'use visdom to visualize training')
cmd:option('-showinterval', 5, 'testset image show interval')
cmd:text()

settings = cmd:parse(arg)

-- default tensor to float
torch.setdefaulttensortype('torch.FloatTensor')

if settings.threads > 0 then
  torch.setnumthreads(settings.threads)
end

-- make reproduciple results
torch.manualSeed(settings.seed)

if settings.backend == 'cuda' or settings.upscalebackend == 'cuda' then
  require 'cutorch'
  require 'cunn'

  cutorch.manualSeed(settings.seed)

  if settings.gpu > 0 then
    cutorch.setDevice(settings.gpu)
  end
end

if settings.visdom then
  visual = require './visual'
end

if settings.color == 'y' then
  settings.channels = 1
else
  settings.channels = 3
end

if settings.upscale ~= '' then
  require './upscale'
else
  require './train'
end
