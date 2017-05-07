local depth = 16
local featuremaps = 64
local filtersize = 3
local upfiltersize = 4
local uppadding = 1
local nchannels = settings.channels

local function nonLinear()
  --return nn.ELU(0.2, true)
  return nn.LeakyReLU(0.2, true)
end

local function featureExtration()
  -- Feture extration
  local net = nn.Sequential()

  -- simple input normalization
  net:add(nn.AddConstant(-0.5))

  net:add(nn.SpatialConvolution(nchannels, featuremaps, filtersize, filtersize, 1, 1, 1, 1))
  net:add(nn.SpatialBatchNormalization(featuremaps))
  net:add(nonLinear())


  for layers = 1, depth do
    net:add(nn.SpatialConvolution(featuremaps, featuremaps, filtersize, filtersize, 1, 1, 1, 1))
    net:add(nn.SpatialBatchNormalization(featuremaps))
    net:add(nonLinear())
  end

  -- decovolution
  net:add(nn.SpatialFullConvolution(featuremaps, nchannels, upfiltersize, upfiltersize, 2, 2, uppadding, uppadding))

  return net
end

local function imageReconstruction()
  local net = nn.Sequential()
  -- bilinear upsampling
  --TODO: THIS!!
  net:add(nn.SpatialUpSamplingBilinear(2))
  return net
end

-- concat image input + VDSR residual
local concat = nn.ConcatTable()
concat:add(featureExtration()):add(imageReconstruction())

-- build the model
local model = nn.Sequential()
model:add(concat)
model:add(nn.CAddTable())

-- initialize weights
tools.weightinit(model, 'kaiming')

return model
