local depth = 20
local featuremaps = 64
local filtersize = 3
local nchannels = settings.channels
local pads = (filtersize-1)/2

local function nonLinear()
  --return nn.ELU(0.3, true)
  return nn.LeakyReLU(0.2, true)
end

-- VDSR model
local net = nn.Sequential()

-- firt conv layer
net:add(nn.SpatialConvolution(nchannels, featuremaps, filtersize, filtersize, 1, 1, pads, pads))
--net:add(nn.SpatialBatchNormalization(featuremaps))
net:add(nonLinear())

-- hidden conv layers
for layers = 1, depth do
  net:add(nn.SpatialConvolution(featuremaps, featuremaps, filtersize, filtersize, 1, 1, pads, pads))
  net:add(nn.SpatialBatchNormalization(featuremaps))
  net:add(nonLinear())
end

-- output conv layer
net:add(nn.SpatialConvolution(featuremaps, nchannels, filtersize, filtersize, 1, 1, pads, pads))

-- concat image input + VDSR residual
local concat = nn.ConcatTable()
concat:add(net):add(nn.Identity())

-- build the model
local model = nn.Sequential()
model:add(concat)
model:add(nn.CAddTable())

-- initialize weights
tools.weightinit(model, 'kaiming')

return model