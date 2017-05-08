return function(opts)
  local opts = opts or {}
  local depth = opts.depth or 16
  local featuremaps = opts.featureMaps or 64
  local filtersize = opts.filterSize or 3
  local nchannels = opts.channels or 3
  local pads = (filtersize-1)/2
  local nonLinear = opts.nonLinear or function() return nn.ReLU(true) end

  -- VDSR model
  local net = nn.Sequential()

  -- simple input normalization
  net:add(nn.AddConstant(-0.5))

  -- firt conv layer
  net:add(nn.SpatialConvolution(nchannels, featuremaps, filtersize, filtersize, 1, 1, pads, pads))
  net:add(nonLinear())

  -- hidden conv layers
  for layers=1,depth do
    net:add(nn.SpatialConvolution(featuremaps, featuremaps, filtersize, filtersize, 1, 1, pads, pads))
    if opts.batchNormalization then
      net:add(nn.SpatialBatchNormalization(featuremaps))
    end
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
end
