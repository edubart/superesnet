return function(opts)
  local opts = opts or {}
  local depth = opts.depth or 16
  local featuremaps = opts.featureMaps or 64
  local filtersize = opts.filterSize or 3
  local pads = (filtersize-1)/2
  local nchannels = opts.channels or 3
  local pads = (filtersize-1)/2
  local nonLinear = opts.nonLinear or function() return nn.LeakyReLU(0.2, true) end

  local function featureExtration()
    local net = nn.Sequential()

    -- Insights:
    -- * removing bias reduces artifacts and increases learning rate
    -- * batch normalization also increases learning rate but generates artifacts again
    -- * the deeper the better

    net:add(nn.SpatialConvolution(nchannels, featuremaps, filtersize, filtersize, 1, 1, pads, pads):noBias())
    net:add(nonLinear())

    for layers = 1, depth do
      net:add(nn.SpatialConvolution(featuremaps, featuremaps, filtersize, filtersize, 1, 1, pads, pads):noBias())
      net:add(nonLinear())
    end

    -- tranposed decovolution
    net:add(nn.SpatialFullConvolution(featuremaps, featuremaps, 4, 4, 2, 2, 1, 1):noBias())
    net:add(nonLinear())
    net:add(nn.SpatialConvolution(featuremaps, nchannels, filtersize, filtersize, 1, 1, pads, pads):noBias())

    --[[
    -- use upsampling + convolution instead of transposed convolution
    net:add(nn.SpatialUpSamplingNearest(2))
    net:add(nn.SpatialConvolution(featuremaps, featuremaps, filtersize, filtersize, 1, 1, pads, pads):noBias())
    net:add(nonLinear())
    net:add(nn.SpatialConvolution(featuremaps, nchannels, filtersize, filtersize, 1, 1, pads, pads):noBias())
    ]]

    --[[
    -- pixel shuffle instead of transposed convolution
    net:add(nn.SpatialConvolution(featuremaps, nchannels * 2 * 2, 3, 3, 1, 1, 1, 1))
    net:add(nn.PixelShuffle(2))
    --]]

    return net
  end

  local function imageReconstruction()
    local net = nn.Sequential()
    net:add(nn.SpatialUpSamplingBilinear(2))
    return net
  end

  -- concat image input + VDSR residual
  local concat = nn.ConcatTable()
  concat:add(featureExtration()):add(imageReconstruction())

  -- build the model
  local model = nn.Sequential()
  model:add(concat)
  model:add(nn.CAddTable(true))

  -- initialize weights
  tools.weightinit(model, 'kaiming')

  return model
end