return function(opts)
  local opts = opts or {}
  local depth = opts.depth or 16
  local updepth = opts.upDepth or 2
  local featuremaps = opts.featureMaps or 64
  local filtersize = opts.filterSize or 3
  local pads = (filtersize-1)/2
  local nchannels = opts.channels or 3
  local upfiltersize = opts.upFilterSize or 3
  local uppads = (upfiltersize-1)/2
  local pads = (filtersize-1)/2
  local nonLinear = opts.nonLinear or function() return nn.LeakyReLU(0.2, true) end

  local function featureExtration()
    -- Feture extration
    local net = nn.Sequential()

    -- simple input normalization
    net:add(nn.AddConstant(-0.5))

    net:add(nn.SpatialConvolution(nchannels, featuremaps, filtersize, filtersize, 1, 1, pads, pads))
    net:add(nonLinear())

    for layers = 1, depth do
      net:add(nn.SpatialConvolution(featuremaps, featuremaps, filtersize, filtersize, 1, 1, pads, pads):noBias())
      net:add(nn.SpatialBatchNormalization(featuremaps))
      net:add(nonLinear())
    end

    -- resize decovolution
    net:add(nn.SpatialUpSamplingNearest(2))

    for layers = 1, updepth do
      net:add(nn.SpatialConvolution(featuremaps, featuremaps, upfiltersize, upfiltersize, 1, 1, uppads, uppads):noBias())
      net:add(nn.SpatialBatchNormalization(featuremaps))
      net:add(nonLinear())
    end

    net:add(nn.SpatialConvolution(featuremaps, nchannels, upfiltersize, upfiltersize, 1, 1, uppads, uppads))

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