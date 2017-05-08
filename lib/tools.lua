local tools = {}

function tools.tabletotensor(t)
  return torch.cat(t,1):resize(torch.LongStorage{#t, unpack(torch.totable(t[1]:size()))})
end

function tools.mse2psnr(mse)
   return 10 * math.log10(1/math.max(mse,1e-16))
end

function tools.fatalerror(msg, ...)
  io.stderr:write(string.format(msg .. "\n",...))
  os.exit(-1)
end

function tools.convertImageColor(colortype, img, back)
  if colortype == 'rgb' then
    return img
  elseif colortype == 'y' then
    if back then
      return img
    else
      return image.rgb2y(img)
    end
  else
    error('unknow image color type')
  end
end

function tools.loadImages(path, max, colortype)
  local files
  if type(path) == 'table' then
    files = {}
    for i,p in pairs(path) do
      tablex.insertvalues(files, dir.getfiles(p))
    end
    path = pretty.write(path, "")
  else
    files = dir.getfiles(path)
  end
  colortype = colortype or 'rgb'
  local images = {}
  local timer = torch.Timer()
  for i,file in ipairs(files) do
    if max and max >= 0 and i>max then break end
    local img = tools.convertImageColor(colortype, image.load(file, 3, "float"))
    table.insert(images, img)
  end
  utils.printf('Loaded dataset "%s" with %d images in %.3fs\n', path, #images, timer:time().real)
  return images
end

function tools.rescaleImages(images, factor)
  local outimgs = {}
  local timer = torch.Timer()
  for i,img in ipairs(images) do
    local height, width = img:size(2), img:size(3)
    local lrimg = image.scale(image.scale(img, "*1/" .. factor), width, height, "bicubic")
    table.insert(outimgs, lrimg)
  end
  utils.printf('Rescaled images in %.3fs\n', timer:time().real)
  return outimgs
end

function tools.downscaleImages(images, factor)
  local outimgs = {}
  local timer = torch.Timer()
  for i,img in ipairs(images) do
    local height, width = img:size(2), img:size(3)
    local lrimg = image.scale(img, "*1/" .. factor)
    table.insert(outimgs, lrimg)
  end
  utils.printf('Downscaled images in %.3fs\n', timer:time().real)
  return outimgs
end

function tools.prepareBackend(backend, opts)
  opts = opts or {}

  -- default tensor to float
  torch.setdefaulttensortype('torch.FloatTensor')

  -- number of threads for blas
  if opts.threads then
    torch.setnumthreads(opts.threads)
  end

  -- make reproduciple results
  if opts.seed then
    torch.manualSeed(opts.seed)
  end

  if backend == 'cuda' then
    require 'cutorch'
    require 'cunn'

    if opts.seed then
      cutorch.manualSeed(opts.seed)
    end

    if opts.gpu then
      cutorch.setDevice(opts.gpu)
    end
  end

  utils.printf('Using backend %s\n', backend)
end

function tools.prepareTensorBackend(backend, tensor, clone)
  if backend == 'cuda' then
    return tensor:cuda()
  end
  if clone then
    return tensor:clone()
  end
  return tensor
end

function tools.prepareTensorsBackend(backend, tensors)
  if backend == 'cuda' then
    for i=1,#tensors do
      tensors[i] = tensors[i]:cuda()
    end
  end
  return tensors
end

function tools.prepareModelBackend(backend, model, criterion)
  if backend == 'cuda' then
    model:cuda()
    criterion:cuda()
  end
end

function tools.loadModel(model, modelOpts, modelFile)
 local model = require('./models/' .. model)(modelOpts)
 local loaded = false
 if modelFile and modelFile ~= '' and path.exists(modelFile) then
    local loadedModel = torch.load(modelFile)
    -- same architeture
    if tostring(model) == tostring(loadedModel) then
      model = loadedModel
      loaded = true
    end
  end
  model:evaluate()
  if loaded then
    print('Loaded trained model from file!')
  else
    print('Created a brand new model!')
  end
  return model
end

function tools.saveModel(path, model)
  if not path or path == '' then
    print 'Model not saved!'
    return
  end
  model:clearState()
  torch.save(path, model)
  print('Model saved to ' .. path)
end

function tools.setupAutoSave(modelFile, model, enabled)
  if not enabled or not modelFile or modeFile == '' then return end

  local function term(signum)
    tools.saveModel(modelFile, model:float())
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR2, signal.SIG_DFL)
    signal.raise(signum)
  end
  signal.signal(signal.SIGINT, term)
  signal.signal(signal.SIGTERM, term)
  signal.signal(signal.SIGUSR2, term)
end

tools.weightinit = require('./weightinit')

return tools