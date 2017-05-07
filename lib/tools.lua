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

function tools.convertImageColor(img, back)
  if settings.color == 'rgb' then
    return img
  elseif settings.color == 'y' then
    if back then
      return img
    else
      return image.rgb2y(img)
    end
  end
end

function tools.loadImages(path, max)
  local images = {}
  local timer = torch.Timer()
  for i,file in ipairs(dir.getfiles(path)) do
    if max and max >= 0 and i>max then break end
    local img = tools.convertImageColor(image.load(file, 3, "float"))
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

function tools.prepareTensorBackend(tensor, clone)
  if settings.backend == 'cuda' then
    return tensor:cuda()
  end
  if clone then
    return tensor:clone()
  end
  return tensor
end

function tools.prepareTensorsBackend(tensors)
  if settings.backend == 'cuda' then
    for i=1,#tensors do
      tensors[i] = tensors[i]:cuda()
    end
  end
  return tensors
end

function tools.prepareModelBackend(model, criterion, tensors)
  if settings.backend == 'cuda' then
    model:cuda()
    criterion:cuda()
  end
  utils.printf('Using backend %s\n', settings.backend)
end

tools.weightinit = require('./weightinit')

return tools