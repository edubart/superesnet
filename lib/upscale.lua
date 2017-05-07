if settings.model == '' then
  tools.fatalerror('No model to load from, please specify one!')
elseif not path.exists(settings.model) then
  tools.fatalerror('Model "%s" not found!', settings.model)
end

local timer = torch.Timer()
print('Loading model ' .. settings.model)
local model = torch.load(settings.model)
model:evaluate()

local infile = settings.upscale
if not path.exists(infile) then
  tools.fatalerror('Image "%s" not found!', infile)
end
local outfile = infile:gsub('(%.%a+)$', settings.outsuffix .. '%1')

utils.printf('Converting "%s" to "%s"\n', infile, outfile)
local img = image.load(infile, 3, 'float')
local nchan, height, width = img:size(1), img:size(2), img:size(3)
local inimg = img
--settings.pyramid = true

if not settings.pyramid then
  inimg = image.scale(img, settings.scalefactor*width, settings.scalefactor*height)
end

if settings.upscalebackend == 'cuda' then
  model:cuda()
  inimg = inimg:cuda()
elseif model._type:find('Cuda') then
  model:float()
end

local upscaledimg = model:forward(inimg)

image.save(outfile, upscaledimg[1])
utils.printf('Finished in %.03fs\n', timer:time().real)