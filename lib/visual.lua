visdom = require('visdom')()

local visual = {}

function visual.showImage(img, name)
  local opts = {img = tools.convertImageColor(img, true)}
  if name then
    opts.win = name
    opts.options = { caption = name }
  end
  visdom:image(opts)
end

local firstShow = {}
function visual.showImageResults(prefix, input, output, predicted)
  for i=1,input:size(1) do
    if not firstShow[prefix] then
      visdom:image { win = 'in_' .. prefix .. i, img = tools.convertImageColor(input[i], true) }
      visdom:image { win = 'out_' .. prefix .. i, img = tools.convertImageColor(output[i], true) }
    end
    visdom:image { win = 'pred_' .. prefix .. i, img = tools.convertImageColor(predicted[i], true) }
  end
  firstShow[prefix] = true
end

local function showLoss(iterations)
  local plotdata
  local legends = {}
  for name,losses in pairs(losseslog) do
    local endy = math.min(#losses,iterations)
    local linedata = torch.Tensor(iterations):fill(losses[endy])
    linedata[{{1,endy}}]:copy(torch.Tensor(losses)[{{1,endy}}])
    --linedata = torch.log(linedata+1)
    if not plotdata then
      plotdata = linedata
    else
      plotdata = torch.cat(plotdata, linedata, 2)
    end
    table.insert(legends, name)
  end
  local id = vis:line {
    X = torch.range(1,iterations),
    Y = plotdata,
    win = 'pane_loss',
    options = {
      title = "Optimizations Loss",
      --ytype = 'log',
      legend = legends,
    }
  }
end

return visual