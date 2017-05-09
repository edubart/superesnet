local CharbonnierCriterion, Criterion = torch.class('nn.CharbonnierCriterion', 'nn.Criterion')

function CharbonnierCriterion:__init(eps)
  eps = eps or 1e-3
  Criterion.__init(self)
  local seq = nn.Sequential()
  seq:add(nn.View(-1))
  seq:add(nn.Square())
  seq:add(nn.AddConstant(eps*eps))
  seq:add(nn.Sqrt())
  seq:add(nn.Mean())
  self.seq = seq
end

function CharbonnierCriterion:updateOutput(input, target)
  self.output = self.seq:forward(target - input)
  return self.output:squeeze()
end

function CharbonnierCriterion:updateGradInput(input, target)
  self.gradInput = self.seq:backward(target - input, torch.Tensor{-1})
  return self.gradInput
end

return nn.CharbonnierCriterion