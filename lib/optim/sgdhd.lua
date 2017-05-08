--[[
Hypergradient descent, SGD-HD and SGD-Nesterov-HD
https://arxiv.org/abs/1703.04782
Atilim Gunes Baydin, University of Oxford, March 2017
ARGS:
- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`         : learning rate
- `config.learningLearningRate` : hypergradient learning rate
- `config.weightDecay`          : weight decay
- `config.momentum`             : Nesterov momentum
- `state`                       : a table describing the state of the optimizer; after each
                                  call the state is modified
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
]]
function optim.sgdhd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local llr = config.learningLearningRate or 1e-6
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0

   -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   state.u = state.u or x.new(dfdx:size()):zero()
   state.lr = state.lr or lr

   -- (3) learning rate update (hypergradient descent)
   state.lr = state.lr + llr * torch.dot(dfdx, state.u)

   -- (3) apply Nesterov momentum
   if mom ~= 0 then
       state.v = state.v or x.new(dfdx:size()):zero()
       state.v:mul(mom):add(dfdx)
       state.u:copy(dfdx):add(mom, state.v)
   else
       state.u:copy(dfdx)
   end

   -- (4) update x
   x:add(-state.lr, state.u)

   -- return x*, f(x) before optimization
   return x, {fx}
end