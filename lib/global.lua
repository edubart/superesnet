require 'pl'
require 'nn'
require 'optim'
require 'torchx'
require 'image'
signal = require("posix.signal")

tools = require './tools'
trainer = require './trainer'
visual = require './visual'

require './optim/sgdhd'
require './optim/adamhd'
require './optim/eve'
require './charbonniercriterion'