require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'code/mnist'
require 'pl'
require 'paths'

train = {}
train.__index = train

-- create model
function train:def_model ()
  self.model = nn.Sequential()

  self.model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
  self.model:add(nn.Tanh())
  self.model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  self.model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
  self.model:add(nn.Tanh())
  self.model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  self.model:add(nn.Reshape(64 * 4 * 4))
  self.model:add(nn.Linear(64 * 4 * 4, 200))
  self.model:add(nn.Tanh())
  self.model:add(nn.Linear(200, 10))

  print("Model: ")
  print(self.model)
end

-- create mini batch
function train:mini_batch (t, dataset)
  local s_data = dataset.data[{{t,
    math.min(t + self.batch_size - 1, dataset.n)}}]
  local inputs = s_data:clone()
  local s_labels = dataset.labels[{{t,
    math.min(t + self.batch_size - 1, dataset.n)}}]
  local targets = s_labels:clone()
  return inputs, targets
end

-- training function
function train:train_part (dataset)
  local epoch = 1 -- epoch and time
  local time = sys.clock()

  -- do one epoch
  print("<training> on training set")
  for t = 1, dataset.n, self.batch_size do
    local inputs, targets = self:mini_batch(t, dataset)

    -- create closure to evaluate f(X) and df/dX
    local feval = function (x)
       collectgarbage() -- just in case:

       -- get new parameters
       if x ~= self.parameters then self.parameters:copy(x) end
       self.grad_parameters:zero() -- reset gradients

       -- evaluate function for complete mini batch
       local outputs = self.model:forward(inputs)
       local f = self.criterion:forward(outputs, targets)
       -- estimate df/dW
       local df_do = self.criterion:backward(outputs, targets)
       self.model:backward(inputs, df_do)

       -- penalties (L1 and L2):
       if self.coefl1 ~= 0 or self.coefl2 ~= 0 then
          local norm, sign = torch.norm, torch.sign -- locals:
          f = f + self.coefl1 * norm(self.parameters, 1) -- loss l1
          f = f + self.coefl2 * norm(self.parameters, 2) ^ 2 / 2 -- loss l2

          -- Gradients:
          self.grad_parameters:add(sign(self.parameters):mul(self.coefl1) +
            self.parameters:clone():mul(self.coefl1))
       end

       for i = 1, self.batch_size do
          self.confusion:add(outputs[i], targets[i]) -- update confusion
       end

       return f, self.grad_parameters
    end

    -- Perform SGD step:
    sgd_state = sgd_state or {
      learningRate = self.learning_rate,
      momentum = self.momentum,
      learningRateDecay = 5e-7
    }
    optim.sgd(feval, self.parameters, sgd_state)

    -- disp progress
    xlua.progress(t, dataset.n)
  end

  -- time taken
  time = sys.clock() - time
  time = time / dataset.n
  print("<trainer> time to learn 1 sample = " .. (time * 1000) .. 'ms')

  -- print confusion matrix
  print(self.confusion)
  self.train_logger:add{['% mean class accuracy (train set)'] =
    self.confusion.totalValid * 100}
  self.confusion:zero()

  epoch = epoch + 1 -- next epoch
end

-- test function
function train:test_part (dataset)
  local time = sys.clock() -- local vars

  -- test over given dataset
  print('<trainer> on testing set')
  for t = 1, dataset.n, self.batch_size do
    xlua.progress(t, dataset.n) -- disp progress
    local inputs, targets = self:mini_batch(t, dataset)
    local preds = self.model:forward(inputs) -- test samples

    for i = 1, self.batch_size do
       self.confusion:add(preds[i], targets[i]) -- confusion
    end
  end

  -- timing
  time = sys.clock() - time
  time = time / dataset.n
  print("<trainer> time to test 1 sample = " .. (time * 1000) .. 'ms')

  -- print confusion matrix
  print(self.confusion)
  self.test_logger:add{['% mean class accuracy (test set)'] =
    self.confusion.totalValid * 100}
  self.confusion:zero()
end

function train.run ()
  local self = setmetatable({}, train)
  torch.setdefaulttensortype('torch.FloatTensor')

  self.seed = 1
  self.thread = 4
  self.batch_size = 10
  self.learning_rate = 0.05
  self.momentum = 0
  self.coefl1 = 0
  self.coefl2 = 0
  self.plot = true
  self.max_iter = 10

  torch.manualSeed(self.seed)
  torch.setnumthreads(self.thread)
  torch.setdefaulttensortype('torch.FloatTensor')

  self:def_model()
  self.parameters, self.grad_parameters = self.model:getParameters()
  self.model:add(nn.LogSoftMax())
  self.criterion = nn.ClassNLLCriterion()

  train_data = mnist.load('data/train')
  test_data = mnist.load('data/test')

  self.confusion = optim.ConfusionMatrix(train_data.classes)

  -- log results to files
  self.train_logger = optim.Logger('tmp/train.log')
  self.test_logger = optim.Logger('tmp/test.log')

  for i = 1, self.max_iter do
    -- train/test
    self:train_part(train_data)
    self:test_part(test_data)

    -- plot errors
    if self.plot then
      self.train_logger:style{['% mean class accuracy (train set)'] = '-'}
      self.test_logger:style{['% mean class accuracy (test set)'] = '-'}
      self.train_logger:plot()
      self.test_logger:plot()
    end
  end

  return self
end
