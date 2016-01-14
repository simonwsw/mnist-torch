require 'image'
require 'torch'

mnist = {}
mnist.__index = mnist

-- random shuffle items in the array, in our case the filenames of images
local function shuffle (array)
  local n, random = #array, math.random
  for i = 1, n do
    local j, k = random(n), random(n)
    array[j], array[k] = array[k], array[j]
  end
  return array
end

-- seperate string by sep
local function split (str, sep)
  local sep, fields = sep or " ", {}
  local pattern = string.format("([^%s]+)", sep)
  str:gsub(pattern, function (c)
    fields[#fields + 1] = c
  end)
  return fields
end

-- load images
function mnist:load_image (dir)
  local files = {}
  for file in paths.files(dir) do
    if file:find("jpg" .. "$") then
      files[#files + 1] = paths.concat(dir, file)
    end
  end
  local files = shuffle(files) -- random shuffle names

  -- load images
  local images, labels = {}, {}
  for i, file in ipairs(files) do
    images[i] = image.load(file,
      1, FloatTensor) -- load each image
    local paths = split(file, "/") -- parse label
    local file_name = paths[#paths]
    local class = split(split(file_name, "_")[2], ".")[1]
    labels[i] = tonumber(class)
  end

  self.n = #images
  self.channel, self.size = images[1]:size(1), images[1]:size(2)
  print("Load " .. #images .. " images")
  return images, labels
end

function mnist:image2data (images, labels)
  local tensor_data = torch.Tensor(self.n, self.channel, self.size, self.size)
  local tensor_labels = torch.Tensor(self.n)

  -- load images and labels into tensor
  for i = 1, self.n do
    tensor_data[{i, 1}] = images[i]
    tensor_labels[i] = labels[i]
  end

  self.dataset = {tensor_data, tensor_labels}
end

-- load mnist data
function mnist.load (dir)
  local self = setmetatable({}, mnist)
  torch.setdefaulttensortype('torch.FloatTensor')

  local images, labels = self:load_image(dir)
  self:image2data(images, labels)
  return self
end
