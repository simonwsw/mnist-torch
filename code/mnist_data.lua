require 'image'
require 'torch'

mnist_data = {}

mnist_data.train_dir = "data/train"
mnist_data.test_dir = "data/test"
mnist_data.image_size = 28

-- random shuffle items in the array, in our case the filenames of images
local function shuffle (array)
  local n, random, j = #array, math.random
  for i = 1, n do
    j, k = random(n), random(n)
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
local function load_image (data_dir)
  local files = {}
  for file in paths.files(data_dir) do
    -- only insert filename with jpg
    if file:find("jpg" .. "$") then
      files[#files + 1] = paths.concat(data_dir, file)
    end
  end

  -- random shuffle names
  local files = shuffle(files)

  -- load images
  local images = {}
  local labels = {}
  for i, file in ipairs(files) do
    -- load each image
    images[#images + 1] = image.load(file,
      mnist_data.channel, mnist_data.tensortype)
    local paths = split(file, "/")
    local file_name = paths[#paths]
    local class = split(split(file_name, "_")[2], ".")[1]
    labels[#labels + 1] = tonumber(class)
  end

  print("Load " .. #images .. " images")
  return images, labels
end

local function image2data (images, labels)
  local data_set = {}
  local n = #images
  local tensor_data = torch.Tensor(n, mnist_data.channel,
    mnist_data.image_size, mnist_data.image_size)
  local tensor_labels = torch.Tensor(n)

  -- load images and labels into tensor
  for i = 1, n do
    tensor_data[{i, 1}] = images[i]
    tensor_labels[i] = labels[i]
  end

  data_set.data = tensor_data
  data_set.labels = tensor_labels

  return data_set
end

function mnist_data.load (dir_type)
  local images, labels = load_image(mnist_data[dir_type])
  local data_set = image2data(images, labels)
  return data_set
end
