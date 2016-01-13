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
    table.insert(fields, c)
  end)
  return fields
end

-- load images
local function load_image (data_dir)
  local files = {}
  for file in paths.files(data_dir) do
    -- only insert filename with jpg
    if file:find("jpg" .. "$") then
      table.insert(files, paths.concat(data_dir, file))
    end
  end

  -- random shuffle names
  local files = shuffle(files)

  -- load images
  local images = {}
  local labels = {}
  for i, file in ipairs(files) do
    -- load each image
    table.insert(images, image.load(file))
    local paths = split(file, "/")
    local file_name = paths[#paths]
    local class = split(split(file_name, "_")[2], ".")[1]
    table.insert(labels, tonumber(class))
  end

  print("Load " .. #images .. " images")
  return images, labels
end

local function image2data (images)
  local n = #images
  local data = torch.Tensor(n, 1, mnist_data.image_size, mnist_data.image_size)

end

load_image(mnist_data.train_dir)
