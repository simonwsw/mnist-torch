require 'image'

-- random shuffle items in the array, in our case the filenames of images
function shuffle (array)
  local n, random, j = #array, math.random
  for i = 1, n do
    j, k = random(n), random(n)
    array[j], array[k] = array[k], array[j]
  end
  return array
end

-- load images
function load_image (data_dir)
  files = {}
  for file in paths.files(data_dir) do
    -- only insert filename with jpg
    if file:find("jpg" .. "$") then
      table.insert(files, paths.concat(data_dir, file))
    end
  end

  -- random shuffle names
  files = shuffle(files)

  -- load images
  images = {}
  for i, file in ipairs(files) do
    -- load each image
    table.insert(images, image.load(file))
  end

  print("Load " .. #images .. " images")
end

train_dir = "data/train"
load_image(train_dir)
test_dir = "data/train"
test_files = {}
