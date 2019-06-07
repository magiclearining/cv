function [x_train, y_train] = load_datasets()
  fid = fopen('/home/mr/code/CNN/MURA-v1.1/train_image_paths.csv');
  train_image_paths = textscan(fid, '%s');
  fclose(fid);
  x_train = cell(size(train_image_paths{1,1},1), 20);
  for i = 1:100 %size(train_image_paths{1,1},1)
    if strfind(train_image_paths{1,1}{i,1}, 'positive');
      y_train(i) = 1;
    else
      y_train(i) = 0;
    end
    
    pos = strfind(train_image_paths{1,1}{i,1}, 'patient');
    num = str2num(train_image_paths{1,1}{i,1}(pos+7 : pos+11));
    index_num = 1;
    while(!isempty(x_train{num, index_num}))
      index_num += 1;
    end
    x_train{num, index_num} = imread(train_image_paths{1,1}{i,1});
  end    

end