function [output, expand_x] = filter3(conv_layer, bias_parameter, img, step, pad)
  img_pad = zeros(size(img, 1)+2*pad, size(img, 2)+2*pad, size(img, 3));
  img_pad(pad+1:end-pad, pad+1:end-pad, :) = img;
  pos_x = 1;
  pos_y = 1;
  conv_size = size(conv_layer, 1);
  out_x_index = 1;
  out_y_index = 1;
  pad_size_x = size(img_pad, 2);
  pad_size_y = size(img_pad, 1);
  
  output = zeros((pad_size_y-conv_size)/step + 1, (pad_size_x-conv_size)/step + 1, size(conv_layer, 4));
  expand_x = zeros(((pad_size_x-conv_size)/step + 1) * ((pad_size_y-conv_size)/step + 1), (conv_size*conv_size)*3 + 1);
%  while(pos_y <= size(img_pad, 2) - conv_size)
%    out_x_index = 1;
%    pos_x = 1;
%    while(pos_x <= size(img_pad, 1) - conv_size)
%      output(out_y_index, out_x_index, :) = output(out_y_index, out_x_index,:) + reshape(sum(sum(sum(conv_layer .* double(img_pad(pos_x : pos_x+conv_size -1, pos_y : pos_y+conv_size-1 , :))))), 1,1,size(conv_layer, 4)) + reshape(bias_parameter, 1,1,size(conv_layer, 4));
%      pos_x =  pos_x + step;
%      out_x_index = out_x_index + 1;
%    end
%    pos_y = pos_y + step;
%    out_y_index = out_y_index + 1;
%  end
  pos_x = 1;
  pos_y = 1;    
  for i = 1:size(expand_x, 1)
    expand_x(i, 1 : conv_size*conv_size)                           = reshape(img_pad(pos_y:pos_y+conv_size-1, pos_x:pos_x+conv_size-1, 1)', 1, conv_size*conv_size);
    expand_x(i, conv_size*conv_size+1 : 2*conv_size*conv_size)     = reshape(img_pad(pos_y:pos_y+conv_size-1, pos_x:pos_x+conv_size-1, 2)', 1, conv_size*conv_size);
    expand_x(i, 2*conv_size*conv_size+1 : 3*conv_size*conv_size+1) = [reshape(img_pad(pos_y:pos_y+conv_size-1, pos_x:pos_x+conv_size-1, 3)', 1, conv_size*conv_size), 1];
    if(pos_x <= pad_size_x - conv_size)
      pos_x = pos_x + step;
    else
      pos_x = 1;
      pos_y = pos_y + step;
    end
  end
  expend_conv = zeros(conv_size*conv_size+1, size(conv_layer, 4));
  for i = 1:size(conv_layer, 4)
    expend_conv(1 : conv_size*conv_size, i)                           = reshape(conv_layer(:,:,1,i)', conv_size*conv_size, 1);
    expend_conv(conv_size*conv_size+1 : 2*conv_size*conv_size, i)   = reshape(conv_layer(:,:,2,i)', conv_size*conv_size, 1);
    expend_conv(2*conv_size*conv_size+1 : 3*conv_size*conv_size+1, i) = [reshape(conv_layer(:,:,3,i)', conv_size*conv_size, 1); bias_parameter(i)];
  end
  expand_y = expand_x * expend_conv;
  for i = 1:size(conv_layer, 4)
    index = 1;
    for j = 1:size(output, 2):size(expand_y, 1)
      output(index, :, i) = expand_y(j:j+size(output, 2)-1 , i)';
      index = index + 1;
    end
  end
end