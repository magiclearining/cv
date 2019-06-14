function [loss] = conv_backward_loss(conv_layer, y, expand_x, size_x, size_y, pad)
  y_pad = zeros(size(y, 1)+2*pad, size(y, 2)+2*pad, size(y, 3));
  y_pad(pad+1:end-pad, pad+1:end-pad, :) = y;
  
  expand_y = expand_matrix(y_pad);
  
  conv_layer = fliplr(flipud(conv_layer));
  conv_size = size(conv_layer, 1);
  for i = 1:size(conv_layer, 4)
    expend_conv(1 : conv_size*conv_size, i)                           = reshape(conv_layer(:,:,1,i)', conv_size*conv_size, 1);
    expend_conv(conv_size*conv_size+1 : 2*conv_size*conv_size, i)   = reshape(conv_layer(:,:,2,i)', conv_size*conv_size, 1);
    expend_conv(2*conv_size*conv_size+1 : 3*conv_size*conv_size, i) = reshape(conv_layer(:,:,3,i)', conv_size*conv_size, 1);
  end
  
  expan_x = expand_y * expand_conv;
  
  loss = zeros(size_x, size_y, size(conv_layer, 4));
  for i = 1:size(conv_layer, 4)
    index = 1;
    for j = 1:size_x:size(expand_x, 1)
      loss(index, :, i) = expand_x(j:j+size_x-1 , i)';
      index = index + 1;
    end
  end
      
  
end