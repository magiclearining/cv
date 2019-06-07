function [output] = filter3(conv_layer, bias_parameter, img, step, pad)
  img_pad = zeros(size(img, 1)+2*pad, size(img, 2)+2*pad, 3);
  img_pad(pad+1:end-pad, pad+1:end-pad, :) = img;
  pos_x = 1;
  pos_y = 1;
  conv_size = size(conv_layer, 1);
  output = zeros(size(img, 1), size(img, 2), size(conv_layer, 4));
  out_x_index = 1;
  out_y_index = 1;
  while(pos_y <= size(img_pad, 2) - conv_size)
    out_x_index = 1;
    pos_x = 1;
    while(pos_x <= size(img_pad, 1) - conv_size)
      for i = 1:size(conv_layer, 4)
        output(out_x_index, out_y_index, i) += sum(sum(sum(conv_layer(:, :, :, i) .* double(img_pad(pos_x : pos_x+conv_size -1, pos_y : pos_y+conv_size-1 , :))))) + bias_parameter(i);
      end
      pos_x += step;
      out_x_index += 1;
    end
    pos_y += step;
    out_y_index += 1;
  end    
end