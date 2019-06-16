function [loss] = conv_backward_loss(conv_layer, y, size_x, size_y, size_z, step, pad)
  conv_layer = rot90(conv_layer,2);
  conv_layer = permute(conv_layer, [1, 2, 4, 3]);
  loss = filter3(conv_layer, zeros(size(conv_layer, 4), 1), y, 1, 1);
  
%   expand_y = expand_matrix(y_pad, size(conv_layer, 1), 1);
  
%   conv_layer = rot90(conv_layer,2);
%   conv_size = size(conv_layer, 1);
%   for i = 1:size(conv_layer, 4)
%       for j = 1:size(conv_layer, 3)
%         expand_conv((j-1)*conv_size*conv_size+1 : j*conv_size*conv_size, i) = reshape(conv_layer(:,:,j,i)', conv_size*conv_size, 1);
%       end
%   end  
%   expand_x = expand_y * expand_conv;

%   expand_x = expand_y * conv_expand';
%   loss = zeros(size_y, size_x, size_z); 
%   for i = 1:size_z
%     index = 1;
%     for j = 1:size_x:size(expand_x, 1) - size_x + 1
%       loss(index, :, i) = expand_x(j:j+size_x-1 , i)';
%       index = index + 1;
%     end
%   end
      
  
end