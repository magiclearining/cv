function [output, expand_x] = filter3(conv_layer, bias_parameter, img, step, pad)
    img_pad = zeros(size(img, 1)+2*pad, size(img, 2)+2*pad, size(img, 3));
    img_pad(pad+1:end-pad, pad+1:end-pad, :) = img;

    conv_size = size(conv_layer, 1);

    pad_size_x = size(img_pad, 2);
    pad_size_y = size(img_pad, 1);

    output = zeros((pad_size_y-conv_size)/step + 1, (pad_size_x-conv_size)/step + 1, size(conv_layer, 4));

    expand_x = expand_matrix(img_pad, conv_size, step);
    expand_x = [expand_x ones(size(expand_x, 1), 1)];
    expand_conv = zeros(conv_size*conv_size, size(conv_layer, 4));
    for i = 1:size(conv_layer, 4)
        for j = 1:size(conv_layer, 3)
          expand_conv((j-1)*conv_size*conv_size+1 : j*conv_size*conv_size, i) = reshape(conv_layer(:,:,j,i)', conv_size*conv_size, 1);
        end
    end
    expand_conv = [expand_conv; bias_parameter'];
    expand_y = expand_x * expand_conv;
    for i = 1:size(conv_layer, 4)
        index = 1;
        for j = 1:size(output, 2):size(expand_y, 1)
            output(index, :, i) = expand_y(j:j+size(output, 2)-1 , i)';
            index = index + 1;
        end
    end
end