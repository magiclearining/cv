function [expand_x] = expand_matrix(matrix, conv_size, step)
    pad_size_x = size(matrix, 2);
    pad_size_y = size(matrix, 1);
    channel = size(matrix, 3);
    expand_x = zeros(((pad_size_x-conv_size)/step + 1) * ((pad_size_y-conv_size)/step + 1), (conv_size*conv_size)*channel);
    pos_x = 1;
    pos_y = 1;    
    for i = 1:size(expand_x, 1)
        for j = 1:channel
            expand_x(i, (j-1)*conv_size*conv_size+1 : j*conv_size*conv_size) = reshape(matrix(pos_y:pos_y+conv_size-1, pos_x:pos_x+conv_size-1, j)', 1, conv_size*conv_size);
        end
        if(pos_x <= pad_size_x - conv_size)
            pos_x = pos_x + step;
        else
            pos_x = 1;
            pos_y = pos_y + step;
        end
    end
end