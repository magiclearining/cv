function [output, max_position] = Max_Pool_layer(input, pool_size, step)
%POOL_LAYER 此处显示有关此函数的摘要
%   此处显示详细说明
output = zeros((size(input, 1)-pool_size)/step + 1, (size(input, 2)-pool_size)/step + 1, size(input, 3));
%max_position = zeros((size(input, 1)-pool_size)/step + 1, (size(input, 2)-pool_size)/step + 1, size(input, 3), 2);
max_position = zeros(size(input));
i = 1;
for y = 1:step:size(input, 2)-pool_size+1
    j = 1;
    for x = 1:step:size(input, 1)-pool_size+1
        [max_y, index_y] = max(input(y:y+pool_size-1, x:x+pool_size-1, :));
        [max_num, index_x] = max(max_y);
        output(i, j, :) = max_num;
        max_position(input == max_num) = 1;
%         max_position(i, j, :, 2) = reshape(index_x, 1, 1, size(input, 3));
%         max_position(i, j, :, 1) = reshape(index_y(sub2ind(size(index_y), ones(1, size(input, 3)), reshape(index_x, 1, size(input, 3)), 1:size(input, 3))) ,1, 1, size(input, 3));
        j = j + 1;
    end
    i = i + 1;
end


end

