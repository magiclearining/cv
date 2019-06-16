function [output, max_position] = Max_Pool_layer(input, pool_size)
    step = pool_size;
    output = zeros(fix((size(input, 1)-pool_size)/step + 1), fix((size(input, 2)-pool_size)/step + 1), size(input, 3));
    %max_position = zeros((size(input, 1)-pool_size)/step + 1, (size(input, 2)-pool_size)/step + 1, size(input, 3), 2);
    max_position = zeros(size(input));
    i = 1;
    for y = 1:step:size(input, 1)-pool_size+1
        j = 1;
        for x = 1:step:size(input, 2)-pool_size+1
            [max_y, index_y] = max(input(y:y+pool_size-1, x:x+pool_size-1, :));
            [max_num, index_x] = max(max_y);
            output(i, j, :) = max_num;
            for k = 1:size(max_position, 3)
                max_position(y+index_y(1, index_x(k), k)-1, x+index_x(k)-1, k) = 1;
            end
    %         max_position(i, j, :, 2) = reshape(index_x, 1, 1, size(input, 3));
    %         max_position(i, j, :, 1) = reshape(index_y(sub2ind(size(index_y), ones(1, size(input, 3)), reshape(index_x, 1, size(input, 3)), 1:size(input, 3))) ,1, 1, size(input, 3));
            j = j + 1;
        end
        i = i + 1;
    end


end

