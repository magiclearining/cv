function [bias_grad, theta_grad] = conv_backward_theta(conv_layer, y, expand_x)
    expand_y = zeros(size(y,1)*size(y,2), 1);
    for i = 1:size(y, 3)
        expand_y(:,i) = reshape(y(:, :, i)', size(y,1)*size(y,2), 1);
    end
    expand_k = expand_x' * expand_y;
    for i = 1:size(conv_layer, 4)
        theta_grad(:,:,:,i) = reshape(expand_k(1:end-1, i), size(conv_layer, 2), size(conv_layer, 1), size(conv_layer, 3));
        bias_grad(i) = expand_k(end, i);
    end

    theta_grad = permute(theta_grad, [2 1 3 4]);
end