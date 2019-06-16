%% Conv - ReLu - Pool - Conv - ReLu - Pool - Fc(logistic regression) 
%clc; close; clear;
%[x_train, y_train] = load_datasets();
learn_rate = 5*10e-6; fc_v = 0; conv1_v = 0; conv2_v = 0; conv1_v_b = 0; conv2_v_b = 0;

conv_layer1_num = 24;
conv_layer1 = conv_layer_init(3, 1, conv_layer1_num, 10e-4);

conv_layer1_b = 10e-4 * (rand(conv_layer1_num,1) - 0.5);

conv_layer2_num = 12;
conv_layer2 = conv_layer_init(3, conv_layer1_num, conv_layer2_num, 10e-4);

conv_layer2_b = 10e-4 * (rand(conv_layer2_num,1) - 0.5);

lambda = 1;


% 3 x 3 x 3 x 24 conv
for i = 1:100
    [conv1_output, layer1_expand] = filter3(conv_layer1, conv_layer1_b, x_train{i, 1}, 1, 1);
    layer1_output = MyReLu(conv1_output);

    pool_size = 2;
    [Pool_layer1_output, max_position1] = Max_Pool_layer(layer1_output, pool_size);

% 3 x 3 x x 24 x 12 conv
    [conv2_output, layer2_expand] = filter3(conv_layer2, conv_layer2_b, Pool_layer1_output, 1, 1);
    layer2_output = MyReLu(conv2_output);

    pool_size = 2;
    [Pool_layer2_output, max_position2] = Max_Pool_layer(layer2_output, pool_size);

    % logistic regression
    Fc_layer1_input = Pool_layer2_output(:);
    Fc_layer1_input = [1;Fc_layer1_input];
    if(i==1)
        theta = 10e-5 * (rand(size(Fc_layer1_input)) - 0.5);
    end
    [J, grad] = costFunctionReg(theta, Fc_layer1_input', y_train(i), lambda);

    if(mod(i, 1)==0) 
        J
    end
    [theta, fc_v] = Updata_Momentum(grad, theta, learn_rate, fc_v, 0);

    % back
    max_position2(max_position2==1) = grad(2:end);

    max_position2(layer2_output==0) = 0;

    [bias_grad2, theta_grad2] = conv_backward_theta(conv_layer2, max_position2, layer2_expand);

    [conv_layer2_loss] = conv_backward_loss(conv_layer2, max_position2, size(Pool_layer1_output, 2), size(Pool_layer1_output, 1), size(Pool_layer1_output, 3), 1, 1);

    [conv_layer2, conv2_v] = Updata_Momentum(theta_grad2, conv_layer2, learn_rate, conv2_v, 0);

    [conv_layer2_b, conv2_v_b] = Updata_Momentum(bias_grad2', conv_layer2_b, learn_rate, conv2_v_b, 0);

    max_position1(max_position1==1) = conv_layer2_loss;

    max_position1(layer1_output==0) = 0;

    [bias_grad1, theta_grad1] = conv_backward_theta(conv_layer1, max_position1, layer1_expand);
    [conv_layer1, conv1_v] = Updata_Momentum(theta_grad1, conv_layer1, learn_rate, conv1_v, 0);
    [conv_layer1_b, conv1_v_b] = Updata_Momentum(bias_grad1', conv_layer1_b, learn_rate, conv1_v_b, 0);

    learn_rate = 0.99 * learn_rate;
end