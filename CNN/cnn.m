%% Conv - ReLu - Pool - Conv - ReLu - Pool - Fc(logistic regression) 
clc; close; clear;
[x_train, y_train] = load_datasets();
learn_rate = 10e-3; fc_v = 0; conv1_v = 0; conv2_v = 0; conv1_v_b = 0; conv2_v_b = 0;

conv_layer1_num = 24;
conv_layer1 = conv_layer_init(3, 3, conv_layer1_num, 10e-4);

conv_layer1_b = 10e-4 * (rand(conv_layer1_num,1) - 0.5);

conv_layer2_num = 12;
conv_layer2 = conv_layer_init(3, 3, conv_layer2_num, 10e-4);

conv_layer2_b = 10e-4 * (rand(conv_layer2_num,1) - 0.5);

theta = 10e-4 * (rand(size(Fc_layer1_input, 1) - 0.5));
lambda = 1;
% 3 x 3 x 24 conv
[conv1_output, layer1_expand] = filter3(conv_layer1, conv_layer1_b, x_train{1,1}, 1, 1);
layer1_output = MyReLu(conv1_output);

pool_size = 2; step = 1;
[Pool_layer1_output, max_position1] = Max_Pool_layer(layer1_output, pool_size, step);

% 3 x 3 x 12 conv
[conv2_output, layer2_expand] = filter3(conv_layer2, conv_layer2_b, Pool_layer1_output, 1, 1);
layer2_output = MyReLu(conv2_output);

pool_size = 2; step = 1;
[Pool_layer2_output, max_position2] = Max_Pool_layer(layer2_output, pool_size, step);

% logistic regression
Fc_layer1_input = Pool_layer2_output(:);
Fc_layer1_input = [1;Fc_layer1_input];

[J, grad] = costFunctionReg(theta, Fc_layer1_input, y(1), lambda);
[theta, fc_v] = Updata_Momentum(grad, theta, learn_rate, fc_v);

% back
max_position2(max_position2==1) = grad;

max_position2(layer2_output==0) = 0;

[bias_grad2, theta_grad2] = conv_backward_theta(conv_layer2, grad, layer2_expand);

[conv_layer2, conv2_v] = Updata_Momentum(theta_grad2, conv_layer2, learn_rate, conv2_v);

[conv_layer2_b, conv2_v_b] = Updata_Momentum(bias_grad1, conv_layer2_b, learn_rate, conv2_v_b);

[conv_layer2_loss] = conv_backward_loss(conv_layer2, grad, layer2_expand, size(layer1_output, 2), size(layer1_output, 1), 1);

max_position1(max_position1==1) = conv_layer2_loss;

max_position1(layer1_output==0) = 0;

[bias_grad1, theta_grad1] = conv_backward_theta(conv_layer1, conv_layer2_loss, layer1_expand);
[conv_layer1, conv1_v] = Updata_Momentum(theta_grad1, conv_layer1, learn_rate, conv1_v);
[conv_layer1_b, conv1_v_b] = Updata_Momentum(bias_grad, conv_layer1_b, learn_rate, conv1_v_b);

learn_rate = 0.9 * learn_rate;