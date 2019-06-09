%% Conv - ReLu - Pool - Conv - ReLu - Pool - Fc(logistic regression) - sigmon - Softmax 
clc; close; clear;
%[x_train, y_train] = load_datasets();

% 3 x 3 x 24 conv
conv_layer1_num = 24;
conv_layer1 = conv_layer_init(3, 3, conv_layer1_num, 10e-4);

conv_layer1_b = 10e-4 * rand(conv_layer1_num,1);

layer1_output = MyReLu(filter3(conv_layer1, conv_layer1_b, x_train{1,1}, 1, 1));
[Pool_layer1_output, max_position1] = Max_Pool_layer(layer1_output, 2, 1);

% 3 x 3 x 12 conv
conv_layer2_num = 24;
conv_layer2 = conv_layer_init(3, 3, conv_layer2_num, 10e-4);

conv_layer2_b = 10e-4 * rand(conv_layer2_num,1);

layer2_output = MyReLu(filter3(conv_layer2, conv_layer2_b, Pool_layer1_output, 1, 1));

[Pool_layer2_output, max_position2] = Max_Pool_layer(layer2_output, 2, 1);

