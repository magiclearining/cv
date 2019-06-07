%% Conv - ReLu - Pool - Conv - ReLu - Pool - Fc - ReLu - Softmax 
%clc; close; clear all;
%[x_train, y_train] = load_datasets();

% 3 x 3 x 24 conv
conv_layer1_num = 24;
conv_layer1 = conv_layer_init(3, 3, conv_layer1_num, 10e-4);

conv_layer1_b = 10e-4 * rand(conv_layer1_num,1);

output = filter3(conv_layer1, conv_layer1_b, x_train{1,1}, 1, 1);
