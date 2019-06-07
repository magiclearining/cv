function [conv_layer] = conv_layer_init(size, channel, num, a)
  conv_layer = a*rand(size, size, 3, num);
end