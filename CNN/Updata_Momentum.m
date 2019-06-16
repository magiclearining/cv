function [output_theta, output_v] = Updata_Momentum(grad, theta, learn_rate, v, beta)
  output_v = v * beta + (1 - beta) * grad;
  output_theta = theta - learn_rate * output_v; 
end