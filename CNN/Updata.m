function [output_theta, output_v] = Updata_Momentum(grad, theta, learn_rate, v, decay_rate = 0.9)
  output_v = v * decay_rate + (1 - decay_rate) * grad;
  output_theta = theta - learn_rate * output_v; 
end