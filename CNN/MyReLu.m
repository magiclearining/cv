function [output] = MyReLu(input)
    input(input<0) = 0;
    output = input;
end

