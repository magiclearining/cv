# cv
MURA：

  由于极客云没有tf2，而我自己的电脑只有1050，模型写不了太大，就写了5层CNN加2层FC，层之间有BN，激活函数用的是selu，只使用elbow的图片，最后准确率在60左右
  模型在week16_7_21_tf2中，模型参数训练好了保持在MURA_keras_7_21.h5中

  新加了3个FC层的week16_7_21_tf2_3FC及其训练60个epoch的模型参数，90个epoch后准确率在70左右，120个epoch后准确率在75左右

  新加了densenet，算力和显存有限，准确率也只有75



chest X Ray 14:
  在chest X Ray 文件夹中
  
