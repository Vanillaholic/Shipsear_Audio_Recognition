# Shipsear音频识别

初次学习深度学习,随便搭了一个模型~~(整个过程不到10分钟)~~,没想到准确率上0.8了......

供学生在以后的基础上修改

使用的是Shipsear数据,训练结果如下所示

<img src="https://cdn.jsdelivr.net/gh/vanillaholic/image-bed@main/img/Shipsear_Loss.png" alt="Shipsear_Loss" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/vanillaholic/image-bed@main/img/Shipsear_Acc.png" alt="Shipsear_Acc" style="zoom:25%;" />

- 环境要求

  - cuda版本:11.8
  - python版本:3.10.18

  | python库     | 版本   |
  | ------------ | :----- |
  | torch        | 2.7.1  |
  | torchaudio   | 2.7.1  |
  | torchmetrics | 1.7.3  |
  | torchtext    | 0.18.0 |
  | torchvision  | 0.22.1 |
  | librosa      | 0.11.0 |
  | pycparser    | 2.22   |
  | swanlab      | 0.6.4  |

  

- 数据集获取:https://underwaternoise.atlanttic.uvigo.es
