# Experiments

Corresponding experiments are able to find in this repo

## U-NET experiments

![](UNet.png)

### Baseline

At first, we decided to implement [UNet](https://arxiv.org/abs/1505.04597), because it is effective and nevertheless simple network.
During the procedure of training I decided to use several augmentations: rotations with small angles, horizontal symetry, scale and perspective changes, crops. First three augmentation keeps image natural, whereas crops should improve model's silhouette detection. I used default Adam optimizer and cross entropy as loss function

### NoCrop

After baseline pipeline, it was noticed that model sometimes leave a lot of empty space inside of persons, or make cuts on clothes edges. I decided to launch model without crops, but it decreased quality of model.

### ExtraCrop
I noticed, that model sometimes can't distinguish persons from background, even if it is monotonous. Thus, I decided to add some colorful crops, especial blue and green for imitation of water, sky and grass. It didn't improve model's quality.

### ChangingLoss
It is often when some pictures from dataset contains huge class disbalance. Therefore, it is possible that weighted cross entropy may improve model's quality. After two experiment I haven't received significant improvements, thus I decided to switch model.


## Эксперименты с DeepUNet

![](DeepUNet.png)

### Baseline
Мне показалось осмысленным, попробовать использовать в данной задаче residual connections и возможно увелчить глубину сети. При рассмотрении различных 
статей я обнаружил [DeepUNet](https://arxiv.org/pdf/1709.00201.pdf). Так как в нашем случае мы имеем достаточно маленькие изображения, то из-за пулингов
просто увеличить количество сверточных слоев не получится, но можно увеличить количество сверток на каждом уровне. Изначально я запустил обычный UNet 
в котором просто добавил residual connection, но это лишь ухудшило результат. Возможно это было связано с тем, что использование суммы не позволяет увеличивать количество слоев на одном сверточном уровне.

### MoreLayers
После этого я решил удвоить количество сверток. В данном случае это позволило повторить результат UNet

### TwoNets
У меня возникла мысль, что имеет смысл рассматривать так же свертки с ядрами большего размера, чтобы сеть смогла распознавать более крупные объекты. К тому же само восприятие нейронной сети в качестве буквы U наталкивает на мысль сочетания нескольких таких букв с разными параметрами. В итогов эксперименте я добавил еще одну U с одинаковым количество сверточных слоев, заменив на каждом слое свертку 3x3 на 7x7 и уменьшив их количество. После, результаты этих сетей складываются. Данная идея позволила улучшить результат на несколько процентов. 

### SumNets
Я решил попробовать и заменить сложение двух результатов U на их конкатенацию, но это лишь ухудшило значение.

## Возможные дальнейшие эксперименты

В дальнейшем можно провести так же несколько экспериментов модернизируя DeepUNet, рассматривая различные аугментации, lr и лосс-функции по аналогии с UNet, однако вполне возможно что данный подход не даст результатов, поэтому на мой взгляд он не самый приоритетный. Стоит отметить, что основная идея которая увеличила производительность DeepUNet &mdash; несколько сетей параллельно, так же работает и для обычного UNet и возможно стоит рассмотреть такую модернизацию UNet. В этом же направлении стоит провести несколько экспериментов по добавлению других сверточных уровней, комбинированию различных сверток
