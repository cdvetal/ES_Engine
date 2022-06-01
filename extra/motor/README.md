- [Instructions](#instructions)
  - [Command Line Arguments](#command-line-arguments)
    - [--save-folder](#--save-folder)
    - [--target-class](#--target-class)
    - [--networks](#--networks)
    - [--random-seed](#--random-seed)
    - [--target-fit](#--target-fit)
    - [--max-gens](#--max-gens)
    - [--img-size](#--img-size)
    - [--pop-size](#--pop-size)
    - [--checkpoint-freq](#--checkpoint-freq)
    - [--save-all](#--save-all)
    - [--from-checkpoint](#--from-checkpoint)
  - [Starting from Checkpoint](#starting-from-checkpoint)

# Instructions

## Command Line Arguments

To check all possible arguments and their defaults:

```python pynevar.py -h```

To change the defaults, just edit the global variables in the file.

### --save-folder
This one is mandatory. Just type the main folder where you want to create a sub_folder for the current run/experiment. Example:

```python pynevar.py --save-folder experiments```

### --target-class
Choose the class to optimize. Example:

```python pynevar.py --save-folder experiments --target-class spider_web```

### --networks
Choose the networks to guide the evolution. Please separate them **with commas and no spaces**! Example:

```python pynevar.py --save-folder experiments --target-class spider_web --networks mobilenet,mobilenetv2```

Also possible to use Tom White's groups:
```python
model_groups = {
    "london,": "xception,vgg16,vgg19,resnet50,resnet50v2,resnet101,resnet152,resnet101v2,resnet152v2,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,densenet121,densenet169,densenet201,nasnet,nasnetmobile,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb07,",
    "standard6,": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,",
    "standard9,": "standard6,inceptionresnetv2,nasnet,nasnetmobile,",
    "standard13,": "standard9,densenet121,densenet169,densenet201,mobilenetv2,",
    "standard18,": "standard13,resnet101,resnet152,resnet50v2,resnet101v2,resnet152v2,",
    "train1,": "vgg19,resnet50,inceptionv3,xception,",
    "standard,":  "standard6,",
    "all,": "standard18,",
}
```

You can also choose from the list of files in **scoring/** folder.

### --random-seed
Choose the random seed. Default is None. Example:

```python pynevar.py --save-folder experiments --target-class spider_web --networks mobilenet,mobilenetv2 --random-seed 69```

### --target-fit
Choose the target fitness value. Default is 0.999. Example:

```python pynevar.py --save-folder experiments --target-class spider_web --networks mobilenet,mobilenetv2 --random-seed 69 --target-fit 0.75```

### --max-gens
Choose maximum generations. Default is 10. Example:

```python pynevar.py --save-folder experiments --target-class spider_web --networks mobilenet,mobilenetv2 --random-seed 69 --target-fit 0.75 --max-gens 500```

### --img-size
Image dimensions during evaluation. Default is 512. Example:

```python pynevar.py --save-folder experiments --target-class spider_web --networks mobilenet,mobilenetv2 --random-seed 69 --target-fit 0.75 --max-gens 500 --img-size 256```

Currently, we're assuming images are always square.

### --pop-size
Population size. Default is 10. Example:

```python pynevar.py --save-folder experiments --target-class spider_web --networks mobilenet,mobilenetv2 --random-seed 69 --target-fit 0.75 --max-gens 500 --img-size 256 --pop-size 100```

### --checkpoint-freq
Checkpoint backup frequency. Default is every 10 generations. Example:

```python pynevar.py --save-folder experiments --target-class spider_web --networks mobilenet,mobilenetv2 --random-seed 69 --target-fit 0.75 --max-gens 500 --img-size 256 --pop-size 100 --checkpoint-freq 50```

### --save-all
Save all Individual images. Can be true or false. Default is False. Example:

```python pynevar.py --save-folder experiments --target-class spider_web --networks mobilenet,mobilenetv2 --random-seed 69 --target-fit 0.75 --max-gens 500 --img-size 256 --pop-size 100 --checkpoint-freq 50 --save-all true```

### --from-checkpoint
Carefully read [Starting from Checkpoint](#starting-from-checkpoint)

## Starting from Checkpoint
Please make sure that for the **--save-folder** you specify the path where the checkpoint ".pkl" file is located and for the **--from-checkpoint** specify the full name of the ".pkl" checkpoint file. Example:

```python pynevar.py --save-folder experiments/pynevar_spider_web_1909_20_100 --target-class spider_web --networks mobilenet,mobilenetv2 --random-seed 1909 --max-gens 50 --pop-size 100 --from-checkpoint pynevar_spider_web_1909_checkpoint.pkl```

It will automatically create a ***from_checkpoint*** folder inside the ***--save-folder***