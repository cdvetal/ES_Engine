# ES Engine

This tool allows the use of a CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm to evolve renderings to aproximate a certaint class of the imagenet dataset. It relies on the use of pre-trained classifiers to guide the evolutionary process by calculating the reward value based on the certainty of these classifiers on the desired class. 


## Instructions
This tool is able to work without any arguments as they have a pre-defined configuration file where all the required information is stored. However, using the command line arguments it is possible to change these values.

Examples:

```python es_engine.py```

```python es_engine.py --img-size 1024 --target-class goldfish```

```python es_engine.py --random-seed 1 --n-gens 200 --renderer organic --img-size 112 --networks inceptionv3,vgg16,xception,mobilenet,efficientnetb4,efficientnetb0 --target-class hummingbird```


- [Command Line Arguments](#command-line-arguments)
  - [--random-seed](#--random-seed)
  - [--save-folder](#--save-folder)
  - [--n-gens](#--n-gens)
  - [--pop-size](#--pop-size)
  - [--save-all](#--save-all)
  - [--verbose](#--verbose)
  - [--num-lines](#--num-lines)
  - [--num-cols](#--num-cols)
  - [--renderer](#--renderer)
  - [--img-size](#--img-size)
  - [--target-class](#--target-class)
  - [--networks](#--networks)
  - [--target-fit](#--target-fit)
  - [--from-checkpoint](#--from-checkpoint)
  - [--init-mu](#--init-mu)
  - [--init-sigma](#--init-sigma)
  - [--sigma](#--sigma)
  - [--clip-influence](#--clip-influence)
  - [--clip-model](#--clip-model)
  

### --random-seed
Choose the random seed. Default is None. Example:

```python es_engine --random-seed 1909```

### --save-folder
Directory to experiment outputs. Default is 'experiments'. Example:

```python es_engine --save-folder saves```

### --n-gens
Maximum generations. Default is 100. Example:

```python es_engine --n-gens 150```

### --pop-size
Population size. Default is 40. Example:

```python es_engine --pop-size 50```

### --save-all
Save all Individual images. Default is False. Example:

```python es_engine --save-all```

### --verbose
Verbose. Default is False. Example:

```python es_engine --verbose```

### --num-lines
Number of lines. Default is 17. Example:

```python es_engine --num-lines 5```

### --num-cols
Number of columns. Default is 8. Example:

```python es_engine --num-cols 5```

### --renderer
Choose the renderer. Default is 'pylinhas'. Example:

```python es_engine --renderer chars```

### --img-size
Image dimensions during testing. Default is 256. Example:

```python es_engine --img-size 512```

### --target-class
Which target classes to optimize. Default is 'birdhouse'. Example:

```python es_engine --target-class goldfish```

### --networks
Comma separated list of networks. Default is 'mobilenet,vgg16'. Example:

```python es_engine --networks vgg16,vgg19,mobilenet```

### --target-fit
Target fitness stopping criteria. Default is 0.999. Example:

```python es_engine --target-fit 0.9```

### --from-checkpoint
Checkpoint file from which you want to continue evolving. Default is None. Example:

```python es_engine --from-checkpoint Experiment_name.pkl```

### --init-mu
Mean value for the initialization of the population. Default is 0.5. Example:

```python es_engine --init-mu 1909```

### --init-sigma
Standard deviation value for the initialization of the population. Default is 0.25. Example:

```python es_engine --init-sigma 0.2```

### --sigma
The initial standard deviation of the distribution. Default is 0.2. Example:

```python es_engine --sigma 0.1```

### --clip-influence
The influence CLIP has in the generation (0.0 - 1.0). Default is 0.0. Example:

```python es_engine --clip-influence 0.5```

### --clip-model
Name of the CLIP model to use. Default is 'ViT-B/32'. Example:

```python es_engine --clip-model RN50x16```
