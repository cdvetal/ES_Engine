# ES Engine

This tool allows the use of a CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm to evolve renderings to approximate a certain class of the imagenet dataset. It relies on the use of pre-trained classifiers to guide the evolutionary process by calculating the reward value based on the certainty of these classifiers on the desired class. 
It is also possible to use the CLIP to guide the evolutionary process by calculating the cosine similarity between the encodings of the desired prompts and the encodings of the generated renderings. It can be use a combination of both methods by controlling the influence of the CLIP. 


## Rendering system
Each image is generated using a "renderer" algorithm. Each "renderer" is located in the render subdirectory and can be visualized using the test_renderer.py script which outputs a sample image using each of the available algorithms. Each renderer is just a module that contains a function called render which is called at each generation.

```
# ind: array of real vectors, img_size: size of the desired output image
def render(ind, img_size):
    ...
    return img
```

This function receives a numpy array which contains a list of vectors to be used by the renderer algorithm to generate an image. The function also receives the target size which indicates the size of the image to output. This image must be a PIL Image so all algorithms are similar and the communication between modules is simplified. Depending on the renderer, the size of the input vector may change. For example, in the “pylinhas”, each vector is a list of length 8 which is used to create a line. If we increase the number of vectors we increase the number of lines. However, other renderers may require a larger vector for each line. This information is stored in the genotype_size variable which defines the size of each vector. 

### Current renderers
- **pylinhas** - Draws using a set of colored lines
- **chars** - Draws using characters
- **organic** - Draws using organic shapes
- **thinorg** - Similar to organic but with thinner lines


## Scoring system
The scoring system is based on the use of pre-trained classifiers. Each classifier is located in the classifiers subdirectory and can be tested using the test_classifier.py script. Each classifier contains three important features, a preprocessing function, a scoring function and a target size. The first one is a function that applies the required preprocess to the input image. The second one is a function that receives the preprocessed image and returns a list of floats. Lastly, the target_size is a set of two values that specify the size of their input. The output of the predict function is a list of values which represent the confidence of the classifier on the input image being each of the Image Net classes. These values are then used as fitness to guide the evolution process and are ranged from 0.0 to 1.0. Multiple classifiers can be used where the reward value is calculated based on the sum of the scores for each classifier.

## Setup
- Create virtual environment: ```python3 -m venv env```
- Activate virtual environment: ```source env/bin/activate```
- Install dependencies: ```pip install -r requirements.txt```

## Usage
This tool is able to work without any arguments as they have a predefined configuration file where all the required information is stored. However, using the command line arguments it is possible to change these values.

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

### --renderer
Choose the renderer algorithm. Default is 'pylinhas'. Example:

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

```python es_engine --init-mu 0.65```

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

### --clip-prompts
CLIP prompts for the generation. Default is the target class. Example:

```python es_engine --clip-prompts "a dog eating cereal"```


## TODO

- [x] Pylinhas, chars, organic and thinorg renderers working
- [x] Add support for multiple classifiers
- [ ] Add support for VQGAN
- [ ] Add support to GPU and CPU
- [ ] Use image as input
- [ ] Add more renderers

