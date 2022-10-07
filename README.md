# ES Engine

This tool allows the use of a CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm and Adam to evolve renderings to approximate a certain class of the imagenet dataset. It relies on the use of pre-trained classifiers to guide the optimization process by calculating the reward value based on the certainty of these classifiers on the desired class. 
It is also possible to use the CLIP to guide the optimization process by calculating the cosine similarity between the encodings of the desired prompts and the encodings of the generated renderings. 


## Rendering system
Each image is generated using a "renderer" algorithm located in the render subdirectory. Each renderer is just a module that contains a set of functions that can be used to generate the images.
- **generate_individual** - Function used for the initialization process. It returns a flat numpy array containing the necessary values to render an image.
- **to_adam** - This function receives a numpy array containing an individual, created either by random initialization or by CMA-ES and converts it to a tensor with gradients to be used by an Adam optimizer.
- **get_individual** - This functions does the opposite of the **to_adam**. It receives the information passed to Adam, places it on CPU, detaches it and converts it to numpy array.
- **chunks** - Auxiliary function to perform all the reshapes necessary to make the flat numpy array with the required shape for the generation.
- **render** - Receives a tensor array and returns a tensor image.

### Current renderers
- **pylinhas** - Draws using a set of colored lines
- **chars** - Draws using characters
- **organic** - Draws using organic shapes
- **thinorg** - Similar to organic but with thinner lines
- **biggan** - It uses a pretrained BigGAN with some modifications.
- **vqgan** - Uses a pretrained VQGAN to render the images.
- **clipdraw** - Draws using a set of differentiable colored lines.
- **linedraw** - Draws using a set of differentiable thin black lines.
- **fftdraw** - Creates an images using Fourier Transforms.
- **fastpixel** - Creates a small images which is then scaled to the desired size.
- **pixeldraw** - Draws using a set of differentiable colored squares.
- **vdiff** - Creates images using a pretrained diffusion model.


## Scoring system
The scoring system is based on the use of pre-trained classifiers. Each classifier is located in the classifiers subdirectory and can be tested using the test_classifier.py script. Each classifier contains three important features, a preprocessing function, a scoring function and a target size. The first one is a function that applies the required preprocess to the input image. The second one is a function that receives the preprocessed image and returns a list of floats. Lastly, the target_size is a set of two values that specify the size of their input. The output of the predict function is a list of values which represent the confidence of the classifier on the input image being each of the Image Net classes. These values are then used as fitness to guide the evolution process and are ranged from 0.0 to 1.0. Multiple classifiers can be used where the reward value is calculated based on the sum of the scores for each classifier.

## Setup
It requires a python version higher than 3.7 and a cudatoolbox version higher than 9.0.
- It might be necessary additional steps before installing the requirements:
  - Install pycairo dependencies: ```sudo apt install libcairo2-dev pkg-config python3-dev```
  - Install wheel package: ```pip install wheel```
  - Install gcc: ```sudo apt-get install gcc```

- If pycairo fails to install (ERROR: Failed building wheel for pycairo) use:
  - ```sudo apt install libcairo2-dev pkg-config python3-dev```
- If pycairo installs but fails to execute (undefined symbol: cairo_svg_surface_set_document_unit) use:
  - ```pip install pycairo==1.11.0```

- External packages that must be installed:
  - ```pip install git+https://github.com/eps696/aphantasia.git```
  - ```pip install git+https://github.com/openai/CLIP.git```

- How to install pytorch_wavelets:
  - ```git clone https://github.com/fbcotter/pytorch_wavelets```
  - ```cd pytorch_wavelets```
  - ```pip install .```

- How to install pydiffvg:
  - ```git clone https://github.com/lmagoncalo/diffvg```
  - ```cd diffvg```
  - ```git submodule update --init --recursive```
  - Verify that all these packages are installed:
    - ```conda install -y scikit-image```
    - ```conda install -y -c anaconda cmake```
    - ```conda install -y -c conda-forge ffmpeg```
    - ```pip install svgwrite```
    - ```pip install svgpathtools```
    - ```pip install cssutils```
    - ```pip install numba```
    - ```pip install torch-tools```
    - ```pip install visdom```
  - Then do ```python setup.py install```


## Usage
This tool is able to work without any arguments except for the target, either clip prompts or input image. A configuration file contains all the required information. However, using the command line arguments it is possible to change these values.

Examples:

```python main.py --clip-prompts "darth vader"```

```python main.py --img-size 512 --clip-prompts "a painting of superman by van gogh" --renderer pixeldraw --lr 0.03```

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

- [ ] Correct BigGAN problem related to loss of quality in last generation.


