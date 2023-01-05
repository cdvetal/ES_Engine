# IMG_SIZE = IMG_WIDTH, IMG_HEIGHT = (512, 512)  # ATTENTION!!!! Only square images now please.
IMG_SIZE = 224

EVOLUTION_TYPE = 'adam'

NUM_LINES = 20

SAVE_FOLDER = 'experiments'
CHECKPOINT_FREQ = 10

POP_SIZE = 3
N_GENS = 100
SIGMA = 0.01

LAMARCK = True
ADAM_STEPS = 0
# LR = 0.1
LR = 1.
# LR = 0.03

TARGET_FITNESS = 0.999
RANDOM_SEED = None

FROM_CHECKPOINT = None  # None or "Experiment_name.pkl""

NETWORKS = ""  # mobilenet,vgg16
CLIP_MODEL = "ViT-B/32"

TARGET_CLASS = "birdhouse"

SAVE_ALL = False
VERBOSE = False

RENDERER = "linedraw"

model_groups = {
    "london,": "xception,vgg16,vgg19,resnet50,resnet50v2,resnet101,resnet152,resnet101v2,resnet152v2,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,densenet121,densenet169,densenet201,nasnet,nasnetmobile,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb7,",
    "standard3,": "vgg16,vgg19,mobilenet,",
    "standard6,": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,",
    "standard9,": "standard6,inceptionresnetv2,nasnet,nasnetmobile,",
    "standard13,": "standard9,densenet121,densenet169,densenet201,mobilenetv2,",
    "standard18,": "standard13,resnet101,resnet152,resnet50v2,resnet101v2,resnet152v2,",
    "train1,": "vgg19,resnet50,inceptionv3,xception,",
    "standard,": "standard6,",
    "fantastic,": "inceptionv3,vgg16,xception,mobilenet,efficientnetb4,efficientnetb0",
    "all,": "standard18,",
}
