IMG_SIZE = IMG_WIDTH, IMG_HEIGHT = (512, 512)  # ATTENTION!!!! Only square images now please.

NUM_LINES = 17
NUM_COLS = 8

RENDER_IMAGE = None

SAVE_FOLDER = 'experiments'

POP_SIZE = 10
N_GENS = 2
ELITISM = 1
CXPB = 0.9
MUTPB = 0.1
# Parameters for Gaussian Mutation
MUT_MU = 0
MUT_SIGMA = 1
MUT_INDPB = 0.2

TOURNAMENT_SIZE = 5

TARGET_FITNESS = 0.999,
RANDOM_SEED = None

CHECKPOINT_FREQ = 10
FROM_CHECKPOINT = None  # None or "Experiment_name.pkl""

NETWORKS = "mobilenet,mobilenetv2"
ACTIVE_MODELS = None
ACTIVE_MODELS_QUANTITY = None

TARGET_CLASS = "goldfish"

IMAGENET_INDEXES = None

SAVE_ALL = False

RENDERER = "pylinhas"

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
