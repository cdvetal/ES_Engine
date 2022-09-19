from es_engine import setup_args
from render.biggan import BigGANRenderer

args = setup_args()

renderer = BigGANRenderer(args)
