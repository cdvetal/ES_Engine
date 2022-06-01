# ES_Engine

python declarative_engine.py --save-folder meuteste --target-class birdhouse --networks standard6 --num-lines 2 --num-cols 13 --pop-size 100 --max-gens 1000 --target-fit 1000000 --renderer organic --target-class birdhouse --tournament-size 5 --random-seed 6 --cxpb 0.6 --mutpb 1 --mut-mu 0 --mut-indpb 1 --mut-sigma 0.02


CMA-ES


https://github.com/DEAP/deap/tree/master/deap


python cma-es-engine.py --random-seed 2 --max-gens 100 --pop-size 10 --save-folder cma-es-local10 --save-all --lamarck

result = big_sleep_cma_es.evaluate_with_local_search(conditional_vector,10) <-- 10 passos do adam