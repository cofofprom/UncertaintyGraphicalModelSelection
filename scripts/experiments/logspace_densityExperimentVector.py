from model import TensorGraphicalModel
from modelSelection import tlasso
from utils import matrix2Edges, evaluate
import argparse
import os
import json
import numpy as np
import pandas as pd
import uuid
import pickle


def evaluateSingleModel(model, config):
    result = []
    
    for n_repl in range(config['S_obs']):
        data = model.rvs(config['n_samples'])
        
        d = model.densities[0]
        reg_param = config['reg_param']
        pred_omegas = tlasso(data, reg_param)

        for k in range(model.order):
            true_edges = matrix2Edges(model.precisions[k])
            pred_edges = matrix2Edges(pred_omegas[k])

            metrics = evaluate(true_edges, pred_edges)
            metrics.append(k)
            
            result.append(metrics)

    return pd.DataFrame(result, columns=['tn', 'fp', 'fn', 'tp', 'fdr', 'fomr', 'tpr', 'tnr', 'ba', 'f1', 'mcc', 'way'])

def evaluateSetOfModels(config):
    result = []
    for d in np.logspace(-3, -1, 30):
        for n_model in range(config['S_sg']):
            model = TensorGraphicalModel(config['dims'], [d for _ in range(config['order'])])

            single_results = evaluateSingleModel(model, config)
            single_results['d'] = d

            result.append(single_results)

    return pd.concat(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str, action='store')

    args = parser.parse_args()

    os.makedirs(args.experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(args.experiment_dir, 'data'), exist_ok=True)

    with open(os.path.join(args.experiment_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    with open(os.path.join(args.experiment_dir, 'interp.pickle'), 'rb') as f:
        reg_param_interp = pickle.load(f)
        config['reg_param_interp'] = reg_param_interp
        
    result_df = evaluateSetOfModels(config)

    fname = f'{uuid.uuid4()}.csv'
    result_df.to_csv(os.path.join(args.experiment_dir, 'data', fname), index=False)

if __name__ == '__main__':
    main()
