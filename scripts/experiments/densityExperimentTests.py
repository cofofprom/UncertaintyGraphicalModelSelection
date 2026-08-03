from model import VectorGraphicalModel
from modelSelection import multipleTesting
from utils import matrix2Edges, evaluate
import argparse
import os
import json
import numpy as np
import pandas as pd
import uuid


CORRECTIONS = ('SI', 'B', 'H', 'BH', 'BY')
METRIC_COLUMNS = [
    'tn', 'fp', 'fn', 'tp', 'fdr', 'fomr', 'tpr', 'tnr', 'ba', 'f1', 'mcc',
    'correction',
]


def evaluateSingleModel(model, config):
    result = []
    true_edges = matrix2Edges(model.precision)

    for n_repl in range(config['S_obs']):
        data = model.rvs(config['n_samples'])

        for correction in CORRECTIONS:
            pred_edges = multipleTesting(data, config['alpha'], correction=correction)
            metrics = evaluate(true_edges, pred_edges)
            metrics.append(correction)
            result.append(metrics)

    return pd.DataFrame(result, columns=METRIC_COLUMNS)


def evaluateSetOfModels(config):
    result = []
    for d in np.linspace(0.1, 0.9, 10):
        for n_model in range(config['S_sg']):
            model = VectorGraphicalModel(config['dim'], d)

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

    result_df = evaluateSetOfModels(config)

    fname = f'{uuid.uuid4()}.csv'
    result_df.to_csv(os.path.join(args.experiment_dir, 'data', fname), index=False)


if __name__ == '__main__':
    main()
