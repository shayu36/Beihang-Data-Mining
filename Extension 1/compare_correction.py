"""Reproduce the exploratory thyroid candidate and the established comparison.

The correction was selected after comparing 96 configurations on the course
test data. It is a development result, not an independent blind evaluation.
Only normal training rows are used to fit transformations and thresholds.
"""
import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import PowerTransformer, RobustScaler
from threadpoolctl import threadpool_limits

from course_data import overlap_with_training, split_normal, thyroid_features
from unified import conformal_threshold

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thyroid', type=Path, default=HERE.parent.parent / 'thyroid.zip')
    parser.add_argument('--output', type=Path, required=True, help='New directory outside 选做一')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(HERE.parent) or output.exists():
        parser.error('Choose a new output directory outside 选做一')
    train, test = thyroid_features(args.thyroid)
    fit, cal, _ = split_normal(train, 42, .2, deduplicate=True)
    with threadpool_limits(limits=4):
        power = PowerTransformer(standardize=False).fit(train[fit][:, [1, 2, 5]])
        scaler = RobustScaler(quantile_range=(10., 90.)).fit(power.transform(train[fit][:, [1, 2, 5]]))
        def scores(x):
            z = scaler.transform(power.transform(x[:, [1, 2, 5]]))
            base = np.maximum(z[:, 0], 0.) * np.maximum(-z[:, 2], 0.)
            return base, base / (1. + np.maximum(z[:, 1], 0.))
        cal_scores, test_scores = scores(train[cal]), scores(test)
    limits = [conformal_threshold(values, .005) for values in cal_scores]
    predictions = [values > limit for values, limit in zip(test_scores, limits)]
    # Labels are read only after the models, thresholds and predictions exist.
    with ZipFile(args.thyroid) as archive:
        reader = csv.DictReader(io.StringIO(archive.read('thyroid/test-set.csv').decode('utf-8-sig')))
        y = np.asarray([int(float(row['label'])) for row in reader])
    overlap = overlap_with_training(train, test)
    rows, results = [], []
    for name, score, pred, limit in zip(('original', 'corrected'), test_scores, predictions, limits):
        for scope, mask in (('official', np.ones(len(test), dtype=bool)), ('no_train_overlap', ~overlap)):
            yy, ss, pp = y[mask], score[mask], pred[mask]
            tn, fp, fn, tp = map(int, confusion_matrix(yy, pp, labels=[0, 1]).ravel())
            results.append(dict(method=name, scope=scope, threshold=float(limit), tn=tn, fp=fp, fn=fn, tp=tp,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn), auroc=float(roc_auc_score(yy, ss)),
                average_precision=float(average_precision_score(yy, ss))))
        rows.extend(dict(method=name, sample=f'test_row_{i+1:04}', score=float(s), threshold=float(limit),
                         prediction=int(p), label=int(y[i]), overlaps_train=int(overlap[i]))
                    for i, (s, p) in enumerate(zip(score, pred)))
    output.mkdir(parents=True)
    (output/'comparison.json').write_text(json.dumps(dict(seed=42, alpha=.005,
        development_selection='correction selected using the course test among 96 candidate configurations',
        test_labels_used_to_fit=False, test_is_independent=False, results=results),
        ensure_ascii=False, indent=2), encoding='utf-8')
    with (output/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for r in results:
        print(f"{r['method']:10} {r['scope']:16} F1={r['f1']:.6f} FP={r['fp']} FN={r['fn']}")
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
