"""Read-only model replay and extra exports; does not alter the experiment algorithm."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument('--implementation', type=Path, required=True)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--weights', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.implementation))

import joblib
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
from course_data import CATEGORIES, ROOT, image_lists
from detector import score_detector
from image_features import Encoder, GRID


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


torch.set_num_threads(4)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
visuals = args.run / 'visuals'
visuals.mkdir(exist_ok=True)
with (args.run / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
boundary = []
for category in CATEGORIES:
    category_rows = [r for r in rows if r['category'] == category]
    normal = [r for r in category_rows if int(r['label']) == 0]
    anomaly = [r for r in category_rows if int(r['label']) == 1]
    boundary.extend([max(normal, key=lambda r: float(r['fusion_score'])),
                     min(anomaly, key=lambda r: float(r['fusion_score']))])

with threadpool_limits(limits=4), ZipFile(args.archive) as archive:
    encoder = Encoder(args.weights)
    for category in CATEGORIES:
        model = joblib.load(args.run / f'{category}.joblib')['detector']
        for record in [r for r in boundary if r['category'] == category]:
            member = record['path']
            stem = category + '_test' + Path(member).stem
            with Image.open(BytesIO(archive.read(ROOT + member))) as source:
                rgb = source.convert('RGB')
            rgb.save(visuals / f'{stem}_original.png')
            gx, patches = encoder.encode([rgb])
            scores, maps = score_detector(model, gx, patches, [0])
            np.save(visuals / f'{stem}_distance.npy', maps[0].reshape(GRID, GRID))
            replay = float(scores['fusion'][0])
            difference = abs(replay - float(record['fusion_score']))
            if difference > 1e-5:
                raise ValueError(f'Boundary replay mismatch: {member} ({difference})')
            record['single_image_fusion_score'] = replay
            record['replay_absolute_difference'] = difference
            record['original_png'] = f'{stem}_original.png'
            record['distance_npy'] = f'{stem}_distance.npy'
            print(f'[visuals] {member}: score={replay:.10f}, delta={difference:.3g}', flush=True)
    training, _ = image_lists(archive)
    matches = [name for name, label in training['hazelnut'] if label == 0 and Path(name).stem == '002']
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely locate hazelnut training good 002: {matches}')
    member = matches[0]
    with Image.open(BytesIO(archive.read(ROOT + member))) as source:
        training_rgb = source.convert('RGB')
    training_rgb.save(visuals / 'hazelnut_train_good002_original.png')
    feature_shapes = {}
    handles = []
    def save_response(name):
        def hook(module, inputs, output):
            response = output[0].mean(dim=0).detach().cpu().numpy()
            np.save(visuals / f'hazelnut_train_good002_{name}_mean.npy', response)
            feature_shapes[name] = {'full_shape': list(output.shape), 'saved_shape': list(response.shape)}
        return hook
    for layer in ('layer2', 'layer3', 'layer4'):
        handles.append(getattr(encoder.net, layer).register_forward_hook(save_response(layer)))
    encoder.encode([training_rgb])
    for handle in handles:
        handle.remove()
    write_json(visuals / 'layer_responses.json', {'member': member, 'aggregation': 'channel mean before local average-pooling',
               'input_size': [384, 384], 'layers': feature_shapes, 'weights_sha256': encoder.weights_sha256})
    write_json(visuals / 'boundary_samples.json', boundary)

parity = []
environment = os.environ.copy()
environment['PYTHONUTF8'] = '1'
for record in boundary:
    stem = record['category'] + '_test' + Path(record['path']).stem
    base = [sys.executable, str(args.implementation / 'run.py'), 'predict',
            '--model', str(args.run / (record['category'] + '.joblib')),
            '--weights', str(args.weights), '--threads', '4']
    outputs = {}
    for mode, extra in (
        ('zip', ['--archive', str(args.archive), '--member', record['path'], '--heatmap', str(visuals / f'{stem}_heatmap.png')]),
        ('disk', ['--image', str(visuals / f'{stem}_original.png')])
    ):
        command = base + extra
        completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=environment)
        (visuals / f'{stem}_{mode}_stdout.json').write_text(completed.stdout, encoding='utf-8')
        (visuals / f'{stem}_{mode}_stderr.log').write_text(completed.stderr, encoding='utf-8')
        if completed.returncode:
            raise RuntimeError(f'{mode} prediction failed: {completed.stderr}')
        outputs[mode] = json.loads(completed.stdout)
    if outputs['zip'] != outputs['disk']:
        raise ValueError(f'ZIP / disk entry mismatch: {record["path"]}')
    parity.append({'member': record['path'], 'zip': outputs['zip'], 'disk': outputs['disk'], 'exactly_equal': True})
    print(f'[predict] ZIP / disk exact match: {record["path"]}', flush=True)
write_json(visuals / 'prediction_entrypoint_parity.json', parity)
print(f'Visual and prediction evidence: {visuals}', flush=True)
