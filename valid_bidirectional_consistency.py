"""Audit validity/response consistency on flawed and original NL2SQL inputs.

Read-only local analysis; no model API calls. Judge labels measure whether a
response recognizes a flaw, not whether the input is actually flawed.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from utils import get_invalid_evaluation_metrics

ROOT = Path(__file__).resolve().parent
DEFAULT_JUDGES = ('claude_claude-opus-4-6', 'openai_gpt-5.4')
METRICS = ('sql_behavior', 'detection', 'type_identification')


def read_records(path, issues):
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, list):
            raise ValueError('expected a JSON list')
        return data
    except (OSError, ValueError) as exc:
        issues.append({'path': str(path), 'problem': str(exc)})
        return []


def indexed_records(records):
    """Align repeated questions by occurrence within a result file."""
    occurrences = Counter()
    indexed = {}
    malformed = 0
    for record in records:
        if not isinstance(record, dict):
            malformed += 1
            continue
        db, question = record.get('db_id'), record.get('question')
        if not isinstance(db, str) or not isinstance(question, str):
            malformed += 1
            continue
        identity = (db, question)
        key = (*identity, occurrences[identity])
        occurrences[identity] += 1
        indexed[key] = record
    return indexed, malformed


def sql_generated(record):
    """Return the SQL-generation decision, or None for malformed SQL fields."""
    if 'sql' not in record:
        return None
    value = record['sql']
    if value is None:
        return False
    if not isinstance(value, str):
        return None
    if value.strip().casefold() in {'', 'null', 'none'}:
        return False
    if value.lstrip().startswith('{'):
        try:
            decoded = json.loads(value.replace("'", '"'))
            if isinstance(decoded, dict) and 'type' in decoded:
                return None
        except ValueError:
            pass
    return True


def rate(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None}


class Summary:
    def __init__(self, judges):
        self.judges = judges
        self.counts = Counter()
        self.by_judge = defaultdict(Counter)

    def positive(self, record, evaluations):
        c = self.counts
        c['positive_raw_records'] += 1
        flag = record.get('is_valid')
        generated = sql_generated(record)
        if type(flag) is not bool:
            c['positive_nonboolean_is_valid'] += 1
        if generated is None:
            c['positive_invalid_sql_field'] += 1
        complete = all(judge in evaluations for judge in self.judges)
        if not complete:
            c['positive_incomplete_judge_coverage'] += 1
        labels_ok = complete and all(
            not get_invalid_evaluation_metrics(evaluations[j]) for j in self.judges
        )
        if complete and not labels_ok:
            c['positive_invalid_judge_metrics'] += 1
        if labels_ok:
            c['positive_metric_records'] += 1
            for metric in METRICS:
                c['sum_' + metric] += sum(evaluations[j][metric] for j in self.judges) / len(self.judges)
            c['positive_joint_detection_records'] += int(all(evaluations[j]['detection'] == 1 for j in self.judges))
            c['positive_joint_detection_and_type_records'] += int(all(
                evaluations[j]['detection'] == 1 and evaluations[j]['type_identification'] == 1 for j in self.judges))
        if type(flag) is not bool or generated is None:
            return
        c['positive_prediction_records'] += 1
        side = 'valid' if flag else 'invalid'
        c['positive_' + side] += 1
        c['positive_' + side + '_generation'] += int(generated)
        if not labels_ok:
            return
        c['positive_consistency_records'] += 1
        c['positive_' + side + '_evaluated'] += 1
        for judge in self.judges:
            r = evaluations[judge]
            det_match = int(r['detection'] == int(not flag))
            abstain = int(r['sql_behavior'] == 2)
            c['positive_' + side + '_detection_match'] += det_match / len(self.judges)
            c['positive_' + side + '_complete_abstention'] += abstain / len(self.judges)
            jc = self.by_judge[judge]
            jc[side + '_total'] += 1
            jc[side + '_detection_match'] += det_match
            jc[side + '_complete_abstention'] += abstain
        c['positive_' + side + '_detection_match_unanimous'] += int(all(
            evaluations[j]['detection'] == int(not flag) for j in self.judges))
        c['positive_' + side + '_complete_abstention_unanimous'] += int(all(
            evaluations[j]['sql_behavior'] == 2 for j in self.judges))

    def negative(self, record):
        c = self.counts
        c['negative_raw_records'] += 1
        flag = record.get('is_valid')
        generated = sql_generated(record)
        if type(flag) is not bool:
            c['negative_nonboolean_is_valid'] += 1
        if generated is None:
            c['negative_invalid_sql_field'] += 1
        if type(flag) is not bool or generated is None:
            return
        c['negative_prediction_records'] += 1
        side = 'valid' if flag else 'invalid'
        c['negative_' + side] += 1
        c['negative_' + side + '_generation'] += int(generated)

    def export(self):
        c = self.counts
        output = {'counts': dict(c), 'positive_metrics': {}, 'conditional_rates': {},
                  'validity_accuracy': {}, 'negative_metrics': {}, 'per_judge': {}}
        n = c['positive_metric_records']
        for metric in METRICS:
            scale = 2 if metric == 'sql_behavior' else 1
            output['positive_metrics'][metric] = rate(c['sum_' + metric] / scale, n)
        positive_n = c['positive_prediction_records']
        negative_n = c['negative_prediction_records']
        output['validity_accuracy']['positive'] = rate(c['positive_invalid'], positive_n)
        output['validity_accuracy']['negative'] = rate(c['negative_valid'], negative_n)
        total = positive_n + negative_n
        output['validity_accuracy']['pooled'] = rate(c['positive_invalid'] + c['negative_valid'], total)
        output['validity_accuracy']['ground_truth_flawed_given_reported_invalid'] = rate(
            c['positive_invalid'], c['positive_invalid'] + c['negative_invalid'])
        output['validity_accuracy']['ground_truth_valid_given_reported_valid'] = rate(
            c['negative_valid'], c['negative_valid'] + c['positive_valid'])
        output['validity_confusion_matrix'] = {
            'rows': ['ground_truth_flawed', 'ground_truth_valid'],
            'columns': ['reported_invalid', 'reported_valid'],
            'counts': [[c['positive_invalid'], c['positive_valid']],
                       [c['negative_invalid'], c['negative_valid']]]
        }
        output['excluded'] = {
            'positive_prediction_records': c['positive_raw_records'] - positive_n,
            'negative_prediction_records': c['negative_raw_records'] - negative_n,
            'positive_consistency_records': c['positive_raw_records'] - c['positive_consistency_records'],
        }
        pos_acc = output['validity_accuracy']['positive']['rate']
        neg_acc = output['validity_accuracy']['negative']['rate']
        output['validity_accuracy']['balanced'] = (pos_acc + neg_acc) / 2 if pos_acc is not None and neg_acc is not None else None
        output['negative_metrics']['GEN'] = rate(c['negative_valid_generation'] + c['negative_invalid_generation'], negative_n)
        output['negative_metrics']['VAL'] = rate(c['negative_valid'], negative_n)
        for side in ('valid', 'invalid'):
            pden = c['positive_' + side + '_evaluated']
            for metric in ('detection_match', 'complete_abstention'):
                name = 'positive_' + side + '_' + metric
                output['conditional_rates'][name] = rate(c[name], pden)
                output['conditional_rates'][name + '_unanimous'] = rate(c[name + '_unanimous'], pden)
            for split in ('positive', 'negative'):
                den = c[split + '_' + side]
                gen = c[split + '_' + side + '_generation']
                output['conditional_rates'][split + '_' + side + '_generation'] = rate(gen, den)
                output['conditional_rates'][split + '_' + side + '_no_sql'] = rate(den - gen, den)
        output['conditional_rates']['type_given_joint_detection'] = rate(
            c['positive_joint_detection_and_type_records'], c['positive_joint_detection_records'])
        for judge, jc in sorted(self.by_judge.items()):
            output['per_judge'][judge] = {
                side + '_' + metric: rate(jc[side + '_' + metric], jc[side + '_total'])
                for side in ('valid', 'invalid')
                for metric in ('detection_match', 'complete_abstention')}
        return output


def analyze(project_root=ROOT, datasets=('bird', 'spider'), modes=('implicit', 'explicit'),
            judges=DEFAULT_JUDGES, target_models=None):
    """Each generation record contributes once; judge-dependent rates are averaged."""
    project_root = Path(project_root)
    judges = tuple(judges)
    if not judges or len(set(judges)) != len(judges):
        raise ValueError('Specify at least one judge and no duplicate judges')
    if len(set(datasets)) != len(datasets) or len(set(modes)) != len(modes):
        raise ValueError('Repeated datasets/settings would duplicate generation records')
    buckets = defaultdict(lambda: Summary(judges))
    issues = []
    coverage = Counter()
    file_counts = defaultdict(dict)
    models_by_dataset = {}
    targets = {m.replace('/', '_') for m in target_models} if target_models else None
    for dataset in datasets:
        positive_root = project_root / 'results' / dataset
        negative_root = project_root / 'validity' / dataset
        if not positive_root.is_dir() or not negative_root.is_dir():
            raise FileNotFoundError(f'Missing results/validity root for {dataset}')
        models = sorted(p.name for p in positive_root.iterdir() if p.is_dir() and (targets is None or p.name in targets))
        if not models or (targets and targets - set(models)):
            raise ValueError(f'Requested model directories missing in {dataset}')
        models_by_dataset[dataset] = models
        for model in models:
            for mode in modes:
                group_buckets = [buckets['overall'], buckets['mode:' + mode],
                                 buckets['dataset:' + dataset], buckets['model:' + model]]
                pos_paths = sorted((positive_root / model / mode).glob('final_*.json'))
                neg_paths = sorted((negative_root / model / mode).glob('*_valid.json'))
                for split, paths in (('positive', pos_paths), ('negative', neg_paths)):
                    if not paths:
                        issues.append({'path': str((positive_root if split == 'positive' else negative_root) / model / mode), 'problem': 'no raw result files'})
                for path in pos_paths:
                    coverage['positive_files'] += 1
                    raw, malformed = indexed_records(read_records(path, issues))
                    file_counts[(dataset, model, mode, path.stem.removeprefix('final_'))]['positive_records'] = len(raw) + malformed
                    coverage['positive_malformed_identities'] += malformed
                    evaluations = {}
                    for judge in judges:
                        ep = path.parent / 'eval' / judge / path.name
                        coverage['expected_judge_files'] += 1
                        eraw, malformed = indexed_records(read_records(ep, issues))
                        coverage['judge_malformed_identities'] += malformed
                        coverage['judge_records_without_raw_record'] += len(set(eraw) - set(raw))
                        coverage['raw_records_without_judge_record'] += len(set(raw) - set(eraw))
                        current = {}
                        for key in set(raw) & set(eraw):
                            if any(raw[key].get(f) != eraw[key].get(f) for f in ('sql', 'reason', 'is_valid')):
                                coverage['stale_judge_records'] += 1
                                issues.append({'path': str(ep), 'problem': 'judge copy differs from current generation', 'db_id': key[0], 'question': key[1], 'occurrence': key[2]})
                            else:
                                current[key] = eraw[key]
                        evaluations[judge] = current
                    for key, record in raw.items():
                        labels = {j: evaluations[j][key] for j in judges if key in evaluations[j]}
                        for bucket in group_buckets:
                            bucket.positive(record, labels)
                for path in neg_paths:
                    coverage['negative_files'] += 1
                    raw, malformed = indexed_records(read_records(path, issues))
                    category = path.stem.removeprefix(dataset + '_').removesuffix('_valid')
                    file_counts[(dataset, model, mode, category)]['negative_records'] = len(raw) + malformed
                    coverage['negative_malformed_identities'] += malformed
                    for record in raw.values():
                        for bucket in group_buckets:
                            bucket.negative(record)
    combined_counts = defaultdict(Counter)
    for (_, model, mode, category), values in file_counts.items():
        combined_counts[(model, mode, category)].update(values)
    pool_totals = Counter()
    for values in file_counts.values():
        pool_totals.update(values)
    return {
        'definitions': {
            'unit': 'One generation record per dataset/model/setting/file/db/question/occurrence; judges are not extra model-query instances.',
            'positive': 'Taxonomy-perturbed flawed inputs in results; ground-truth is_valid=false.',
            'negative': 'Original valid inputs in validity; ground-truth is_valid=true.',
            'detection': 'Judge-assessed flaw recognition in the response, not input ground truth.',
            'detection_match': 'P(DET=1 | reported invalid) or P(DET=0 | reported valid), restricted to positive samples.',
            'judge_aggregation': 'Conditional numerators average binary judge labels per generation record; denominators count unique generation records with all selected judges usable. Separate per-judge and unanimous statistics are also provided.',
            'no_sql': "null/blank/string 'None' or 'null'; GEN measures output presence, not SQL execution correctness.",
            'eligibility': 'Boolean is_valid and structurally usable SQL field; judge consistency additionally requires complete, valid, current evaluator records. Unused reason content does not determine eligibility.',
            'type_given_joint_detection': 'Both TYPE=1 among records where every selected judge assigned DET=1; uses a different denominator from mean scores.',
            'zero_denominator': 'null (undefined), never silently zero.'},
        'scope': {'project_root': str(project_root.resolve()), 'datasets': list(datasets),
                  'modes': list(modes), 'judges': list(judges), 'models_by_dataset': models_by_dataset},
        'coverage': dict(coverage), 'issues': issues,
        'sample_pool_totals': dict(pool_totals),
        'combined_dataset_category_count_mismatches': [
            {'model': key[0], 'mode': key[1], 'category': key[2], **values}
            for key, values in sorted(combined_counts.items())
            if values.get('positive_records', 0) != values.get('negative_records', 0)
        ],
        'sample_pool_count_mismatches': [
            {'dataset': key[0], 'model': key[1], 'mode': key[2], 'category': key[3], **values}
            for key, values in sorted(file_counts.items())
            if values.get('positive_records') != values.get('negative_records')
        ],
        'pairing_note': 'Judge alignment is verified within positive outputs. Dataset/category allocation differences do not imply unequal overall or combined-dataset category totals, missing outputs, or a broken source pairing. Record-level positive/negative source pairing is a separate check and is not verified by this script.',
        'overall': buckets['overall'].export(),
        'by_mode': {m: buckets['mode:' + m].export() for m in modes},
        'by_dataset': {d: buckets['dataset:' + d].export() for d in datasets},
        'by_model': {key[6:]: bucket.export() for key, bucket in sorted(buckets.items()) if key.startswith('model:')},
    }


def display_rate(label, values):
    value = values['rate']
    rendered = 'N/A' if value is None else f'{value:.2%}'
    print(f'  {label}: {rendered} ({values["numerator"]:g}/{values["denominator"]})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root; default is the script directory')
    parser.add_argument('--datasets', nargs='+', choices=['bird', 'spider'], default=['bird', 'spider'])
    parser.add_argument('--modes', nargs='+', choices=['implicit', 'explicit'], default=['implicit', 'explicit'])
    parser.add_argument('--judges', nargs='+', default=list(DEFAULT_JUDGES), help='Evaluator directory names')
    parser.add_argument('--target-models', nargs='+', help='Optional generation-model names')
    parser.add_argument('--report', type=Path, default=Path('output/valid_bidirectional_consistency_results.json'))
    args = parser.parse_args()
    result = analyze(args.root, args.datasets, args.modes, args.judges, args.target_models)
    output = args.report if args.report.is_absolute() else args.root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Unit: generation records; judge-dependent numerators use mean judge labels.')
    print('Detection agreement is response consistency, NOT validity correctness.')
    for mode, summary in [('overall', result['overall']), *result['by_mode'].items()]:
        c = summary['counts']
        print(f'\n[{mode}] positive={c.get("positive_raw_records", 0)}, negative={c.get("negative_raw_records", 0)}')
        print(f'  positive boolean/SQL usable={c.get("positive_prediction_records",0)}, judge-aligned={c.get("positive_consistency_records",0)}')
        for key in ('positive_invalid_detection_match', 'positive_valid_detection_match',
                    'positive_invalid_complete_abstention', 'negative_invalid_no_sql', 'negative_valid_generation'):
            display_rate(key, summary['conditional_rates'][key])
        for split in ('positive', 'negative', 'pooled',
                      'ground_truth_flawed_given_reported_invalid',
                      'ground_truth_valid_given_reported_valid'):
            display_rate('validity_accuracy_' + split, summary['validity_accuracy'][split])
        for metric, values in summary['positive_metrics'].items():
            score = values['rate']
            rendered = 'N/A' if score is None else f'{score:.4f}'
            print(f'  positive_{metric}: {rendered} (n={values["denominator"]})')
    print(f'\nCoverage: {json.dumps(result["coverage"])}')
    totals = result['sample_pool_totals']
    print(f'Overall raw sample pools: positive={totals.get("positive_records", 0)}, negative={totals.get("negative_records", 0)}')
    print(f'Combined-dataset category count differences: {len(result["combined_dataset_category_count_mismatches"])}')
    print(f'Dataset/category allocation differences: {len(result["sample_pool_count_mismatches"])}; these alone do not indicate missing outputs or unequal overall totals.')
    print(f'Issues: {len(result["issues"])}; complete report: {output.resolve()}')


if __name__ == '__main__':
    main()
