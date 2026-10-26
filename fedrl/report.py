# 엣지 노드 관점 보고서.
#
# train.run() 이 남긴 metrics 를 방법별로 모아 노드별 최종 return, 노드 간
# 중앙값/IQR, 드리프트 노드의 적응 지표, 통신량을 한 표로 정리한다.

import json

import numpy as np

METHOD_LABEL = {
    'local': 'Local only (uniform replay)',
    'local_per': 'Local only (PER)',
    'fedavg': 'FedAvg (full parameters)',
    'proto_per': 'Failure-prototype sharing',
    'private_proto_per': 'Failure-prototype sharing + noise',
    'noisy_per': 'Content-free control',
    'per_ere': 'Local ERE-PER (recency)',
}


def _med_iqr(vals):
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return float('nan'), float('nan')
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return float(q2), float(q3 - q1)


def _mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float('nan')


def summarize(exp, runs):
    """runs: {method: [metrics dict per seed]} -> report dict."""
    names = [n['name'] for n in exp['nodes']]
    rows = []
    for method, ms in runs.items():
        per_node = [_mean([m.get('C%d_final_return' % (i + 1)) for m in ms])
                    for i in range(len(names))]
        med, iqr = _med_iqr(per_node)
        t80 = [m['recovery_step_T80'] for m in ms]
        rows.append({
            'method': method,
            'label': METHOD_LABEL.get(method, method),
            'seeds': len(ms),
            'node_final_return': dict(zip(names, per_node)),
            'median_return': med,
            'iqr_return': iqr,
            'drift_pre_return': _mean([m['pre_change_return'] for m in ms]),
            'drift_post_auc': _mean([m['post_change_auc'] for m in ms]),
            'drift_recovered': sum(t is not None for t in t80),
            'drift_failures': _mean([m['post_change_failures_20k'] for m in ms]),
            'comm_bytes': _mean([m['comm_bytes'] for m in ms]),
        })
    base = next((r for r in rows if r['method'] == exp.get('baseline')), None)
    for r in rows:
        if base and base['drift_post_auc'] and not np.isnan(base['drift_post_auc']):
            r['auc_vs_baseline_pct'] = 100.0 * (r['drift_post_auc'] / base['drift_post_auc'] - 1)
        else:
            r['auc_vs_baseline_pct'] = float('nan')
    return {'nodes': exp['nodes'], 'drift': exp['drift'], 'env_id': exp['env_id'],
            'baseline': exp.get('baseline'), 'rows': rows}


def _bytes(b):
    if b is None or np.isnan(b):
        return '-'
    if b == 0:
        return '0'
    for unit in ('B', 'KB', 'MB', 'GB'):
        if b < 1024:
            return '%.1f %s' % (b, unit)
        b /= 1024
    return '%.1f TB' % b


def _f(x, fmt='%.0f'):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else fmt % x


def to_markdown(rep, meta=None):
    names = [n['name'] for n in rep['nodes']]
    L = ['# Edge FedRL report', '']
    L.append('Task `%s`, %d edge nodes. Drift: %s actuator scale -> %.2f midway.'
             % (rep['env_id'], len(names), rep['drift']['node'], rep['drift']['scale_after']))
    if meta:
        L.append('')
        L.append(', '.join('%s: %s' % kv for kv in meta.items()))
    L += ['', '## Node-level performance (final evaluation return)', '',
          '| Method | ' + ' | '.join(names) + ' | Median | IQR |',
          '|---|' + '---:|' * (len(names) + 2)]
    for r in rep['rows']:
        L.append('| %s | ' % r['label'] + ' | '.join(_f(r['node_final_return'][n]) for n in names)
                 + ' | %s | %s |' % (_f(r['median_return']), _f(r['iqr_return'])))
    L += ['', '## Adaptation of the drifting node (%s)' % rep['drift']['node'], '',
          '| Method | Return before drift | Post-drift AUC | vs %s | Recovered to 80%% | Failed episodes after drift | Communication |' % rep['baseline'],
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in rep['rows']:
        L.append('| %s | %s | %s | %s | %d/%d | %s | %s |' % (
            r['label'], _f(r['drift_pre_return']), _f(r['drift_post_auc']),
            _f(r['auc_vs_baseline_pct'], '%+.1f%%'), r['drift_recovered'], r['seeds'],
            _f(r['drift_failures'], '%.1f'), _bytes(r['comm_bytes'])))
    L += ['', 'Post-drift AUC is the sum of evaluation returns of the drifting node over the '
          'adaptation window. Communication is the total bytes exchanged between nodes and the '
          'server during the run.', '']
    return '\n'.join(L)


def to_console(rep, meta=None):
    return to_markdown(rep, meta)


def to_json(rep, meta=None):
    return json.dumps({'meta': meta or {}, **rep}, indent=2, ensure_ascii=False, default=float)
