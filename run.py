#!/usr/bin/env python3
# 비정상 환경의 엣지 노드 연합 강화학습 실험.
#
# 엣지 노드 4개가 같은 제어 과제를 서로 다른 액추에이터 세기로 학습하고,
# 학습 도중 한 노드의 액추에이터가 약해진다(도메인 드리프트). 방법마다 노드끼리
# 무엇을 공유하는지(없음 / 전체 파라미터 / 실패 직전 상태의 prototype)가 다르고,
# 노드별 성능, 노드 간 중앙값·IQR, 드리프트 노드의 적응 속도, 통신량을 비교한다.
#
#   python run.py                       quick 설정, 콘솔 보고서 (CPU 로 방법당 수 분)
#   python run.py --preset full         논문 설정 (방법·seed 당 수 시간)
#   python run.py --all                 마크다운, JSON 을 out/ 에 저장
#   python run.py --methods local_per proto_per --seeds 0 1
#   python run.py -c config.json

import argparse
import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fedrl import report                                          # noqa: E402
from fedrl.train import Config, METHODS, run                      # noqa: E402

OUT = os.path.join(HERE, 'out')
DEF_CFG = os.path.join(HERE, 'config', 'experiment.json')


def make_config(exp, preset, method, seed, out_dir):
    p = dict(exp['presets'][preset])
    names = [n['name'] for n in exp['nodes']]
    noise = exp.get('noise_sigma', 0.1)
    if method == 'private_proto_per':
        out_dir = os.path.join(out_dir, 'ns%s' % noise)
    return Config(
        method=method, seed=seed, env_id=exp['env_id'],
        scales=tuple(n['actuator_scale'] for n in exp['nodes']),
        change_client=names.index(exp['drift']['node']),
        change_scale=exp['drift']['scale_after'],
        noise_sigma=noise, out_dir=out_dir, **p)


def save(path, body):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    return os.path.relpath(path, HERE)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description='엣지 노드 연합 강화학습 실험')
    ap.add_argument('-c', '--config', default=DEF_CFG)
    ap.add_argument('--preset', default='quick')
    ap.add_argument('--methods', nargs='+', choices=METHODS)
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--md', nargs='?', const=os.path.join(OUT, 'report.md'))
    ap.add_argument('--json', nargs='?', const=os.path.join(OUT, 'result.json'))
    ap.add_argument('--quiet', action='store_true', help='학습 로그를 숨긴다')
    args = ap.parse_args(argv)

    with open(args.config, encoding='utf-8') as f:
        exp = json.load(f)
    if args.preset not in exp['presets']:
        ap.error('preset 은 %s 중 하나' % ', '.join(exp['presets']))
    methods = args.methods or exp['methods']
    seeds = args.seeds if args.seeds is not None else exp['seeds']
    run_dir = os.path.join(OUT, 'runs', args.preset)

    t0 = time.time()
    runs = {}
    log = open(os.devnull, 'w') if args.quiet else sys.stderr
    for m in methods:
        for s in seeds:
            sys.stderr.write('  %s seed %d\n' % (m, s))
            cfg = make_config(exp, args.preset, m, s, run_dir)
            with contextlib.redirect_stdout(log):
                runs.setdefault(m, []).append(run(cfg))

    rep = report.summarize(exp, runs)
    meta = {'preset': args.preset, 'seeds': len(seeds),
            'elapsed': '%.0fs' % (time.time() - t0)}
    print(report.to_console(rep, meta))
    if args.all:
        args.md = args.md or os.path.join(OUT, 'report.md')
        args.json = args.json or os.path.join(OUT, 'result.json')
    if args.md:
        print('saved', save(args.md, report.to_markdown(rep, meta)))
    if args.json:
        print('saved', save(args.json, report.to_json(rep, meta)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
