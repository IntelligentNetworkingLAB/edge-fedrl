import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from fedrl import report                       # noqa: E402
from fedrl.sac import SACAgent, fedavg_sync    # noqa: E402
from fedrl.train import Config, run            # noqa: E402

TINY = dict(total_steps=400, change_step=200, warmup_steps=100, eval_interval=200,
            eval_episodes=1, share_interval=200, buffer_size=1000, batch_size=32,
            hidden=(16, 16), adapt_window=200, fail_count_window=200)


class TestFedAvg(unittest.TestCase):
    def test_parameters_become_equal_mean(self):
        torch.manual_seed(0)
        agents = [SACAgent(3, 2, hidden=(8,)) for _ in range(3)]
        before = [a.actor.net[0].weight.detach().clone() for a in agents]
        fedavg_sync(agents)
        mean = torch.stack(before).mean(0)
        for a in agents:
            self.assertTrue(torch.allclose(a.actor.net[0].weight, mean))


class TestTrainLoop(unittest.TestCase):
    def test_methods_run_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            runs = {}
            for m in ('local_per', 'fedavg', 'proto_per'):
                cfg = Config(method=m, seed=0, out_dir=d, **TINY)
                with open(os.devnull, 'w') as null:
                    import contextlib
                    with contextlib.redirect_stdout(null):
                        runs[m] = [run(cfg)]
                self.assertTrue(os.path.exists(os.path.join(d, '%s_seed0' % m, 'metrics.json')))
            self.assertGreater(runs['fedavg'][0]['comm_bytes'], 0)
            self.assertEqual(runs['local_per'][0]['comm_bytes'], 0)

            with open(os.path.join(ROOT, 'config', 'experiment.json'), encoding='utf-8') as f:
                exp = json.load(f)
            rep = report.summarize(exp, runs)
            self.assertEqual(len(rep['rows']), 3)
            for r in rep['rows']:
                self.assertEqual(set(r['node_final_return']), {n['name'] for n in exp['nodes']})
                self.assertFalse(np.isnan(r['median_return']))
            md = report.to_markdown(rep)
            self.assertIn('EDGE-03', md)
            json.loads(report.to_json(rep))


if __name__ == '__main__':
    unittest.main()
