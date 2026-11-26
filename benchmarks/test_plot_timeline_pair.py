"""The timeline pair plot keeps every step's critical rank and every rank, from synthetic Nsight exports."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from benchmarks import plot_timeline_pair as plot
from runtime import combined_timeline as timeline
from runtime.test_combined_timeline import export


def summary(tmp, name, **kwargs):
    return timeline.analyze(export(Path(tmp) / f"{name}.sqlite", **kwargs), layers=(0, 1, 2, 3), tp_size=2)


class TimelinePair(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.control = summary(self.tmp, "stock", offload=(0, 1, 2, 3))
        self.candidate = summary(self.tmp, "t40", slow=(1, 1, 50), waits={(0, 1): 30})

    def test_parts_split_the_critical_rank_layers_and_keep_all_ranks(self):
        data = plot.pair_data(self.control, self.candidate, ("Stock", "T2"))
        stock, fixed = data["arms"]
        self.assertEqual((data["swapped"], fixed["critical"], fixed["tp_wait_us"][0]), ([1, 3], [1] * 5, [30, 0]))
        rows = {row["layer"]: row for row in self.candidate["critical"]["layers"]}
        for index, layer in enumerate(fixed["layers"]):
            self.assertEqual(sum(fixed["parts"][part][index] for part in plot.PARTS), rows[layer]["layer_us"])
        self.assertEqual(fixed["parts"]["swap"][1], 50, "the critical rank's own slow swap")
        self.assertEqual(len(stock["whole_us"][0]), 2, "every rank of every step")

    def test_mismatched_or_unattributed_timelines_are_refused(self):
        with self.assertRaisesRegex(ValueError, "same layers"):
            plot.pair_data(self.control, timeline.analyze(export(Path(self.tmp) / "l3.sqlite", layers=(0, 1, 2),
                                                                 offload=(1,)), layers=(0, 1, 2), tp_size=2))
        with self.assertRaisesRegex(ValueError, "attribute TP waits"):
            plot.pair_data(self.control, summary(self.tmp, "unpaired", missing=(1, 2)))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib draws the figure")
    def test_the_figure_is_written_once(self):
        output = Path(self.tmp) / "pair.png"
        plot.draw(plot.pair_data(self.control, self.candidate), output)
        self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
