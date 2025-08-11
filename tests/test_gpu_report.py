import csv
import os
import unittest

from roofline import gpu_report, model

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "synthetic_bench.csv")


class GpuReportTest(unittest.TestCase):
    # A made-up device with round numbers so expected percentages are easy to check by hand.
    dev = model.Device("x", "test", peak_gflops=10_000.0, bw_gbs=100.0, source="test")

    def rows(self):
        with open(FIXTURE) as f:
            return list(csv.DictReader(f))

    def test_work_mapping(self):
        r = self.rows()
        self.assertEqual(gpu_report.work_for(r[0]).flops, 2e9)
        self.assertEqual(gpu_report.work_for(r[1]).bytes, 8e6)
        self.assertEqual(gpu_report.work_for(r[2]).bytes, 4 * (2 * 1000 * 10 * 100 + 2 * 10 * 100))

    def test_percent_of_roofline(self):
        r = self.rows()
        # GEMM: 2e9 flop / 1e13 flop/s = 200 us minimum (compute bound); measured 1000 us -> 20%.
        self.assertAlmostEqual(model.percent_of_roofline(self.dev, gpu_report.work_for(r[0]), 1e-3), 20.0)
        # softmax: 8e6 B / 1e11 B/s = 80 us minimum (memory bound); measured 80 us -> 100%.
        self.assertAlmostEqual(model.percent_of_roofline(self.dev, gpu_report.work_for(r[1]), 80e-6), 100.0)

    def test_report_renders_every_row(self):
        text = gpu_report.report(self.rows(), self.dev)
        self.assertEqual(text.count("\n| "), 3 + 1)  # header row + 3 data rows
        self.assertIn("20.0%", text)
        self.assertIn("splits=8", text)


if __name__ == "__main__":
    unittest.main()
