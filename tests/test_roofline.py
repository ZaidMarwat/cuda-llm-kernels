import math
import unittest

from roofline import model


class RooflineTest(unittest.TestCase):
    dev = model.Device("x", "test device", peak_gflops=1000.0, bw_gbs=100.0, source="")

    def test_ridge_point(self):
        self.assertEqual(self.dev.ridge, 10.0)

    def test_square_gemm_intensity_is_n_over_6(self):
        # 2n^3 flops / (3 n^2 * 4 bytes) = n/6
        for n in (64, 512, 4096):
            self.assertAlmostEqual(model.gemm(n, n, n).intensity, n / 6)

    def test_decode_gemm_is_memory_bound(self):
        # Batch-1 decode GEMM is a matrix-vector product: AI ~ 0.5 flop/B, well below any ridge.
        w = model.gemm(1, 768, 768)
        self.assertLess(w.intensity, 0.6)
        self.assertEqual(model.bound(self.dev, w)["bound"], "memory")

    def test_large_gemm_is_compute_bound(self):
        self.assertEqual(model.bound(self.dev, model.gemm(2048, 2048, 2048))["bound"], "compute")

    def test_attention_decode_intensity_near_half(self):
        # 4THD flops over 8THD bytes, plus small terms
        self.assertTrue(0.5 <= model.attention_decode(4096, 12, 64).intensity < 0.52)

    def test_bound_time_and_attainable(self):
        w = model.Work("k", "", flops=2e9, bytes=1e8)  # AI 20 > ridge 10: compute bound
        b = model.bound(self.dev, w)
        self.assertTrue(math.isclose(b["min_time_s"], 2e-3))
        self.assertTrue(math.isclose(b["attainable_gflops"], 1000.0))
        self.assertTrue(math.isclose(model.percent_of_roofline(self.dev, w, 4e-3), 50.0))

    def test_devices_file_loads(self):
        devs = model.load_devices()
        self.assertIn("t4", devs)
        for d in devs.values():
            self.assertGreater(d.peak_gflops, 0)
            self.assertGreater(d.bw_gbs, 0)


if __name__ == "__main__":
    unittest.main()
