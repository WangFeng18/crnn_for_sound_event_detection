import runner
from runner import Config

seg_tests = [5., 10., 40.]
for seg_test in seg_tests:
    config = Config(segment_length=seg_test)
    runner.firefunc(config)

