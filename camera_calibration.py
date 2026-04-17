import pyrealsense2 as rs

def test(calib_dev):
    new_calib, health = calib_dev.run_on_chip_calibration(json_content='', timeout_ms=5000)
    print("health factor = ", health)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
conf = pipeline.start(config)
calib_dev = rs.auto_calibrated_device(conf.get_device())

test(calib_dev)

