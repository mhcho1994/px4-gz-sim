#!/usr/bin/env bash
python3 tools/scenario/scenario_generator.py planar_n_pts --outdir ./data/sitl_logs --runs 1 --start-run-id 0 \
--ardupilot-vehicle ArduCopter --ardupilot-frame gazebo-px4vision --ardupilot-model JSON --ardupilot-world px4vision_default \
--ardupilot-location Purdue --px4-vehicle 4001 --px4-frame gz_px4vision --px4-world default \
--px4-location Purdue --n 3 --edge-m 25 25 --vertex-deg random --vertex-deg-range 0 360 \
--speed-m-s random --speed-m-s-range 1 20 \
--alt-m random --alt-m-range 2 20

python3 tools/launcher/run_px4_gz_sitl.py --run-root ./data/sitl_logs --startup-delay 2 --max-run-s 300 --verbose --headless
python3 tools/launcher/run_ardupilot_gz_sitl.py --run-root ./data/sitl_logs --startup-delay 2 --max-run-s 300 --verbose --headless