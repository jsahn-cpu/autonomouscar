#!/bin/bash
# Collects front-camera-only frames (color) from all vehicle-recorded bags
# under data/. Rear camera is intentionally excluded (dropped from scope).
# Run from the repo root: bash ml/labeling/collect_all_bags.sh
set -e

cd "$(dirname "$0")/../.."  # repo root

source /opt/ros/humble/setup.bash
source install/setup.bash

for bag in data/rosbag2_2026_07_23-*/; do
  session=$(basename "$bag")
  if compgen -G "ml/data/raw_frames/${session}_*" > /dev/null; then
    echo "=== $session: already collected, skipping ==="
    continue
  fi
  echo "=== $session ==="
  python3 ml/labeling/collect_frames.py --session-name "$session" --topic /camera/front --every-n-frames 2 &
  COLLECTOR_PID=$!
  sleep 1
  # --rate 10: our sampling is message-count-based (--every-n-frames), not
  # wall-clock-based, and 29/30 messages are discarded before decode even
  # happens -- so playing back faster doesn't change which frames get
  # picked, just how long real-time playback takes to get there.
  ros2 bag play --rate 3 "$bag"
  kill $COLLECTOR_PID
  wait $COLLECTOR_PID 2>/dev/null
done

echo "Done. Frame count:"
ls ml/data/raw_frames | wc -l
