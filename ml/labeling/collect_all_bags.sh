#!/bin/bash
# Collects front+back frames from all vehicle-recorded bags under data/.
# Run from the repo root: bash ml/labeling/collect_all_bags.sh
set -e

cd "$(dirname "$0")/../.."  # repo root

source /opt/ros/humble/setup.bash
source install/setup.bash

for bag in data/rosbag2_2026_07_23-*/; do
  session=$(basename "$bag")
  if compgen -G "ml/data/raw_frames/${session}_front_*" > /dev/null; then
    echo "=== $session: already collected, skipping ==="
    continue
  fi
  echo "=== $session ==="
  python3 ml/labeling/collect_frames.py --session-name "$session" --topic /camera/front /camera/back &
  COLLECTOR_PID=$!
  sleep 1
  ros2 bag play "$bag"
  kill $COLLECTOR_PID
  wait $COLLECTOR_PID 2>/dev/null
done

echo "Done. Frame counts:"
ls ml/data/raw_frames | grep -c _front_
ls ml/data/raw_frames | grep -c _back_
