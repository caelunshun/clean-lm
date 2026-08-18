#!/bin/bash
for cap in profiler-device profiler-context trace-device; do
    minor=$(awk -v c="$cap" '$1==c{print $2}' /proc/driver/nvidia-caps/sys-minors)
    nvidia-modprobe -f /proc/driver/nvidia/capabilities/$cap
    chmod a+r /dev/nvidia-caps/nvidia-cap$minor
    echo "DeviceFileModify: 0" > /proc/driver/nvidia/capabilities/"$cap"
done
