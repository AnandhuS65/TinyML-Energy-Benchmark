#!/usr/bin/env bash
set -e

echo "Locking CPU governor to performance..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

if [[ "$1" == "--radios-off" ]]; then
    echo "Disabling Wi-Fi and Bluetooth for low-noise energy measurement..."
    sudo rfkill block wifi bluetooth
fi

echo "Current CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "Current CPU temp:     $(vcgencmd measure_temp)"
