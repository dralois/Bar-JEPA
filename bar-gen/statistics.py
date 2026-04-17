import argparse
import json
import os
import glob

import numpy as np


def analyze_charts(directory):
    y_ticks = []
    bar_counts = []
    aspect_ratios = []

    for file in glob.glob(os.path.join(directory, "*.json")):
        with open(file, 'r') as f:
            data = json.load(f)
            chart_type = data['chart_metadata']['type']
            size_bbox = data['chart_metadata']['size']['bbox']
            width = size_bbox[2] - size_bbox[0]
            height = size_bbox[3] - size_bbox[1]
            aspect_ratios.append(width / height)

            if chart_type == 'vertical':
                y_axis_ticks = data['data']['value_axis']['ticks']
                y_ticks.append(len(y_axis_ticks))

            total_bars = sum(len(feature['data']) for feature in data['data']['features'])
            bar_counts.append(total_bars)

    return {
        'bar_counts': bar_counts,
        'y_axis_ticks': y_ticks,
        'aspect_ratios': aspect_ratios,
    }


def summarize(results):
    percentiles = [1, 5, 10, 50, 75, 90, 95, 99]
    for key, values in results.items():
        if not values:
            print(f"No data for {key}\n")
            continue
        arr = np.array(values)
        print(f"{key.replace('_', ' ')} (n={len(arr)})")
        print(f"  min={arr.min():.2f}  max={arr.max():.2f}  mean={arr.mean():.2f}")
        for p in percentiles:
            print(f"  p{p:2d} = {np.percentile(arr, p):.4f}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    summarize(analyze_charts(args.directory))