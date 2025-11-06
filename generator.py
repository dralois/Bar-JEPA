import os
import re
import json
import random
import string
import matplotlib.pyplot as plt
import numpy as np
import argparse
import uuid
import shutil

from jsonschema import validate, ValidationError

def generate_random_string(max_length: int = 10):
    """
    Generates a random alphanumerical string

    :param length: Length of the string, defaults to 10
    :return: Random string of length
    """
    letters = string.ascii_letters + string.digits + " "
    return ''.join(random.choice(letters) for _ in range(random.randint(1, max_length)))

def generate_random_samples(samples: int = 1):
    """
    Generates of samples of a random distribution type

    :param samples: How many samples to generate, defaults to 1
    :return: Generated samples
    """
    distribution_type = random.choice(['uniform', 'gamma', 'gaussian', 'exponential'])
    if distribution_type == 'uniform':
        values = np.random.uniform(1, 100, size=samples)
    elif distribution_type == 'gamma':
        shape, scale = random.uniform(0.5, 3), random.uniform(1, 10)
        values = np.random.gamma(shape, scale, size=samples)
    elif distribution_type == 'gaussian':
        mean, std = random.uniform(20, 80), random.uniform(5, 20)
        values = np.random.normal(mean, std, size=samples)
        values = np.clip(values, 1, 100)
    elif distribution_type == 'exponential':
        scale = random.uniform(1, 20)
        values = np.random.exponential(scale, size=samples)
        values = np.clip(values, 1, 100)
    return values

def generate_random_bar_chart(
    output_dir,
    num_charts: int = 5,
    max_bars: int = 10,
    clear_output: bool = False,
    validate_schema: bool = False
):
    """
    Generates random bar charts (horizontal & vertical), with json annotations

    :param output_dir: Where to put the charts
    :param num_charts: Number of charts to generate, defaults to 5
    :param max_bars: Maximum number of bars per chart, defaults to 64
    :param clear_output: Flag whether to clear the output directory, defaults to False
    :param validate_schema: Flag whether to validate the generated JSON against the schema, defaults to False
    """

    # Clear output directory if requested
    if clear_output and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create separate directories for images and JSON annotations
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Load the schema if validation is enabled
    schema = None
    if validate_schema:
        with open('format.json', 'r') as f:
            schema = json.load(f)

    for _ in range(num_charts):
        # Generate a unique UUID for the chart
        chart_uuid = str(uuid.uuid4())

        # Randomly choose between bar and barh
        chart_type = random.choice(['bar', 'barh'])

        # Randomly decide the number of categories per bar (1-5)
        num_series = random.randint(1, 5)
        num_categories = random.randint(2, max_bars)

        # Ensure the total number of bars does not exceed the configured limit
        max_series = min(num_series, max_bars // num_categories)
        if max_series < 1:
            max_series = 1
            num_categories = min(num_categories, max_bars)

        num_series = random.randint(1, max_series)

        # Generate random categories and values for each bar
        categories = [generate_random_string(15) for _ in range(num_categories)]
        features = [generate_random_string(15) for _ in range(num_series)]
        values = {feature: generate_random_samples(num_categories) for feature in features}
        colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_series)]

        # Random metadata
        chart_title = generate_random_string(25)
        y_axis_title = generate_random_string(25)
        x_axis_title = generate_random_string(25)
        legend_title = generate_random_string(25)

        fig_dpi = 100.0
        fig_size = (6.4, 4.8)
        bar_width = random.randint(1, 9) / 10.0

        # Setup plot
        fig, ax = plt.subplots()
        fig.set_dpi(fig_dpi)
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])
        x = np.arange(len(categories))
        width = bar_width / num_series

        # Plot bars
        bars = []
        for i, (feature, samples) in enumerate(values.items()):
            offset = width * i
            for bar in getattr(ax, chart_type)(x + offset, samples, width, label=feature, color=colors[i]):
                bars.append(bar)

        # Plot labels
        title = ax.set_title(chart_title)
        xlabel = ax.set_xlabel(x_axis_title if chart_type == "bar" else y_axis_title)
        ylabel = ax.set_ylabel(y_axis_title if chart_type == "bar" else x_axis_title)
        getattr(ax, f"set_{'x' if chart_type == 'bar' else 'y'}ticks")(x + width * (num_series - 1) / 2, categories)

        # Plot legend
        legend = ax.legend(title=legend_title, loc='upper left', ncols=num_series)

        # Save image
        fig.savefig(os.path.join(images_dir, f"{chart_uuid}.png"), format='png')
        fig_height = fig.bbox.ymax - fig.bbox.ymin

        # Get the bounding boxes for the axes
        xaxis = ax.get_xaxis()
        yaxis = ax.get_yaxis()

        def invert_bbox(bbox, height):
            return [
                int(bbox.x0),
                int(height - bbox.y0),
                int(bbox.x1),
                int(height - bbox.y1)
            ]

        # Get the bounding boxes for the static text elements
        chart_bboxes = {}
        for name, elem in {
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "xaxis": xaxis,
            "yaxis": yaxis,
            "legend": legend
        }.items():
            bbox = elem.get_tightbbox()
            chart_bboxes[name] = invert_bbox(bbox, fig_height)

        # Get the bounding boxes for the legend labels
        legend_bboxes = []
        for text, hint in zip(legend.get_texts(), legend.get_patches()):
            text_bbox = text.get_tightbbox()
            hint_bbox = hint.get_tightbbox()
            legend_bboxes.append([
                invert_bbox(text_bbox, fig_height),
                invert_bbox(hint_bbox, fig_height)
            ])

        # Get the bounding boxes for the category labels
        category_bboxes = []
        cat_ticks = ax.get_xticklabels() if chart_type == "bar" else ax.get_yticklabels()
        for cat_tick in cat_ticks:
            cat_bbox = cat_tick.get_tightbbox()
            category_bboxes.append(invert_bbox(cat_bbox, fig_height))

        # Bounding boxes for the bars
        bar_bboxes = []
        for bar in bars:
            bar_bboxes.append(bar.get_tightbbox())

        bar_width = int(bar_bboxes[0].x1 - bar_bboxes[0].x0 if chart_type == "bar" else bar_bboxes[0].y1 - bar_bboxes[0].y0)

        plt.close()

        # JSON annotation
        annotation = {
            "chart_metadata": {
                "title": {
                    "text": chart_title,
                    "bbox": chart_bboxes["title"]
                },
                "type": chart_type,
                "bar_width": bar_width,
                "legend": {
                    "entries": [
                        {
                            "feature": {
                                "text": attribute,
                                "bbox": legend_bboxes[i][0]
                            },
                            "patch": {
                                "bbox": legend_bboxes[i][1]
                            }
                        } for i, attribute in enumerate(features)
                    ],
                    "bbox": chart_bboxes["legend"]
                }
            },
            "data": {
                "categories": [
                    {
                        "text": category,
                        "bbox": category_bboxes[i]
                    } for i, category in enumerate(categories)
                ],
                "category_axis": {
                    "label": {
                        "text": x_axis_title if chart_type == 'bar' else y_axis_title,
                        "bbox": chart_bboxes["xlabel" if chart_type == 'bar' else "ylabel"]
                    },
                    "bbox": chart_bboxes["xaxis" if chart_type == 'bar' else "yaxis"]
                },
                "value_axis": {
                    "label": {
                        "text": y_axis_title if chart_type == 'bar' else x_axis_title,
                        "bbox": chart_bboxes["ylabel" if chart_type == 'bar' else "xlabel"]
                    },
                    "min_value": min(ax.get_ylim() if chart_type == 'bar' else ax.get_xlim()),
                    "max_value": max(ax.get_ylim() if chart_type == 'bar' else ax.get_xlim()),
                    "bbox": chart_bboxes["yaxis" if chart_type == 'bar' else "xaxis"]
                },
                "features": [
                    {
                        "feature": feature,
                        "data": [{
                                "category": category,
                                "value": value,
                                "bbox": invert_bbox(bar_bboxes[i * len(categories) + j], fig_height)
                            } for j, (category, value) in enumerate(zip(categories, values[feature]))
                        ]
                    } for i, feature in enumerate(features)
                ]
            }
        }

        # Validate the annotation against the schema if validation is enabled
        if validate_schema:
            try:
                validate(instance=annotation, schema=schema)
            except ValidationError as e:
                print(f"Validation error for chart {chart_uuid}: {e}")
                continue

        def bbox_one_line(json_str):
            # Regular expression to match bbox arrays
            bbox_pattern = re.compile(r'"bbox":\s*\[([^\]]+)\]', re.DOTALL)
            # Replace bbox arrays with a single line of coordinates
            return bbox_pattern.sub(lambda m: f'"bbox": [{",".join([x.strip() for x in m.group(1).split(",")])}]', json_str)

        # Save JSON
        json_path = os.path.join(annotations_dir, f"{chart_uuid}.json")
        with open(json_path, 'w') as f:
            json_str = json.dumps(annotation, indent=2)
            fixed_json_str = bbox_one_line(json_str)
            f.write(fixed_json_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random bar charts and save them as images along with JSON annotations.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the images and JSON files.')
    parser.add_argument('--num_charts', type=int, default=5, help='Number of random bar charts to generate.')
    parser.add_argument('--max_bars', type=int, default=64, help='Maximum number of bars per chart.')
    parser.add_argument('--clear_output', action='store_true', help='Clear the output directory before generation.')
    parser.add_argument('--validate', action='store_true', help='Validate the generated JSON annotations against the schema.')

    args = parser.parse_args()

    generate_random_bar_chart(
        args.output_dir,
        args.num_charts,
        args.max_bars,
        args.clear_output,
        args.validate
    )
