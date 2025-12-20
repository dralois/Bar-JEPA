# Adapted from:
# https://github.com/csuvis/BarchartReverseEngineering/blob/master/generate_random_bar_chart.py

import multiprocessing as mp

import os
import matplotlib

matplotlib.use("Agg")

import re
import json
import shutil
import random
import string
import itertools
import argparse

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager
from jsonschema import validate, ValidationError
from tqdm import tqdm

### random bar chart configuration ###
bar_direction_list = ["vertical"]
bar_per_loc_list = [1]  # how many bars are there in one ordinal position
bar_num_min = 5
bar_num_max = 5
bar_value_min = 10
bar_value_max = 200
bar_width_min = 0.4
bar_width_max = 1.0

axis_allowed = True
axis_label_length_min = 5
axis_label_length_max = 15
axis_label_size_min = 15
axis_label_size_max = 18

legend_allowed = False
legend_position = ["top", "right", "bottom"]
legend_length_min = 3
legend_length_max = 6
legend_size_min = 15
legend_size_max = 18

ticks_label_length_min = 1
ticks_label_length_max = 5
ticks_label_size_min = 14
ticks_label_size_max = 16

dpi_min = 100
dpi_max = 200
figsize_min = 6
figsize_max = 8

title_allowed = True
title_length_min = 5
title_length_max = 15
title_size_min = 18
title_size_max = 20
title_location = ["left", "center", "right"]


def test_latin_font(font):
    try:
        curr = font_manager.get_font(font)
        return curr.get_charmap().get(ord("a")) is not None
    except:
        return False


fonts_list = list(filter(test_latin_font, font_manager.findSystemFonts()))


styles = plt.style.available
if 'dark_background' in styles:
    styles.remove('dark_background')


def generate_random_samples(
    per_location,
    locations,
    allow_negative = False
):
    """
    Generates of samples of a random distribution type

    :param per_location: How many samples to generate per location (category)
    :param locations: For many locations (categories) to generate
    :param allow_negative: Whether to allow for negative values
    :return: Generated samples (per_location, locations)
    """
    sample_shape = (per_location, locations)
    distribution_type = random.choice(['uniform', 'gamma', 'gaussian', 'exponential'])

    if distribution_type == 'uniform':
        values = np.random.uniform(size=sample_shape)
    elif distribution_type == 'gamma':
        shape, scale = random.uniform(0.5, 3), random.uniform(1, 10)
        values = np.random.gamma(shape, scale, size=sample_shape)
    elif distribution_type == 'gaussian':
        mean, std = random.uniform(20, 80), random.uniform(5, 20)
        values = np.random.normal(mean, std, size=sample_shape)
    elif distribution_type == 'exponential':
        scale = random.uniform(1, 20)
        values = np.random.exponential(scale, size=sample_shape)

    # Normalize to [0, 1]
    mean = np.mean(values)
    norm = (values - mean) / np.linalg.norm(values - mean)

    return norm if allow_negative else np.abs(norm)


def get_random_plot(filename):
    """
    Random bar chart generation method.

    :param filename: Filename to save the plot to
    :return: Generated plot
    """

    # Outputs
    ax = None
    fig = None
    bars = []
    data = None
    title = None
    legend = None
    axis_ticks = None
    axis_label = None

    # plot style 
    style = random.choice(styles)
    plt.style.use(style)

    # resolution and figure size
    dpi = random.randint(dpi_min, dpi_max)
    figsize = [random.randint(figsize_min, figsize_max), random.randint(figsize_min, figsize_max)]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # bars setting
    bar_num = random.randint(bar_num_min, bar_num_max)
    bar_per_loc = random.choice(bar_per_loc_list)
    bar_direction = random.choice(bar_direction_list)
    bar_width = bar_width_min + (random.random() * (bar_width_max - bar_width_min))

    # generate random data according to bar_per_loc 
    bar_value_range = random.randint(int(bar_value_max * 0.2), bar_value_max)
    y = generate_random_samples(bar_per_loc, bar_num)
    y = bar_value_min + y * bar_value_range
    y = y.astype(np.int32)

    bar_dist = random.choice([0.5, 1, 1.5])
    bar_start = random.choice([0, 0.4, 0.8])
    # x stores the start position of every group of bars(bar_per_loc bars in one group)
    x = [bar_start]
    last = bar_start
    for i in range(bar_num - 1):
        last = last + bar_width * bar_per_loc + bar_dist
        x.append(last)
    x = np.array(x)

    if bar_direction == "horizontal":
        bar_generator = ax.barh
        set_hticks = ax.set_yticks
        set_hticklabels = ax.set_yticklabels
        get_hticklines = ax.get_yticklines
        get_vticklabels = ax.get_xticklabels
        get_vticklines = ax.get_xticklines
        # if the bars are horizontal, invert the y axis
        ax.invert_yaxis()
    else:
        bar_generator = ax.bar
        set_hticks = ax.set_xticks
        set_hticklabels = ax.set_xticklabels
        get_hticklines = ax.get_xticklines
        get_vticklabels = ax.get_yticklabels
        get_vticklines = ax.get_yticklines

    colors = plt.colormaps[random.choice(list(plt.colormaps))](np.random.rand(bar_per_loc))
    linewidth = random.choice([0, 1])
    for i in range(bar_per_loc):
        temp = bar_generator(x + bar_width * i, y[i], bar_width, align="edge", color=colors[i], # type: ignore
                             linewidth=linewidth, edgecolor="black")
        bars.append(temp)

    # fonts and fonts size
    font = random.choice(fonts_list)
    title_size = random.choice(range(title_size_min, title_size_max + 1))
    axis_label_size = random.choice(range(axis_label_size_min, axis_label_size_max + 1))
    ticks_label_size = random.choice(range(ticks_label_size_min, ticks_label_size_max + 1))
    legend_size = random.choice(range(legend_size_min, legend_size_max + 1))
    ticks_label_font = font_manager.FontProperties(fname=font, size=ticks_label_size)
    title_font = font_manager.FontProperties(fname=font, size=title_size)
    axis_label_font = font_manager.FontProperties(fname=font, size=axis_label_size)
    legend_font = font_manager.FontProperties(fname=font, size=legend_size)

    # Title and Label text
    letter_weights = np.ones((len(string.ascii_letters) + 1))
    # increase the weight of white space character
    letter_weights[-1] = int(len(letter_weights) * 0.2)
    letter_weights = list(itertools.accumulate(letter_weights))
    letters = string.ascii_letters + " "
    title_length = random.choice(range(title_length_min, title_length_max))
    title_text = "".join(random.choices(letters, cum_weights=letter_weights, k=title_length)).strip()
    xlabel_length = random.choice(range(axis_label_length_min, axis_label_length_max))
    xlabel = "".join(random.choices(letters, cum_weights=letter_weights, k=xlabel_length)).strip()
    ylabel_length = random.choice(range(axis_label_length_min, axis_label_length_max))
    ylabel = "".join(random.choices(letters, cum_weights=letter_weights, k=ylabel_length)).strip()

    ticks_label = []
    for i in range(bar_num):
        ticks_label_length = random.choice(range(ticks_label_length_min, ticks_label_length_max))
        ticks_label.append("".join(random.choices(string.ascii_letters, k=ticks_label_length)).strip())

    legend_char = []
    for i in range(bar_per_loc):
        legend_length = random.choice(range(legend_length_min, legend_length_max))
        legend_char.append("".join(random.choices(letters, k=legend_length)).strip())

    data = (x, y, ticks_label, legend_char)

    axis_label_switch = random.choice([True, False])
    title_switch = random.choice([True, False])
    legend_switch = random.choice([True, False])
    legend_pos = random.choice(legend_position)

    if axis_label_switch and axis_allowed:
        xlabel = ax.set_xlabel(xlabel, fontproperties=axis_label_font)
        ylabel = ax.set_ylabel(ylabel, fontproperties=axis_label_font)
        if bar_direction == "horizontal":
            axis_label = {"value": ylabel, "category": xlabel}
        else:
            axis_label = {"value": xlabel, "category": ylabel}

    # set the ticks and tick labels
    set_hticks(x + (bar_width / 2) * bar_per_loc)
    hticklabels = set_hticklabels(ticks_label, fontproperties=ticks_label_font)
    hticklines = get_hticklines()
    vticklabels = get_vticklabels()
    vticklines = get_vticklines()
    for label in vticklabels:
        label.set_fontproperties(ticks_label_font)
    axis_ticks = {"value": zip(vticklabels, vticklines),
                  "category": zip(hticklabels, hticklines)}

    # set legend, possible positions include: top, bottom, upper right and center right
    ax_bbox = ax.get_position()
    tight_rect = [0, 0, 1, 1]
    if legend_switch and legend_allowed:
        if legend_pos == "top":
            ax.set_position([ax_bbox.x0, ax_bbox.y0, ax_bbox.width, ax_bbox.height * 0.85])
            legend = ax.legend(legend_char, prop=legend_font, ncol=bar_per_loc, loc="lower center",
                               bbox_to_anchor=(0.5, 1))
            tight_rect = [0, 0, 1, 0.85]
        if legend_pos == "bottom":
            ax.set_position([ax_bbox.x0, ax_bbox.y0 + ax_bbox.height * 0.15, ax_bbox.width, ax_bbox.height * 0.85])
            tight_rect = [0, 0.15, 1, 1]
            if axis_label_switch == "on":
                legend = ax.legend(legend_char, prop=legend_font, ncol=bar_per_loc, loc="upper center",
                                   bbox_to_anchor=(0.5, -0.12))
            else:
                legend = ax.legend(legend_char, prop=legend_font, ncol=bar_per_loc, loc="upper center",
                                   bbox_to_anchor=(0.5, -0.05))
        if legend_pos == "right":
            ax.set_position([ax_bbox.x0, ax_bbox.y0, ax_bbox.width * 0.85, ax_bbox.height])
            tight_rect = [0, 0, 0.85, 1]
            if random.choice(["top", "center"]) == "center":
                legend = ax.legend(legend_char, prop=legend_font, ncol=1, loc="center left", bbox_to_anchor=(1, 0.5))
            else:
                legend = ax.legend(legend_char, prop=legend_font, ncol=1, loc="upper left", bbox_to_anchor=(1, 1))

    if title_switch and title_allowed:
        title_loc = random.choice(title_location)
        if legend_pos == "top":
            title = ax.set_title(title_text, fontproperties=title_font, loc="center", y=1.1)
        else:
            title = ax.set_title(title_text, fontproperties=title_font, loc=title_loc, y=1.01)

    plt.tight_layout(rect=tight_rect)
    fig.savefig(filename, dpi="figure")
    return fig, ax, bars, data, title, legend, axis_ticks, axis_label, bar_direction


def build_bar_annotations(fig_height, bars, data, direction):
    """
    Returns json annotations & meta data for bars
    """
    b = {"max_value": 0, "min_value": 0, "width": np.inf, "features": []}

    if bars is None:
        return b

    bar_heights = data[1]
    bar_per_loc = len(bars)
    bar_nums = len(bars[0])

    b["min_value"] = int(np.min(bar_heights))
    b["max_value"] = int(np.max(bar_heights))

    for j in range(bar_per_loc):
        b["features"].append({"feature": data[3][j], "data": []})
        for i in range(bar_nums):
            h = int(bar_heights[j][i])
            b_cor = compute_bbox(fig_height, bars[j][i].get_tightbbox())

            if direction == "horizontal":
                w = int(b_cor[3] - b_cor[1])
            else:
                w = int(b_cor[2] - b_cor[0])

            b["width"] = w if w < b["width"] else b["width"]
            b["features"][j]["data"].append({
                "bbox": b_cor,
                "value": h,
                "category": data[2][i]
            })

    return b


def build_tick_annotations(fig_height, axes, axis_ticks):
    """
    Returns json annotations for axis ticks
    """
    ti = {
        "value": {"bbox": [0,0,0,0], "entries": []},
        "category": {"bbox": [0,0,0,0], "entries": []}
    }

    if axis_ticks is None:
        return ti

    ti["value"]["bbox"] = compute_bbox(fig_height, axes.yaxis.get_tightbbox())
    for lbl, ln in axis_ticks["value"]:
        text = lbl.get_text()
        lbl_cor = compute_bbox(fig_height, lbl.get_tightbbox())
        ln_cor = compute_bbox(fig_height, ln.get_tightbbox())
        ti["value"]["entries"].append({"bbox": lbl_cor, "text": text, "tick": ln_cor})

    ti["category"]["bbox"] = compute_bbox(fig_height, axes.xaxis.get_tightbbox())
    for lbl, ln in axis_ticks["category"]:
        text = lbl.get_text()
        lbl_cor = compute_bbox(fig_height, lbl.get_tightbbox())
        ln_cor = compute_bbox(fig_height, ln.get_tightbbox())
        ti["category"]["entries"].append({"bbox": lbl_cor, "text": text, "tick": ln_cor})

    return ti


def build_axis_annotations(fig_height, axis_label):
    """
    Returns json annotations for axis labels
    """
    al = {
        "value":{"bbox": [0,0,0,0], "text": ""},
        "category":{"bbox": [0,0,0,0], "text": ""}
    }

    if axis_label is None:
        return al

    al["value"]["bbox"] = compute_bbox(fig_height, axis_label["value"].get_tightbbox())
    al["value"]["text"] = axis_label["value"].get_text()

    al["category"]["bbox"] = compute_bbox(fig_height, axis_label["category"].get_tightbbox())
    al["category"]["text"] = axis_label["category"].get_text()

    return al


def build_title_annotations(fig_height, title):
    """
    Returns json annotations for chart title
    """
    t = {"bbox": [0, 0, 0, 0], "text": ""}

    if title is None:
        return t

    t["text"] = title.get_text()
    t["bbox"] = compute_bbox(fig_height, title.get_window_extent())

    return t


def build_legend_annotations(fig_height, legend):
    """
    Returns json annotations for chart legend
    """
    l = {"bbox": [0, 0, 0, 0], "entries": []}

    if legend is None:
        return l

    l["bbox"] =  compute_bbox(fig_height, legend.get_window_extent())
    for t, h in zip(legend.get_texts(), legend.get_patches()):
        text = t.get_text()
        b_cor = compute_bbox(fig_height, t.get_window_extent())
        h_cor = compute_bbox(fig_height, h.get_window_extent())
        l["entries"].append({"text_bbox": b_cor, "text": text, "hint_bbox": h_cor})

    return l


def compute_bbox(fig_height, bbox):
    """
    Convert bbox from bottom-left to top-left format
    """
    cor = bbox.get_points()
    tmp = cor[0][1]
    cor[0][1] = fig_height - cor[1][1]
    cor[1][1] = fig_height - tmp
    b_cor = [round(cor[0][0]), round(cor[0][1]), round(cor[1][0]), round(cor[1][1])]
    return [int(i) for i in b_cor]


def annotate(plot_objs):
    """
    Generates & returns json annotations for chart
    """
    fig, ax, bars, data, title, legend, axis_ticks, axis_label, bar_direction = plot_objs

    imgsize = list(map(int, fig.get_size_inches() * fig.dpi))
    fig_height = imgsize[1]

    bar_ann = build_bar_annotations(fig_height, bars, data, bar_direction)
    tick_ann = build_tick_annotations(fig_height, ax, axis_ticks)
    axis_ann = build_axis_annotations(fig_height, axis_label)
    title_ann = build_title_annotations(fig_height, title)
    legend_ann = build_legend_annotations(fig_height, legend)

    return {
            "chart_metadata": {
                "title": {
                    "text": title_ann["text"],
                    "bbox": title_ann["bbox"]
                },
                "type": bar_direction,
                "bar_width": bar_ann["width"],
                "legend": {
                    "entries": [
                        {
                            "feature": {
                                "text": e["text"],
                                "bbox": e["text_bbox"]
                            },
                            "patch": {
                                "bbox": e["hint_bbox"]
                            }
                        } for e in legend_ann["entries"]
                    ],
                    "bbox": legend_ann["bbox"]
                },
                "size": {
                    "bbox": [0, 0, imgsize[0], imgsize[1]]
                },
                "origin": {
                    "bbox": tick_ann["value"]["entries"][0]["tick"]
                }
            },
            "data": {
                "category_axis": {
                    "label": {
                        "text": axis_ann["category"]["text"],
                        "bbox": axis_ann["category"]["bbox"]
                    },
                    "ticks": [
                        {
                            "label": {
                                "text": e["text"],
                                "bbox": e["bbox"],
                            },
                            "bbox": e["tick"]
                        } for e in tick_ann["category"]["entries"]
                    ],
                    "bbox": tick_ann["category"]["bbox"]
                },
                "value_axis": {
                    "label": {
                        "text": axis_ann["value"]["text"],
                        "bbox": axis_ann["value"]["bbox"]
                    },
                    "ticks": [
                        {
                            "label": {
                                "text": e["text"],
                                "bbox": e["bbox"],
                            },
                            "bbox": e["tick"]
                        } for e in tick_ann["value"]["entries"]
                    ],
                    "min_value": bar_ann["min_value"],
                    "max_value": bar_ann["max_value"],
                    "bbox": tick_ann["value"]["bbox"],
                },
                "features": [
                    {
                        "feature": feature["feature"],
                        "data": [{
                                "category": f_data["category"],
                                "value": f_data["value"],
                                "bbox": f_data["bbox"]
                            } for f_data in feature["data"]
                        ]
                    } for feature in bar_ann["features"]
                ]
            }
        }


def bbox_one_line(json_str):
    """
    Rewrites multi-line bbox to be one line
    """
    bbox_pattern = re.compile(r'"(?:bbox|origin)":\s*\[([^\]]+)\]', re.DOTALL)
    return bbox_pattern.sub(lambda m: f'"bbox": [{",".join([x.strip() for x in m.group(1).split(",")])}]', json_str)


def generate_plots_chunk(args):
    i, store_dir, train_or_test, val, schema = args
    try:
        img_id = "{}_{}.{}".format(train_or_test, i, "png")
        img_path = os.path.join(store_dir, "images", img_id)
        plot_objs = get_random_plot(img_path)

        annotation = annotate(plot_objs)

        # Validate the annotation against the schema if validation is enabled
        if val:
            try:
                validate(instance=annotation, schema=schema)
            except ValidationError as e:
                raise e from e

        # Save JSON
        json_id = "{}_{}.{}".format(train_or_test, i, "json")
        json_path = os.path.join(store_dir, "annotations", json_id)
        with open(json_path, 'w') as f:
            json_str = bbox_one_line(json.dumps(annotation, indent=2))
            f.write(json_str)

        # close the figure
        plt.close(plot_objs[0])
    except Exception as e:
        with open(os.path.join(store_dir, "error_log.txt"), "a") as f_error:
            f_error.write("{} error: {}".format(img_path, e))
            f_error.write("\n")


def generate_plots(output_dir, n, num_processes, clear, val, train_or_test):
    """
    Generates annotated bar charts

    :param output_dir: Where to put the charts
    :param n: Number of charts to generate
    :param num_processes: Number of processes to use for generation
    :param clear: Flag whether to clear the output directory
    :param val: Flag whether to validate the annotations
    :param train_or_test: Whether to generate train or test data
    """
    # Clear output directory if requested
    store_dir = os.path.join(output_dir, train_or_test)
    if clear and os.path.exists(store_dir):
        shutil.rmtree(store_dir)

    # Create separate directories for images and JSON annotations
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(os.path.join(store_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(store_dir, "annotations"), exist_ok=True)

    # Load the schema if validation is enabled
    schema = None
    if val:
        with open('format.json', 'r') as f:
            schema = json.load(f)

    # Create a list of arguments for each task
    tasks = [(i, store_dir, train_or_test, val, schema) for i in range(n)]

    # Use multiprocessing.Pool to generate plots in parallel
    with mp.Pool(processes=num_processes) as pool, tqdm(total=n) as pbar:
        for _ in pool.imap_unordered(generate_plots_chunk, tasks):
            pbar.update()

    print(train_or_test + " plot generation done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This python script generates random bar charts and their elements' bounding boxes")
    parser.add_argument('--output_dir', help='Directory to save the images and JSON files.', required=True, type=str)
    parser.add_argument("--train_total", help="Number of traning images", required=True, type=int)
    parser.add_argument("--test_total", help="Number of test images", required=True, type=int)
    parser.add_argument("--num_processes", help="Number of processes to use", default=1, type=int)
    parser.add_argument('--clear_output', help='Clear the output directory before generation.', action='store_true')
    parser.add_argument('--validate', help='Validate the generated JSON annotations against the schema.', action='store_true')

    args = parser.parse_args()

    print("generating {} training data:".format(args.train_total))
    generate_plots(
        args.output_dir,
        args.train_total,
        args.num_processes,
        args.clear_output,
        args.validate,
        "train"
    )

    print("generating {} test data".format(args.test_total))
    generate_plots(
        args.output_dir,
        args.test_total,
        args.num_processes,
        args.clear_output,
        args.validate,
        "test"
    )