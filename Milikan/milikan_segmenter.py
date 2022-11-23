import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

MOTION_DATA_PATH = "motion_data"
STOPPING_VOLTAGES_FILENAME = "stopping_voltages.xlsx"
PROCESSED_DATASET_FILENAME = "segmented_data.json"


def extract_idnum(filename):
    return int("".join([char for char in list(str(filename)) if char.isnumeric()]))


def load_stp_volts(stp_volts_filename):
    stp_volts = pd.read_excel(stp_volts_filename)

    stp_volts = stp_volts.iloc[:, [0, 1]]
    stp_volts.set_index("trial", inplace=True)

    return stp_volts


def find_largest_upslope(data, patience=4, min_inc=1):
    upslope_ranges = []

    start_idx = 0
    recording = False
    for i in range(len(data) - patience):
        if not recording:
            if all([data[i + j + 1] > data[i + j] + min_inc for j in range(patience - 1)]):
                start_idx = i
                recording = True
        else:
            if all([data[i + j + 1] <= data[i + j] + min_inc for j in range(patience - 1)]):
                upslope_ranges.append((start_idx, i))
                recording = False

    if recording:
        upslope_ranges.append((start_idx, len(data) - 1))

    print(max(upslope_ranges, key=lambda x: x[1] - x[0]))
    return max(upslope_ranges, key=lambda x: x[1] - x[0])


def find_largest_downslope(data, patience=4, min_dec=1):
    downslope_ranges = []

    start_idx = 0
    recording = False
    for i in range(len(data) - patience):
        if not recording:
            if all([data[i + j + 1] < data[i + j] - min_dec for j in range(patience - 1)]):
                start_idx = i
                recording = True
        else:
            if all([data[i + j + 1] >= data[i + j] - min_dec for j in range(patience - 1)]):
                downslope_ranges.append((start_idx, i))
                recording = False

    if recording:
        downslope_ranges.append((start_idx, len(data) - 1))

    return max(downslope_ranges, key=lambda x: x[1] - x[0])


def visualize(dataset):
    fig = plt.figure()
    fig.subplots_adjust(0.075, 0.075, 1 - 0.075, 1 - 0.075, 0.6, 0.45)

    i = 1
    for sample_id in dataset:
        plt.subplot(6, 10, i)
        plt.plot(dataset[sample_id]["data"], color='tab:blue')
        plt.vlines(dataset[sample_id]["upslope_range"], ymin=min(dataset[sample_id]["data"]),
                   ymax=max(dataset[sample_id]["data"]), color='tab:orange')
        plt.vlines(dataset[sample_id]["downslope_range"], ymin=min(dataset[sample_id]["data"]),
                   ymax=max(dataset[sample_id]["data"]), color='tab:orange')
        i += 1


def generate_dataset(data_pathname, stp_volts_filename, output_filename):
    dataset = {}

    stp_volts = load_stp_volts(stp_volts_filename)

    for filename in pathlib.Path(data_pathname).iterdir():
        data_df = pd.read_excel(filename)
        data = [int(i) for i in list(data_df.iloc[:, 0])]

        upslope_range = find_largest_upslope(data)
        downslope_range = find_largest_downslope(data)
        peak_range = (upslope_range[1] + 1, downslope_range[0] - 1)

        sample_id = extract_idnum(filename)

        dataset[sample_id] = {}
        dataset[sample_id]["stp_volt"] = int(stp_volts.at[sample_id, "voltage (V)"])
        dataset[sample_id]["data"] = data
        dataset[sample_id]["upslope_range"] = upslope_range
        dataset[sample_id]["peak_range"] = peak_range
        dataset[sample_id]["downslope_range"] = downslope_range

    dataset_file = open(output_filename, 'w')
    json.dump(dataset, dataset_file, indent=4)
    dataset_file.close()

    return dataset


dataset = generate_dataset(MOTION_DATA_PATH, STOPPING_VOLTAGES_FILENAME, PROCESSED_DATASET_FILENAME)
visualize(dataset)

plt.show()
