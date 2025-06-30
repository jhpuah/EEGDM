import mne
import os
import numpy as np
from tqdm import tqdm
import pickle
import shutil
from typing import OrderedDict
from omegaconf import DictConfig

input_files = train_val_split = output_files = processing = None

mne.set_log_level("CRITICAL")

ave_ch_order = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

bipolar_ch_order = [  
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "A1-T3", "T3-C3", "C3-CZ", "C4-CZ", "T4-C4", "A2-T4", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

# Bipolar
anode = ["EEG {0}-REF".format(ch.split("-")[0].strip()) for ch in bipolar_ch_order]
cathode = ["EEG {0}-REF".format(ch.split("-")[1].strip()) for ch in bipolar_ch_order]


def process_one_edf(filename):
    raw: mne.io.Raw = mne.io.read_raw_edf(filename, preload=True)

    raw.filter(l_freq=processing["bandpass_l"], h_freq=processing["bandpass_h"])
    raw.notch_filter(processing["notch"])
    raw.resample(processing["sfreq"], n_jobs=5)
    
    raw.drop_channels(list(set(raw.info["ch_names"]) - set(ave_ch_order)))
    try:
        raw.reorder_channels(ave_ch_order)
    except:
        if not processing["bipolar"]:
            # LaBraM include T1 and T2 channels which are missing in 1 file: gped_022_a_.edf
            # and they just ignore the file
            # hence exit early here to mimic their behavior
            return []
            # when using bipolar (TCP) can just proceed
    
    if processing["bipolar"]:
        raw = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode, ch_name=bipolar_ch_order)
        raw.drop_channels(set(raw.info["ch_names"]) - set(bipolar_ch_order))
        raw.reorder_channels(bipolar_ch_order)

    signal, times = raw[:]
    signal *= processing["scale"]
    
    offset = signal.shape[1]
    signal = np.concatenate([signal, signal, signal], axis=1)

    rec_filename = filename[:-3] + "rec"
    extracted_samples = []
    if processing["unique"]:
        start_end__ch_ev = OrderedDict() # dict does not gaurantee the order of iteration
        with open(rec_filename) as rec:
            for line in rec:
                ch_idx, start_sec, end_sec, ev_code = line.strip().split(",")
                ch_idx = int(ch_idx)
                ev_code = int(ev_code)

                s_e = f"{start_sec}-{end_sec}"
                
                if s_e in start_end__ch_ev:
                    start_end__ch_ev[s_e][0].append(ch_idx)
                    start_end__ch_ev[s_e][1].append(ev_code)
                else:
                    start_end__ch_ev[s_e] = [[ch_idx], [ev_code]]
        
        match processing["overlap"]:
            case "take":
                pass
            case "drop":
                to_drop = []
                for k, (ch, ev) in start_end__ch_ev.items():
                    unique_ev_count = len(np.unique(np.array(ev)))
                    if unique_ev_count > 1:
                        to_drop.append(k)
                for k in to_drop:
                    del start_end__ch_ev[k]
            case "multi":
                ...
                # TODO support for multilabel?

        for s_e, (ch, ev) in start_end__ch_ev.items():
            start_sec, end_sec = s_e.split("-")
            start_sec = float(start_sec)
            end_sec = float(end_sec)
            start = np.where(times >= start_sec)[0][0]
            end = np.where(times >= end_sec)[0][0]
            
            data = signal[
                :,
                offset + start - int(processing["sec_before_start"] * processing["sfreq"])
                    : offset + end + int(processing["sec_after_end"] * processing["sfreq"])
            ]
            
            label = np.zeros((data.shape[0]))
            label[ch] = np.array(ev) # assign event to each channel: only make sense for bipolar montage

            if processing["single_ch"]:
                ch_idx = np.arange(data.shape[0])
                for d, l, ci in zip(data, label, ch_idx):
                    sample = {
                        "data": d.reshape(1, -1),
                        "ch_idx": ci
                    }
                    if processing["bipolar"]: # TUEV annotation is channel-wise on bipolar (TCP) montage
                        if processing["exclude_null_class"] and l.item() == 0: continue
                        sample["label"] = l
                    
                    extracted_samples.append(sample)

            else:
                unique_label, count = np.unique(label[np.nonzero(label)], return_counts=True)
                major_label = unique_label[np.argmax(count)]
                sample = {
                    "data": data,
                    "label": major_label,
                } 
                if processing["bipolar"]:
                    sample["ch_label"] = label
                
                extracted_samples.append(sample)
    else:

# Starting from this point, the code are modified from preprocessing code of LaBraM
# which is based on BIOT
# https://github.com/ycq091044/BIOT
        event_data = np.genfromtxt(rec_filename, delimiter=",")

        num_ev = event_data.shape[0]
        features = np.zeros([num_ev, signal.shape[0], int(processing["sfreq"]) * 5])
        offending_channel = np.zeros([num_ev, 1])
        labels = np.zeros([num_ev, 1])
        for i, (ch_idx, start_sec, end_sec, ev_code) in enumerate(event_data):    
            ch_idx = int(ch_idx)
            start = np.where(times >= start_sec)[0][0]
            end = np.where(times >= end_sec)[0][0]
            
            features[i, :] = signal[
                :, offset + start - 2 * int(processing["sfreq"]) : offset + end + 2 * int(processing["sfreq"])
            ]
            
            offending_channel[i, :] = ch_idx
            labels[i, :] = int(ev_code)

        for idx, (f, oc, l) in enumerate(
            zip(features, offending_channel, labels)
        ):
            extracted_samples.append({
                "signal": f,
                "offending_channel": oc,
                "label": l,
            })
                
    return extracted_samples

def load_all_edf(base_dir, out_dir):    
    dir = sorted(os.walk(base_dir))
    for dirName, subdirList, fileList in tqdm(dir):
        fileList = sorted(fileList)
        for fname in fileList:
            fname_ext = fname[-4:]
            if fname_ext == ".edf":
                samples = process_one_edf(os.path.join(dirName, fname))
                for idx, sample in enumerate(samples):
                    out_filename = os.path.join(out_dir, fname.split(".")[0] + "-" + str(idx) + ".pkl")
                    with open(out_filename, "wb") as f:
                        pickle.dump(sample, f)
                        # Ladder!


def load_files():
    root_train = os.path.join(input_files["root"], input_files["train_dir"])
    root_test = os.path.join(input_files["root"], input_files["test_dir"])

    out_train = os.path.join(output_files["root"], output_files["train_dir"])
    out_test = os.path.join(output_files["root"], output_files["test_dir"])
    out_val = os.path.join(output_files["root"], output_files["val_dir"])
    if not os.path.exists(out_train):
        os.makedirs(out_train)
    if not os.path.exists(out_test):
        os.makedirs(out_test)
    if not os.path.exists(out_val):
        os.makedirs(out_val)

    load_all_edf(root_train, out_train)
    load_all_edf(root_test, out_test)

    np.random.seed(train_val_split["seed"])

    train_files = sorted(os.listdir(out_train))
    train_sub = sorted(list(set([f.split("_")[0] for f in train_files])))
    val_sub = np.random.choice(train_sub, size=int(len(train_sub) * train_val_split["pct"]), replace=False)

    val_files = [f for f in train_files if f.split("_")[0] in val_sub]

    for file in val_files:
        shutil.move(os.path.join(out_train, file), os.path.join(out_val))

def entry(config: DictConfig):
    # HACK
    global input_files, train_val_split, output_files, processing
    input_files = config["input_files"]
    train_val_split = config["train_val_split"]
    output_files = config["output_files"]
    processing = config["processing"]
    load_files()

# if __name__ == "__main__":
    # load_files()