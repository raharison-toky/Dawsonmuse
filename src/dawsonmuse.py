import multiprocessing
import time
from time import time
import logging
from multiprocessing import Process
import os
from os import path, makedirs
import numpy as np
import pandas as pd
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
from typing import Union, List, Optional
from pathlib import Path
from sklearn.linear_model import LinearRegression
from time import time, strftime, gmtime
from time import strftime, gmtime
from muselsl.constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK, LSL_PPG_CHUNK, LSL_ACC_CHUNK, LSL_GYRO_CHUNK
from pathlib import Path
import matplotlib.pyplot as plt
import mne

"""
Code from: https://github.com/NeuroTechX/eeg-notebooks/blob/master/eegnb/__init__.py

Title: EEG notebooks
Author: NeuroTechX
Date: December 26, 2021
"""

logger = logging.getLogger(__name__)

DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def get_recording_dir(
    board_name: str,
    experiment: str,
    subject_id: int,
    session_nb: int,
    site="local",
    data_dir=DATA_DIR,
) -> Path:
    # convert subject ID to 4-digit number
    subject_str = f"subject{subject_id:0>4}"
    session_str = f"session{session_nb:0>3}"
    return _get_recording_dir(
        board_name, experiment, subject_str, session_str, site, data_dir=data_dir
    )


def _get_recording_dir(
    board_name: str,
    experiment: str,
    subject_str: str,
    session_str: str,
    site: str,
    data_dir=DATA_DIR,
) -> Path:
    """A subroutine of get_recording_dir that accepts subject and session as strings"""
    # folder structure is /DATA_DIR/experiment/site/subject/session/*.csv
    recording_dir = (
        Path(data_dir) / experiment / site / board_name / subject_str / session_str
    )

    # check if directory exists, if not, make the directory
    if not path.exists(recording_dir):
        makedirs(recording_dir)

    return recording_dir


def generate_save_fn(
    board_name: str,
    experiment: str,
    subject_id: int,
    session_nb: int,
    data_dir=DATA_DIR,
) -> Path:
    """Generates a file name with the proper trial number for the current subject/experiment combo"""
    recording_dir = get_recording_dir(
        board_name, experiment, subject_id, session_nb, data_dir=data_dir
    )

    # generate filename based on recording date-and-timestamp and then append to recording_dir
    return recording_dir / (
        "recording_%s" % strftime("%Y-%m-%d-%H.%M.%S", gmtime()) + ".csv"
    )

"""
Code adapted from: https://github.com/alexandrebarachant/muse-lsl/blob/master/muselsl/record.py

Title: muselsl/record
Author: Alexandre Barachant, PhD
Date: December 26, 2021
"""

class muserecorder(Process):

    """
    This is a child class of Process that allows a variable recording time. It takes in the path where the EEG data will be saved and a timout in case the marker stream is not found.
    """

    def __init__(self,save_fn,markers_timout=120):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self.filename = save_fn
        self.markers_timout = markers_timout
    def run(self):
        filename= self.filename,
        dejitter=False,
        data_source="EEG",
        continuous: bool = True
        chunk_length = LSL_EEG_CHUNK
        if data_source == "PPG":
            chunk_length = LSL_PPG_CHUNK
        if data_source == "ACC":
            chunk_length = LSL_ACC_CHUNK
        if data_source == "GYRO":
            chunk_length = LSL_GYRO_CHUNK

        if not filename:
            filename = os.path.join(os.getcwd(), "%s_recording_%s.csv" %
                                    (data_source,
                                    strftime('%Y-%m-%d-%H.%M.%S', gmtime())))

        print("Looking for a %s stream..." % (data_source))
        streams = resolve_byprop('type', "EEG", timeout = LSL_SCAN_TIMEOUT)

        if len(streams) == 0:
            print("Can't find %s stream." % (data_source))
            return

        print("Started acquiring data.")
        inlet = StreamInlet(streams[0], max_chunklen=chunk_length)
        # eeg_time_correction = inlet.time_correction()

        print("Looking for a Markers stream...")
        marker_streams = resolve_byprop(
            'type', 'Markers', timeout=LSL_SCAN_TIMEOUT)

        if marker_streams:
            inlet_marker = StreamInlet(marker_streams[0])
            print(marker_streams[0].name())
        else:
            inlet_marker = False
            raise("Can't find Markers stream.")

        info = inlet.info()
        description = info.desc()

        Nchan = info.channel_count()

        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]
        for i in range(1, Nchan):
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))

        res = []
        timestamps = []
        markers = []
        t_init = time()
        time_correction = inlet.time_correction()
        last_written_timestamp = None
        print('Start recording at time t=%.3f' % t_init)
        print('Time correction: ', time_correction)

        #creating a counter for the number of times no markers have been found so that the program stops if something weird happens

        no_markers = 0

        while not self.exit.is_set():
            try:
                data, timestamp = inlet.pull_chunk(
                    timeout=1.0, max_samples=chunk_length)

                if timestamp:
                    res.append(data)
                    timestamps.extend(timestamp)
                    tr = time()
                if inlet_marker:
                    marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
                    if timestamp:
                        markers.append([marker, timestamp])

                # Save every 5s
                if continuous and (last_written_timestamp is None or last_written_timestamp + 5 < timestamps[-1]):
                    missing_markers = final_save(
                        self.filename,
                        res,
                        timestamps,
                        time_correction,
                        dejitter,
                        inlet_marker,
                        markers,
                        ch_names,
                        last_written_timestamp=last_written_timestamp,
                    )
                    last_written_timestamp = timestamps[-1]
                    no_markers += 5*missing_markers

                    #it will stop after 2 minutes without any markers
                    if no_markers >= self.markers_timout:
                        raise("something went wrong with the markers")

                    print("--------")
                    print("time passed without markers (s): " + str(no_markers))
                    print("--------")

            except KeyboardInterrupt:
                break

        time_correction = inlet.time_correction()
        print("Time correction: ", time_correction)

        show_data(
            self.filename,
            res,
            timestamps,
            time_correction,
            dejitter,
            inlet_marker,
            markers,
            ch_names,
        )

        print("Done - wrote file: {}".format(filename))
    
    def end_recording(self):
        self.exit.set()

class markers_stream:

    def __init__(self):
        self.StreamInfo = StreamInfo("Markers", "Markers", 1, 0, "int32", "myuidw43536")
        self.StreamOutlet = StreamOutlet(self.StreamInfo)

    def mark(self, marker):
        if isinstance(marker,list):
            self.StreamOutlet.push_sample(marker)
            print("marked " + str(marker))
        elif isinstance(marker,int):
            print([marker])
            self.StreamOutlet.push_sample([marker])
            print("marked " + str(marker))


def final_save(
    filename: Union[str, Path],
    res: list,
    timestamps: list,
    time_correction,
    dejitter: bool,
    inlet_marker,
    markers,
    ch_names: List[str],
    last_written_timestamp: Optional[float] = None,
):
    res = np.concatenate(res, axis=0)
    timestamps = np.array(timestamps) + time_correction

    if dejitter:
        y = timestamps
        X = np.atleast_2d(np.arange(0, len(y))).T
        lr = LinearRegression()
        lr.fit(X, y)
        timestamps = lr.predict(X)

    res = np.c_[timestamps, res]
    data = pd.DataFrame(data=res, columns=["timestamps"] + ch_names)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if inlet_marker and markers:
        print(inlet_marker)
        print(markers)
        print("\n")
        n_markers = len(markers[0][0])
        for ii in range(n_markers):
            data['Marker%d' % ii] = 0
        # process markers:
        for marker in markers:
            # find index of markers
            ix = np.argmin(np.abs(marker[1] - timestamps))
            for ii in range(n_markers):
                data.loc[ix, "Marker%d" % ii] = marker[0][ii]
        missing_markers = 0

    if not (inlet_marker and markers):
        print("markers not found")
        missing_markers = 1

    # If file doesn't exist, create with headers
    # If it does exist, just append new rows
    if not Path(filename).exists():
        # print("Saving whole file")
        data.to_csv(filename, float_format='%.3f', index=False)
    else:
        # print("Appending file")
        # truncate already written timestamps
        data = data[data['timestamps'] > last_written_timestamp]
        data.to_csv(filename, float_format='%.3f', index=False, mode='a', header=False)

    return missing_markers

def show_data(
    filename: Union[str, Path],
    res: list,
    timestamps: list,
    time_correction,
    dejitter: bool,
    inlet_marker,
    markers,
    ch_names: List[str],
    last_written_timestamp: Optional[float] = None,
):
    res = np.concatenate(res, axis=0)
    timestamps = np.array(timestamps) + time_correction

    if dejitter:
        y = timestamps
        X = np.atleast_2d(np.arange(0, len(y))).T
        lr = LinearRegression()
        lr.fit(X, y)
        timestamps = lr.predict(X)

    res = np.c_[timestamps, res]
    data = pd.DataFrame(data=res, columns=["timestamps"] + ch_names)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if inlet_marker and markers:
        print(inlet_marker)
        print(markers)
        print("\n")
        n_markers = len(markers[0][0])
        for ii in range(n_markers):
            data['Marker%d' % ii] = 0
        # process markers:
        for marker in markers:
            # find index of markers
            ix = np.argmin(np.abs(marker[1] - timestamps))
            for ii in range(n_markers):
                data.loc[ix, "Marker%d" % ii] = marker[0][ii]

    # If file doesn't exist, create with headers
    # If it does exist, just append new rows
    if not Path(filename).exists():
        # print("Saving whole file")
        data.to_csv(filename, float_format='%.3f', index=False)
    else:
        # print("Appending file")
        # truncate already written timestamps
        data = data[data['timestamps'] > last_written_timestamp]
        data.to_csv(filename, float_format='%.3f', index=False, mode='a', header=False)
    print(data)
    print("\n \n \n")

"""
This is original code for data analysis
"""
class session_set:

    EEG_COL = ['TP', 'AF','markers']
    CHANNEL_TYPES = ["eeg","eeg","stim"]
    S_FRQ = 256

    def __init__(self,filelist: list, name:str , names_dict: dict, tmin: float = -0.2, tmax: float = 1.0, lag: int = 0):
        
        self.name = name
        self.names_dict = names_dict
        self.erp_dict = {}
        self.grand_avg_dict = {}
        keylist = self.names_dict.keys()
        self.tmin = tmin
        self.tmax = tmax

        for file in filelist:
            col_names = ['timestamps', 'TP9', 'AF7', 'AF8', 'TP10', 'Right AUX','markers']
            full_EEG_df = pd.read_csv(file,names = col_names, skiprows=1)
            full_EEG_df["timestamps"] = full_EEG_df["timestamps"] - lag
            averaged_df = pd.DataFrame()
            averaged_df["TP"] = full_EEG_df[["TP9","TP10"]].mean(axis=1)
            averaged_df["AF"] = full_EEG_df[["AF7","AF8"]].mean(axis=1)
            averaged_df["markers"] = full_EEG_df["markers"]
            data = np.array([averaged_df[i].to_numpy() for i in self.EEG_COL])
            info = mne.create_info(self.EEG_COL,self.S_FRQ,self.CHANNEL_TYPES)
            raw = mne.io.RawArray(data,info)
            filtered_iir = raw.filter(l_freq=1, h_freq=15,method="iir")
            iir_events = mne.find_events(filtered_iir)
            iir_epochs = mne.Epochs(filtered_iir, iir_events, event_id=self.names_dict, tmin=self.tmin, tmax=self.tmax,
                    preload=True)
            dropped = iir_epochs.copy()
            reject_criteria = dict(eeg=100)
            _ = dropped.drop_bad(reject=reject_criteria)

            for key in keylist:
                averaged_erp = dropped[key].average()
                if key in self.erp_dict:
                    self.erp_dict[key].append(averaged_erp)
                else:
                    self.erp_dict[key] = [averaged_erp]

        for key in keylist:
            self.grand_avg_dict[key] = mne.grand_average(self.erp_dict[key])

    def plot(self,electrode: str= "TP",markers=None):

        if electrode.upper() not in ["TP","AF"]:
          raise ValueError(f"you must choose between TP and AF, you entered {electrode}")

        if markers == None:
            keylist = self.names_dict.keys()
        else:
            keylist = markers

        for key in keylist:
            df = self.grand_avg_dict[key].to_data_frame()
            plt.plot(df["time"],df[electrode.upper()],label = key)
            plt.xlim((1000*self.tmin,1000*self.tmax))
            plt.xlabel("t (ms)",fontsize=8)
            plt.ylabel(f"{electrode.upper()} electrodes voltage (μV)",fontsize=8)
            plt.gca().tick_params(labelsize=8)
            plt.legend(loc="upper right", prop={'size': 8},markerscale=1)
