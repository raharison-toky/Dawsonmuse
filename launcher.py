from dawsonmuse import generate_save_fn, muserecorder
import multiprocessing
import os
import time

def run_exp():
    import Experiment


if __name__ == "__main__":

    os.system(r"start bluemuse://start?streamfirst=true")

    cwd = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(cwd,"data")
    datadir = os.path.join(datadir,"experiment")
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    board_name = 'muse2016'
    experiment = 'name of experiment'
    session = 1
    subject = 1
    file_dir = datadir

    save_fn = generate_save_fn(board_name, experiment, subject, session, file_dir)
    data_recorder = muserecorder(save_fn)
    data_recorder.start()
    
    p = multiprocessing.Process(target=run_exp)
    p.start()
    p.join()
    time.sleep(1)
    data_recorder.end_recording()