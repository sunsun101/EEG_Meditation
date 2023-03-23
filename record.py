import time
import mne
from mne.datasets import eegbci

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
import os


def main():
    BOARD_ID = 8
    PARTICIPANT_ID = "amp"
    RECORDING_DIR = "./"
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream ()
    time.sleep(1200)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer
    raw = getdata(data,BOARD_ID)
    save_raw(raw,RECORDING_DIR, PARTICIPANT_ID)
    board.stop_stream()
    board.release_session()

    print(data)

def getdata(data,board):

    eeg_channels = BoardShim.get_eeg_channels(board)
    data[eeg_channels] = data[eeg_channels] / 1e6


    data = data[eeg_channels]
    ch_names = BoardShim.get_eeg_names(board)
    ch_types = (['eeg'] * len(eeg_channels))

    sfreq = BoardShim.get_sampling_rate(board)
    
    #Create Raw data from MNE
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    # print(raw)
    raw_data = raw.copy()
    eegbci.standardize(raw_data)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_data.set_montage(montage)
    
    return raw_data

def create_session_folder(subj,dir):
    base_path = os.getcwd() + "\\"
    dir = base_path + dir
    folder_name = f'{subj}'
    print(folder_name)
    if os.path.isdir(os.path.join(dir, folder_name)):
        folder_path = os.path.join(dir, folder_name)
    else:
        folder_path = os.path.join(dir, folder_name)
        os.makedirs(folder_path)
    return folder_path

def save_raw(raw, dir, participant_id):
    folder_path = create_session_folder(participant_id,dir)
    raw.save(os.path.join(folder_path, f'{participant_id}{".fif"}'), overwrite = True)
    return os.path.basename(folder_path)


if __name__ == "__main__":
    main()