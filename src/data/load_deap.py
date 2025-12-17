import os
import pickle
import numpy as np

class DEAPLoader:
    def __init__(self, data_dir, use_eeg_only=True):
        self.data_dir = data_dir
        self.use_eeg_only = use_eeg_only

    def load_subject(self, subject_id):
        """
        subject_id: int, 1~32
        """
        filename = f"s{subject_id:02d}.dat"
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')

        data = data_dict['data']      # (40, 40, 8064)
        labels = data_dict['labels']  # (40, 4)

        if self.use_eeg_only:
            data = data[:, :32, :]    # EEG only

        return data, labels

    def load_all_subjects(self):
        all_data = []
        all_labels = []
        all_subject_ids = []

        for sid in range(1, 33):
            data, labels = self.load_subject(sid)
            all_data.append(data)
            all_labels.append(labels)
            all_subject_ids.extend([sid] * data.shape[0])

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_subject_ids = np.array(all_subject_ids)

        return all_data, all_labels, all_subject_ids


if __name__ == "__main__":
    data_dir = "./data/data_preprocessed_python/"
    loader = DEAPLoader(data_dir)

    X, y, subjects = loader.load_all_subjects()
    print("EEG shape:", X.shape)       # (1280, 32, 8064)    1280 trials × 32 EEG 通道 × 原始时序
    print("Labels shape:", y.shape)    # (1280, 4)          1280 trials × 4 个标签
    print("Subjects shape:", subjects.shape)
