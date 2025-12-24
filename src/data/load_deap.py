import os
import pickle
import numpy as np

class DEAPLoader:
    def __init__(self, data_dir, use_eeg_only=True):
        self.data_dir = data_dir
        self.use_eeg_only = use_eeg_only
        
        # 验证数据目录是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        print(f"DEAPLoader initialized with data_dir: {data_dir}")
        print(f"Data directory exists: {os.path.exists(data_dir)}")
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            print(f"Number of files in data directory: {len(files)}")
            if files:
                print(f"Sample files: {files[:3]}")

    def load_subject(self, subject_id):
        """
        subject_id: int, 1~32
        """
        filename = f"s{subject_id:02d}.dat"
        filepath = os.path.join(self.data_dir, filename)
        
        print(f"Trying to load: {filepath}")
        print(f"File exists: {os.path.exists(filepath)}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Subject data file not found: {filepath}")

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
            try:
                data, labels = self.load_subject(sid)
                all_data.append(data)
                all_labels.append(labels)
                all_subject_ids.extend([sid] * data.shape[0])
            except FileNotFoundError as e:
                print(f"Skipping subject {sid}: {e}")
                continue

        if not all_data:
            raise RuntimeError("No data loaded. Check your data directory.")

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_subject_ids = np.array(all_subject_ids)

        return all_data, all_labels, all_subject_ids