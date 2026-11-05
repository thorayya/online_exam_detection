import math
import random
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


PROJECT_DIR = Path("/content/drive/MyDrive/Online_Exam_Cheating")
VISION_DIR = PROJECT_DIR / "vision_fusion_data"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_files():

    splits = {}

    for split in ["train", "validation", "test"]:

        audio = np.load(
            VISION_DIR / f"{split}_aligned_audio_features.npy"
        )

        vision = np.load(
            VISION_DIR / f"{split}_aligned_vision_features.npy"
        )

        start_time = np.load(
            VISION_DIR / f"{split}_start_seconds.npy"
        )

        video_names = np.load(
            VISION_DIR / f"{split}_video_names.npy"
        )

        assert len(audio) == len(vision) == len(start_time) == len(video_names)

        splits[split] = {
            "audio": audio,
            "vision": vision,
            "start_time": start_time,
            "video_names": video_names
        }

    return splits


data = load_files()

train_audio = data["train"]["audio"]
train_vision = data["train"]["vision"]
train_start_time = data["train"]["start_time"]
train_vision_name = data["train"]["video_names"]

validation_audio = data["validation"]["audio"]
validation_vision = data["validation"]["vision"]
validation_start_time = data["validation"]["start_time"]
validation_vision_name = data["validation"]["video_names"]

test_audio = data["test"]["audio"]
test_vision = data["test"]["vision"]
test_start_time = data["test"]["start_time"]
test_vision_name = data["test"]["video_names"]


print("Train:", train_audio.shape, train_vision.shape)
print("Validation:", validation_audio.shape, validation_vision.shape)
print("Test:", test_audio.shape, test_vision.shape)

# ============================================================
# 3. ORIGINAL TRAIN ANNOTATIONS
# Ranges are one-based and inclusive.
# ============================================================

train_lengths = [
    960, 879, 856, 864, 1507, 943, 920, 1114,
    1063, 860, 1143, 826, 945, 913, 1537, 1098
]

label_ranges_first_train = [
    (873, 888, 1),
]#"joorablo"

label_ranges_second_train = [
    (829, 830, 1),
    (876, 877, 1),
]#"meowseph"

label_ranges_third_train = [
    (498, 505, 1),
    (508, 514, 1),
]#muath

label_ranges_forth_train = [
    (498, 505, 1),
    (508, 514, 1),
]#yousef

label_ranges_fifth_train = [
    (1, 27, 1), (37, 47, 1), (121, 139, 1),
    (181, 216, 1), (217, 226, 1), (227, 239, 1),
    (244, 256, 1), (271, 300, 1), (346, 358, 1),
    (381, 384, 1), (423, 437, 1), (442, 446, 1),
    (450, 461, 1), (465, 527, 1), (530, 534, 1),
    (541, 561, 1), (627, 633, 1), (635, 639, 1),
    (643, 669, 1), (675, 790, 1), (792, 832, 1),
    (914, 933, 1), (996, 998, 1), (1000, 1002, 1),
    (1003, 1011, 1), (1013, 1047, 1), (1054, 1060, 1),
    (1063, 1082, 1), (1086, 1090, 1), (1097, 1140, 1),
    (1171, 1192, 1), (1215, 1225, 1), (1227, 1240, 1),
    (1244, 1247, 1), (1282, 1344, 1), (1349, 1370, 1),
    (1380, 1395, 1), (1397, 1401, 1), (1406, 1506, 1),
]#yuqianyi

label_ranges_sixth_train = [
    (937, 939, 1), (941, 943, 1),
]#zach

label_ranges_seventh_train = [
    (683, 684, 1), (688, 690, 1),
    (693, 694, 1), (698, 718, 1),
]#xiyin

label_ranges_eighth_train = [
    (650, 670, 1), (674, 697, 1),
    (1095, 1099, 1), (1103, 1114, 1),
]#vaibhav

label_ranges_ninth_train = [
    (151, 164, 1), (175, 195, 1), (1058, 1059, 1),
]#vahid

label_ranges_eleventh_train = [
    (523, 524, 1), (528, 529, 1), (534, 543, 1),
    (855, 856, 1), (858, 859, 1),
]#mustaffa

label_ranges_twelveth_train = [
    (11, 15, 1), (162, 178, 1), (183, 188, 1),
    (226, 232, 1), (320, 331, 1), (333, 360, 1),
    (367, 380, 1), (381, 422, 1), (423, 450, 1),
    (453, 483, 1), (488, 493, 1), (553, 578, 1),
    (657, 662, 1), (664, 667, 1), (675, 685, 1),
    (693, 703, 1), (715, 752, 1), (833, 838, 1),
    (853, 868, 1), (873, 892, 1), (897, 901, 1),
    (904, 927, 1), (930, 950, 1), (953, 978, 1),
    (980, 981, 1), (1017, 1026, 1), (1040, 1043, 1),
    (1048, 1093, 1), (1096, 1143, 1),
]#wanguys

label_ranges_thirteenth_train = [
    (254, 255, 1), (755, 761, 1), (765, 768, 1),
]#yarub

label_ranges_forteenth_train = [
    (296, 321, 1), (399, 406, 1), (413, 414, 1),
    (419, 420, 1), (424, 450, 1), (453, 454, 1),
    (456, 457, 1), (460, 462, 1), (464, 468, 1),
    (471, 476, 1), (484, 496, 1), (500, 503, 1),
    (508, 511, 1), (514, 517, 1), (524, 526, 1),
    (529, 532, 1), (538, 543, 1), (545, 558, 1),
    (563, 574, 1), (579, 594, 1), (657, 658, 1),
    (688, 690, 1), (695, 697, 1), (700, 701, 1),
    (704, 714, 1), (722, 723, 1), (778, 796, 1),
    (805, 806, 1), (812, 813, 1), (819, 827, 1),
    (902, 909, 1), (912, 930, 1), (932, 936, 1),
]#zhoulian

label_ranges_fifteenth_train = [
    (171, 173, 1), (257, 258, 1), (262, 267, 1),
    (277, 282, 1), (290, 292, 1), (437, 439, 1),
    (458, 465, 1), (548, 549, 1), (564, 567, 1),
    (576, 581, 1), (664, 666, 1), (684, 685, 1),
    (690, 705, 1), (709, 730, 1),
]#mark

label_ranges_sixteenth_train = [
    (15, 18, 1), (21, 22, 1), (24, 26, 1),
    (30, 38, 1), (194, 270, 1), (348, 386, 1),
    (406, 412, 1), (415, 422, 1), (579, 584, 1),
    (595, 617, 1), (632, 642, 1), (1047, 1120, 1),
    (1121, 1181, 1), (1183, 1191, 1), (1193, 1224, 1),
    (1227, 1228, 1), (1230, 1250, 1), (1251, 1279, 1),
    (1358, 1372, 1), (1377, 1383, 1), (1391, 1396, 1),
    (1397, 1415, 1), (1433, 1435, 1), (1441, 1450, 1),
    (1454, 1477, 1), (1480, 1494, 1), (1497, 1504, 1),
    (1512, 1537, 1),
]#huangpi

label_ranges_seventeenth_train = [
    (1, 5, 1),
    (857, 860, 1),
    (861, 883, 1),
    (1087, 1088, 1),
    (1090, 1091, 1),
]#husseinhijazi

train_ranges = [
    label_ranges_first_train,
    label_ranges_second_train,
    label_ranges_third_train,
    label_ranges_forth_train,
    label_ranges_fifth_train,
    label_ranges_sixth_train,
    label_ranges_seventh_train,
    label_ranges_eighth_train,
    label_ranges_ninth_train,
    label_ranges_eleventh_train,
    label_ranges_twelveth_train,
    label_ranges_thirteenth_train,
    label_ranges_forteenth_train,
    label_ranges_fifteenth_train,
    label_ranges_sixteenth_train,
    label_ranges_seventeenth_train,
]


# ============================================================
# 4. ORIGINAL VALIDATION ANNOTATIONS
# ============================================================

validation_lengths = [1055, 1366, 1061, 1162]

label_ranges_first = [
    (1, 5, 1), (8, 10, 1), (17, 21, 1),
    (170, 179, 1), (265, 274, 1), (785, 821, 1),
    (823, 829, 1), (841, 854, 1), (856, 862, 1),
    (875, 877, 1), (880, 881, 1), (1034, 1035, 1),
    (1039, 1044, 1), (1049, 1054, 1),
]#alhashe

label_ranges_second = [
    (2, 5, 1), (14, 23, 1), (393, 395, 1),
    (544, 550, 1), (554, 566, 1), (569, 583, 1),
    (595, 614, 1), (618, 620, 1), (627, 633, 1),
    (985, 1022, 1), (1033, 1042, 1), (1053, 1070, 1),
    (1080, 1081, 1), (1359, 1362, 1),
]#makiluke

label_ranges_third = [
    (476, 524, 2),
]#chelingpi

label_ranges_forth = [
    (1, 8, 1), (14, 19, 1), (124, 134, 1),
    (290, 292, 1), (295, 312, 1), (329, 333, 1),
    (392, 396, 1), (499, 540, 1), (810, 857, 1),
    (980, 992, 1), (1139, 1149, 1), (1152, 1161, 1),
]#bajajpak

validation_ranges = [
    label_ranges_first,
    label_ranges_second,
    label_ranges_third,
    label_ranges_forth,
]


# ============================================================
# 5. ORIGINAL TEST ANNOTATIONS
# ============================================================

test_lengths = [736, 928, 942, 1457]

label_ranges_second_test = [
    (1, 7, 1), (151, 156, 1), (274, 288, 1),
    (304, 369, 1), (513, 568, 1), (606, 613, 1),
    (674, 676, 1), (679, 682, 1), (685, 699, 1),
    (707, 714, 1), (732, 735, 1),
]#chuangy

label_ranges_third_test = [
    (3, 4, 1), (346, 349, 1), (358, 364, 1),
    (854, 858, 1), (860, 862, 1), (865, 867, 1),
    (870, 871, 1), (922, 924, 1),
]#cool

label_ranges_first_test = [
    (1, 2, 1), (3, 9, 1), (15, 25, 1),
    (278, 286, 1), (289, 293, 1), (425, 433, 1),
    (438, 445, 2), (449, 452, 1), (458, 463, 1),
    (466, 467, 1), (518, 523, 1), (554, 556, 1),
    (843, 873, 1), (874, 888, 1), (901, 906, 1),
    (937, 941, 1),
]#dhingra

label_ranges_forth_test = [
    (1, 5, 1), (14, 16, 1), (189, 212, 1),
    (328, 363, 1), (547, 555, 1), (561, 564, 1),
    (569, 583, 1), (682, 688, 1), (697, 700, 1),
    (707, 711, 1), (748, 831, 1), (833, 846, 1),
    (941, 1050, 1), (1084, 1105, 1), (1109, 1130, 1),
    (1133, 1148, 1), (1151, 1153, 1), (1157, 1160, 1),
    (1164, 1178, 1), (1228, 1235, 1), (1248, 1282, 1),
    (1317, 1326, 1), (1345, 1357, 1), (1386, 1420, 1),
    (1407, 1414, 1), (1417, 1443, 1),
]#guolingh

test_ranges = [
    label_ranges_first_test,
    label_ranges_second_test,
    label_ranges_third_test,
    label_ranges_forth_test,
]


# ============================================================
# 3. FEATURE FUSION
# ============================================================

train_tensor = torch.cat((
    torch.from_numpy(train_vision).float(),
    torch.from_numpy(train_audio).float()
), dim=2)

validation_tensor = torch.cat((
    torch.from_numpy(validation_vision).float(),
    torch.from_numpy(validation_audio).float()
), dim=2)

test_tensor = torch.cat((
    torch.from_numpy(test_vision).float(),
    torch.from_numpy(test_audio).float()
), dim=2)


print("Train:", train_tensor.shape)
print("Validation:", validation_tensor.shape)
print("Test:", test_tensor.shape)


# ============================================================
# 4. POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


# ============================================================
# 5. TRANSFORMER MODEL
# ============================================================

class Transformer(nn.Module):

    def __init__(self, input_dim=31, k=64, heads=8,
                 depth=1, num_classes=1):

        super().__init__()

        self.input_projection = nn.Linear(input_dim, k)

        self.pos_enc = PositionalEncoding(
            d_model=k,
            dropout=0.1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=k,
            nhead=heads,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.classification_head = nn.Linear(k, num_classes)

    def forward(self, x):

        x = self.input_projection(x)

        x = self.pos_enc(x)

        x = self.transformer_encoder(x)

        logits = self.classification_head(x)

        logits = logits.squeeze(-1)

        return logits


# ============================================================
# 6. TIMESTEP LABELS
# ============================================================

def make_timestep_labels(names, start_times, ranges, window_size=10):

    ordered_names = list(dict.fromkeys(names.tolist()))

    assert len(ordered_names) == len(ranges)

    range_by_video = dict(zip(ordered_names, ranges))

    labels = np.zeros(
        (len(names), window_size),
        dtype=np.float32
    )

    for i, (name, start) in enumerate(zip(names, start_times)):

        times = int(round(start)) + np.arange(window_size) + 1

        for start_range, end_range, label in range_by_video[name]:

            if label > 0:
                labels[i, (times >= start_range) & (times <= end_range)] = 1.0

    return labels


train_labels = make_timestep_labels(
    train_vision_name, train_start_time, train_ranges
)

validation_labels = make_timestep_labels(
    validation_vision_name, validation_start_time, validation_ranges
)

test_labels = make_timestep_labels(
    test_vision_name, test_start_time, test_ranges
)


print("Train labels:", train_labels.shape)
print("Validation labels:", validation_labels.shape)
print("Test labels:", test_labels.shape)

print("Train distribution:", np.unique(train_labels, return_counts=True))
print("Validation distribution:", np.unique(validation_labels, return_counts=True))
print("Test distribution:", np.unique(test_labels, return_counts=True))


# ============================================================
# 7. DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    train_tensor,
    torch.tensor(train_labels)
)

validation_dataset = TensorDataset(
    validation_tensor,
    torch.tensor(validation_labels)
)

test_dataset = TensorDataset(
    test_tensor,
    torch.tensor(test_labels)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# 8. SEED AND LOSS
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

loss_fn = nn.BCEWithLogitsLoss()


# ============================================================
# 9. TRAINING
# ============================================================

def train(model, loss_fn, train_loader,
          n_iter=30, learning_rate=1e-4,
          test_loader=None, device="cpu"):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    train_loss = []
    test_loss = []

    for epoch in range(n_iter):

        model.train()
        iter_train_loss = []

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device).float()

            out = model(inputs)
            loss = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_train_loss.append(loss.item())

        train_loss.append(statistics.mean(iter_train_loss))

        if test_loader is not None:

            model.eval()
            iter_loss_test = []

            with torch.no_grad():

                for inputs, labels in test_loader:

                    inputs = inputs.to(device)
                    labels = labels.to(device).float()

                    out = model(inputs)
                    loss = loss_fn(out, labels)

                    iter_loss_test.append(loss.item())

            test_loss.append(statistics.mean(iter_loss_test))

        print(
            f"Epoch {epoch+1}/{n_iter} | "
            f"Train Loss: {train_loss[-1]:.4f} | "
            f"Validation Loss: {test_loss[-1]:.4f}"
            if test_loader is not None
            else
            f"Epoch {epoch+1}/{n_iter} | "
            f"Train Loss: {train_loss[-1]:.4f}"
        )

    return train_loss, test_loss


# ============================================================
# 10. MODEL TRAINING
# ============================================================

model = Transformer(
    input_dim=31,
    k=64,
    heads=8,
    depth=1,
    num_classes=1
).to(DEVICE)


sample = train_tensor[:32].to(DEVICE)

print("Model output:", model(sample).shape)


train_loss, validation_loss = train(
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader,
    n_iter=5,
    learning_rate=1e-4,
    test_loader=validation_loader,
    device=DEVICE
)


# ============================================================
# 11. TEST EVALUATION
# ============================================================

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(DEVICE)

        logits = model(inputs)

        probabilities = torch.sigmoid(logits)

        predictions = (probabilities >= 0.5).int()

        all_predictions.extend(
            predictions.cpu().numpy().flatten()
        )

        all_labels.extend(
            labels.numpy().flatten()
        )


print("F1:", f1_score(all_labels, all_predictions))
print("Precision:", precision_score(all_labels, all_predictions))
print("Recall:", recall_score(all_labels, all_predictions))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))
