import os
import numpy as np
import pandas as pd
import scipy
from scipy.io import wavfile
from scipy.fftpack import fft,rfft
from scipy.signal import get_window
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models.models import CNNLSTMModel
from config.config_parameters import DATASET_INFO,N_FFT,N_FILTERS
from config.config_model import HIDDEN_SIZE,NUM_LAYERS,NUM_CLASSES,DROPOUT_RATE
from config.config_train import BATCH_SIZE,LEARNING_RATE,EPOCHS
from config.config_parameters import TARGET_dB,SAMPLE_RATE,NOISE_LEVEL,NUM_SAMPLES,N_FFT
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# 创建 SummaryWriter 实例
writer = SummaryWriter('runs/experiment_3')
# 添加配置参数到 TensorBoard

config_str_1 = f"N_FFT: {N_FFT}, N_FILTERS: {N_FILTERS}, HIDDEN_SIZE: {HIDDEN_SIZE}, NUM_CLASSES: {NUM_CLASSES}, DROPOUT_RATE: {DROPOUT_RATE}, BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}, EPOCHS: {EPOCHS}"
config_str_2 = f"TARGET_dB: {TARGET_dB},SAMPLE_RATE: {SAMPLE_RATE}, NOISE_LEVEL: {NOISE_LEVEL}, NUM_SAMPLES: {NUM_SAMPLES}, N_FFT: {N_FFT},N_FILTERS: {N_FILTERS}"

writer.add_text('Config Parameters_1', config_str_1, 0)
writer.add_text('Config Parameters_2', config_str_2, 0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_mel(audio_path, n_fft, n_filters, SAMPLE_RATE):
    # Read the audio file
    sample_rate, signal = wavfile.read(audio_path)
    # signal = np.mean(signal, axis=1)  # Convert to mono if stereo

    # Frame size and stride 46560
    frame_size = 0.03
    frame_stride = 0.02
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))

    # Frame the signal
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start_idx = i * frame_step
        end_idx = min(start_idx + frame_length, signal_length)
        frames[i, :end_idx - start_idx] = signal[start_idx:end_idx]

    # Apply window function
    frames *= scipy.signal.windows.hamming(frame_length)

    # FFT and power spectrum
    mag_frames = np.absolute(rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * ((mag_frames) ** 2)

    # Mel-filterbanks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((n_fft + 1) * hz_points / sample_rate)

    fbank = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # Normalize
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    # Add an additional axis to indicate 'channel' for CNN input
    filter_banks = np.expand_dims(filter_banks, axis=0)

    return filter_banks

def read_dataset(csv_file_path, audio_base_path):
    df = pd.read_csv(csv_file_path)
    mfccs = []
    labels = []

    for _, row in df.iterrows():
        audio_file_rel_path = row['audio_filename']
        audio_file_abs_path = os.path.join(audio_base_path, audio_file_rel_path)
        mfccs.append(extract_mel(audio_file_abs_path,N_FFT,N_FILTERS,SAMPLE_RATE))
        labels.append(row[1:])

    X_tensors = torch.stack([torch.tensor(mfcc, dtype=torch.float32) for mfcc in mfccs])
    y_tensors = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels])

    return X_tensors, y_tensors

datasets_info = DATASET_INFO

X_tensors, y_tensors = [], []

for dataset in datasets_info:
    X, y = read_dataset(dataset["csv"], dataset["audio_base"])
    X_tensors.append(X)
    y_tensors.append(y)

X_tensor = torch.cat(X_tensors, dim=0)
y_tensor = torch.cat(y_tensors, dim=0)

X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42)

# 创建训练集和测试集的TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 创建训练集和测试集的DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_size = X_tensor.shape[2]
# 创建模型实例
model = CNNLSTMModel(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)# 创建一个虚拟输入张量
# batch_size 和 sequence_length 可以是任何合适的数值
dummy_input = torch.randn(BATCH_SIZE, 1, 49,N_FILTERS).to(device)  # [batch_size, channels, height, width]
# 将模型和虚拟输入添加到 TensorBoard
writer.add_graph(model, dummy_input)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate_model(model, data_loader, device):
    model.eval()
    predicted = []
    labels = []
    with torch.no_grad():
        for inputs, labels_batch in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted_batch = outputs.round()
            predicted.extend(predicted_batch.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
    return np.array(predicted), np.array(labels)

losses = []

for epoch in range(EPOCHS):
    total_loss = 0
    total_batches = 0

    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + total_batches)

    avg_loss = total_loss / total_batches
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}')
    writer.add_scalar('Average Training Loss', avg_loss, epoch)

    if (epoch + 1) % 10 == 0:
        train_predicted, train_labels = evaluate_model(model, train_loader, device)
        train_accuracy = accuracy_score(train_labels, train_predicted)
        train_f1 = f1_score(train_labels, train_predicted, average='weighted')

        test_predicted, test_labels = evaluate_model(model, test_loader, device)
        test_accuracy = accuracy_score(test_labels, test_predicted)
        test_f1 = f1_score(test_labels, test_predicted, average='weighted')

        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('F1 Score/Train', train_f1, epoch)
        writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
        writer.add_scalar('F1 Score/Test', test_f1, epoch)

torch.save(model, 'F:\\DLProject\\models\\model_03.pth')


#TEST
# Define a function to calculate accuracy
def calculate_accuracy(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

# Testing the model
model.eval()  # Set the model to evaluation mode
all_predicted = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted = outputs.round()  # Round to get binary output
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Converting lists to numpy arrays for metric calculation
all_predicted = np.array(all_predicted)
all_labels = np.array(all_labels)

# Calculating F1 score and Accuracy
f1 = f1_score(all_labels, all_predicted, average='weighted')
accuracy = calculate_accuracy(all_labels, all_predicted)
print(f"F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

writer.add_text('Test Metrics', f'F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}', 0)

# Error Analysis
errors = all_labels != all_predicted
error_distribution = np.sum(errors, axis=0) / len(errors)
print(f"Error Distribution (per class): {error_distribution}")
import matplotlib.pyplot as plt

# 已有的错误分析代码
errors = all_labels != all_predicted
error_distribution = np.sum(errors, axis=0) / len(errors)
print(f"Error Distribution (per class): {error_distribution}")

# 可视化错误分布
fig = plt.figure(figsize=(12, 6))
plt.bar(range(len(error_distribution)), error_distribution)
plt.xlabel('Class Label')
plt.ylabel('Error Rate')
plt.title('Error Distribution per Class')
plt.xticks(range(len(error_distribution)))
writer.add_figure('Error Distribution', fig, 0)
# plt.show()
writer.close()