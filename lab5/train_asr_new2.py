import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import torch.nn.functional as F
import re
import os
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
import glob
import time
from torch.amp import GradScaler, autocast
from torchaudio.transforms import MelSpectrogram, Resample
def levenshtein_distance(ref, hyp):
    """
    计算两个字符串的 Levenshtein 距离
    :param ref: 参考文本
    :param hyp: 假设文本
    :return: Levenshtein 距离
    """
    m = len(ref)
    n = len(hyp)
    dp = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[m][n]


def cer_single(reference, hypothesis):
    """
    计算字错误率
    :param reference: 参考文本
    :param hypothesis: 假设文本
    :return: 字错误率
    """
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def cer_multiple(refs, hyps):
    ## Start
    total_characters = 0
    total_distance = 0
    for ref, hyp in zip(refs, hyps):
        # ref 和 hyp 通常是 (1, seq_len) 的张量
        ref_seq = ref[0].cpu().numpy()
        hyp_seq = hyp[0].cpu().numpy()
        
        # 过滤掉填充值（假设 2666 是填充值，根据 collate_fn）和 <blank>(0), <eos>
        ref_filtered = [c for c in ref_seq if c != 2666 and c != 0 and c != len(char2id)-1]
        hyp_filtered = [c for c in hyp_seq if c != 2666 and c != 0 and c != len(char2id)-1]
        
        distance = levenshtein_distance(ref_filtered, hyp_filtered)
        total_distance += distance
        total_characters += len(ref_filtered)
    
    return total_distance / total_characters if total_characters > 0 else 0
    ## End

def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")

class ThchsData(Dataset):
    def __init__(self, wav_dir, sample_rate=16000):
        """
        Args:
            wav_dir (str): 音频文件夹路径
            transcription_dir (str): 转录文本文件夹路径
            sample_rate (int): 重采样的目标采样率
            transform (callable, optional): 音频和文本的预处理操作
        """
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate
        self.transcriptions={}
        # 获取所有音频文件
        
        self.texts = []
        self.paths = []
        text_paths = glob.glob(self.wav_dir)
        for path in text_paths:
            with open(path, 'r', encoding='utf8') as fr:
                lines = fr.readlines()
                line = lines[0].strip('\n').replace(' ', '').replace('../data','/home/scc/pb24000031/HW11/data_thchs30/data') # 注: 需要修改此处路径
                with open(line, 'r', encoding='utf8') as fr2:
                    lines2 = fr2.readlines()
                    self.texts.append(lines2[0].strip('\n').replace(' ', ''))
                    self.paths.append(path.rstrip('.trn'))
        chars = {}
        for text in self.texts:
            for c in text:
                chars[c] = chars.get(c, 0) + 1

        chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
        chars = [char[0] for char in chars]
        
        # print(len(chars), chars[:100])
        self.char2id =torch.load('/home/scc/pb24000031/HW11/char2id.pth') # 注: 需要修改此处路径
        self.id2char =torch.load('/home/scc/pb24000031/HW11/id2char.pth') # 注: 需要修改此处路径
        new_char2id = {char: id + 1 for char, id in self.char2id.items()}
        self.char2id = new_char2id

        # 更新 self.id2char 字典
        new_id2char = {id + 1: char for id, char in self.id2char.items()}
        self.id2char = new_id2char
        self.blank_index=0
        
        self.char2id['<blank>'] = self.blank_index
        self.id2char[self.blank_index] = '<blank>'
        self.eos_index=len(self.char2id)
        self.char2id['<eos>'] = self.eos_index
        self.id2char[self.eos_index]='<eos>'
        # print(len(self.char2id))
        # exit()
        # self.char2id = {c: i for i, c in enumerate(chars)}
        # self.id2char = {i: c for i, c in enumerate(chars)}
        # torch.save(self.char2id,'char2id.pth')
        # torch.save(self.id2char,'id2char.pth')
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        wav_path = self.paths[idx]
        # wav_path = os.path.join(self.wav_dir, wav_file)
        # 加载音频
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # 如果采样率不符合要求，进行重采样
        if sample_rate != self.sample_rate:
            resampler = Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        # print(transcription)
        # 应用预处理操作（如Mel Spectrogram）
        path1=wav_path.replace('train/','mfcc/').replace('wav','npy').replace('dev/','mfcc/').replace('test/','mfcc/')
        fea=torch.FloatTensor(np.load(path1))
        # phone=torch.LongTensor([self.char2id[c] for c in self.texts[idx]])
        phone_list = []  # 用于存储处理后的字符表示

        for i in range(len(self.texts[idx])):
            char = self.texts[idx][i]
            if char not in self.char2id:
                continue
            phone_list.append(self.char2id[char])  # 将字符映射为对应的id
            
            # 如果是最后一个字符，则跳出循环
            if i < len(self.texts[idx]) - 1 and self.texts[idx][i] == self.texts[idx][i + 1]:
                phone_list.append(0)  # 如果当前字符和下一个字符相同，则在中间插入 0

        # 将处理后的 phone_list 转换为 LongTensor
        phone = torch.LongTensor(phone_list)
        transcription=self.texts[idx]
        return fea, phone,waveform,transcription
def collate_fn(batch):
    # 获取音频和文本数据
    feas, phones ,waveform,transcription= zip(*batch)
    # print(transcriptions.shape)
    # print(transcriptions.shape)
    input_lengths=[]
    label_lengths=[]
    # 找到最大长度的音频（填充时用到）
    max_fea_len = max([fea.size(0) for fea in feas])
    for fea in feas:
        input_lengths.append(fea.size(0))
    # 填充音频
    padded_feas = []
    for fea in feas:
        fea=fea.transpose(0,1)
        # print(fea.shape)
        padding = max_fea_len - fea.size(1)
        padded_feas.append(torch.nn.functional.pad(fea, (0, padding)))
    
    # 找到最长文本的长度
    max_phone_len = max([len(phone) for phone in phones])
    for phone in phones:
        label_lengths.append(len(phone)+1) 
    # 填充文本
    padded_phones = []
    
    max_phone_len = max([phone.size(0) for phone in phones])
    for phone in phones:
        # 假设文本是字符级的，可以直接填充空格
        padding = max_phone_len - phone.size(0)+1
        padded_phones.append(torch.nn.functional.pad(phone, (0, padding), value=2666))
    
    # 转换为张量
    padded_fea = torch.stack(padded_feas, dim=0)
    # print(padded_fea.shape)
    padded_phones = torch.stack(padded_phones, dim=0)
    
    return padded_fea, padded_phones,input_lengths,label_lengths



class SimplifiedASRModel(nn.Module):
    def __init__(self, mfcc_dim, num_classes, hidden_size=128, num_layers=2):
        super(SimplifiedASRModel, self).__init__()
        # 添加 GRU 层
        self.gru = nn.GRU(mfcc_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 调整维度以适应 GRU 输入 [batch_size, seq_len, input_size]
        x = x.permute(0, 2, 1)

        # 通过 GRU 层
        output, _ = self.gru(x)

        # 通过全连接层
        y_pred = self.fc(output)

        # 调整维度以适应后续 CTCLoss 计算 [seq_len, batch_size, num_classes]
        y_pred = y_pred.permute(1, 0, 2)

        return y_pred
class ResBlock(nn.Module):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(filters, filters, kernel_size, stride=1, padding='same', dilation=dilation_rate)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, 1, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)

    def forward(self, x):
        hf = torch.tanh(self.bn1(self.conv1(x)))
        hg = torch.sigmoid(self.bn1(self.conv1(x)))
        h0 = hf * hg
        ha = torch.tanh(self.bn2(self.conv2(h0)))
        hs = torch.tanh(self.bn2(self.conv2(h0)))
        return ha + x, hs

class ResBlock(nn.Module):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(filters, filters, kernel_size, stride=1, padding='same', dilation=dilation_rate)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, 1, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        hf = self.leaky_relu(self.bn1(self.conv1(x)))
        hg = self.leaky_relu(self.bn1(self.conv1(x)))
        h0 = hf * hg
        ha = self.leaky_relu(self.bn2(self.conv2(h0)))
        hs = self.leaky_relu(self.bn2(self.conv2(h0)))
        return ha + x, hs

class ASRModel(nn.Module):
    def __init__(self, mfcc_dim, num_blocks, filters, num_classes, hidden_size=128, num_layers=2):
        super(ASRModel, self).__init__()
        self.conv1 = nn.Conv1d(mfcc_dim, filters, 3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.blocks = []
        for i in range(num_blocks):
            for r in [1, 2, 4]:
                self.blocks.append(ResBlock(filters, 7, r))
        self.blocks = nn.ModuleList(self.blocks)
        self.conv2 = nn.Conv1d(filters, filters, 3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)
        self.dropout2 = nn.Dropout(0.2)

        # 添加GRU层
        self.gru = nn.GRU(filters, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = self.leaky_relu(self.bn1(self.conv1(x)))
        h0 = self.dropout1(h0)
        shortcut = []
        for block in self.blocks:
            h0, s = block(h0)
            shortcut.append(s)
        h1 = self.leaky_relu(sum(shortcut))
        h1 = self.leaky_relu(self.bn2(self.conv2(h1)))
        h1 = self.dropout2(h1)

        # 调整维度以适应GRU输入 [batch_size, seq_len, input_size]
        h1 = h1.permute(0, 2, 1)

        # 通过GRU层
        output, _ = self.gru(h1)

        # 通过全连接层
        y_pred = self.fc(output)

        # 调整维度以适应后续CTCLoss计算 [seq_len, batch_size, num_classes]
        y_pred = y_pred.permute(1, 0, 2)

        return y_pred

epochs = 10
num_blocks = 3
filters = 128
char2id=torch.load('/home/scc/pb24000031/HW11/char2id.pth')
id2char=torch.load('/home/scc/pb24000031/HW11/id2char.pth')
new_char2id = {char: id + 1 for char, id in char2id.items()}
char2id = new_char2id

        # 更新 self.id2char 字典
new_id2char = {id + 1: char for id, char in id2char.items()}
id2char = new_id2char
id2char[0]='<blank>'
index=len(id2char)
id2char[index]='<eos>'
char2id['<blank>']=0
char2id['<eos>']=index
num_classes = len(char2id)
mfcc_dim = 13
model = ASRModel(mfcc_dim, num_blocks, filters, num_classes).cuda()
st=torch.load('/home/scc/pb24000031/HW11/g_0612_0.601')['model']
# for name, param in model.named_parameters():
#     if name in st:
#         param.requires_grad = False
#     else:
#         param.requires_grad = True
model.load_state_dict(st)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss()

train_losses = []
valid_losses = []
batch_size = 32

train_path='/home/scc/pb24000031/HW11/data_thchs30/train/*.trn'
dataset = ThchsData(train_path, sample_rate=16000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

dev_path='/home/scc/pb24000031/HW11/data_thchs30/dev/*.trn'
dataset = ThchsData(dev_path, sample_rate=16000)
dev_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

test_path='/home/scc/pb24000031/HW11/data_thchs30/test/*.trn'
dataset = ThchsData(test_path, sample_rate=16000)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    time_start=time.time()
    for batch in dataloader:
        
        fea, labels, input_lengths, label_lengths = batch
        optimizer.zero_grad()
        Y_pred = model(fea.cuda())
        Y_pred = Y_pred.log_softmax(2)

        ## Start
        input_lengths_tensor = torch.IntTensor(input_lengths)
        label_lengths_tensor = torch.IntTensor(label_lengths)

        # 计算损失
        # 注意：labels 包含填充值，CTCLoss 需要知道实际 label 的长度
        loss = criterion(Y_pred, labels.cuda(), input_lengths_tensor, label_lengths_tensor)
        ## End

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        Y_pred=Y_pred.permute(1, 0, 2)
        Y_pred=torch.argmax(Y_pred, dim=2).squeeze(0)
        non_blank_sequence = [num for num in Y_pred[0]]
    train_loss /= len(dataloader) // batch_size
    train_losses.append(train_loss)
    # torch.save(model.state_dict(),'final_asr.pth')
    time_end=time.time()
    elapsed_time=time_end-time_start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"第 {epoch+1} 轮花费了 {minutes} 分 {seconds:.2f} 秒")
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
    if (epoch+1)%1==0:
      devlabels=[]
      devpre=[]
      for batch in dev_dataloader:
          model.eval()
          fea, labels, input_lengths, label_lengths = batch
          with torch.no_grad():
              Y_pred = model(fea.cuda() )
          Y_pred=Y_pred.permute(1, 0, 2)
          Y_pred=torch.argmax(Y_pred, dim=2).squeeze(0)
          non_blank_sequence = [num for num in Y_pred]
          final_sequence = []
          for i, num in enumerate(non_blank_sequence):
                      if i == 0 or (num != non_blank_sequence[i - 1]&num!=0):
                          final_sequence.append(num)
          new_final_sequence = []
          for t in final_sequence:
              if t.dim() == 0:
                  t = t.unsqueeze(0)  # 将零维张量转换为一维张量
              new_final_sequence.append(t)
  
          result_tensor = torch.cat(new_final_sequence, dim=0)
          devlabels.append(labels)
          devpre.append(result_tensor.unsqueeze(0))
          # print(devlabels)
      
      # print(final_sequence)
      final_text = ''.join([id2char[num.item()] for num in final_sequence])
      print(final_text)
      true_label=''.join([id2char[num.item()] for num in labels[0]])
      print(true_label)
      error_rate = cer_multiple(devlabels, devpre)
      print(f"字错误率 (CER): {error_rate * 100:.2f}%")

## Start
print("开始在测试集上进行最终评估...")
model.eval()
test_labels = []
test_preds = []

with torch.no_grad():
    for batch in test_dataloader:
        fea, labels, input_lengths, label_lengths = batch
        Y_pred = model(fea.cuda())
        
        # 解码预测结果 (Greedy Search)
        Y_pred = Y_pred.permute(1, 0, 2)
        predicted_ids = torch.argmax(Y_pred, dim=2).squeeze(0)
        
        # CTC 后处理：合并重复字符并移除 <blank>(0)
        final_sequence = []
        for i, num in enumerate(predicted_ids):
            if num != 0: # 移除 blank
                if i == 0 or num != predicted_ids[i-1]:
                    final_sequence.append(num)
        
        test_labels.append(labels)
        test_preds.append(torch.tensor(final_sequence).unsqueeze(0))

# 计算最终 CER
final_test_cer = cer_multiple(test_labels, test_preds)
print(f"测试集最终字错误率 (Final Test CER): {final_test_cer * 100:.2f}%")
## End