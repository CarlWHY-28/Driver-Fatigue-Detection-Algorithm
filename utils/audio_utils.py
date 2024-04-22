import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine


def calculate_mfcc(sliding_window, rate, mfccs0):

    # 将新读取的数据添加到滑动窗口列表

    combined_audio = np.concatenate(sliding_window, axis=0)
    # 提取mfcc特征
    mfccs = librosa.feature.mfcc(y=combined_audio, sr=rate, n_mfcc=13)
    # 将mfcc0特征压缩到和mfccs一样的维度

    mfccs = np.pad(
        mfccs, ((0, 0), (0, max(0, mfccs0.shape[1] - mfccs.shape[1]))), 'constant')
    # 计算余弦相似度
    cosine_similarity = 1 - cosine(mfccs.flatten(), mfccs0.flatten())
    return cosine_similarity
