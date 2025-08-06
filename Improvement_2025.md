### 模型改进思路: CRNN (卷积循环神经网络)

旧版模型直接使用 TTS 网络中的隐变量 `z` 作为输入，这依赖于 `z` 能够充分捕捉所有与表情相关的语音特征。一种更直接、可能更有效的方法是直接从音频的梅尔频谱图（mel-spectrogram）中学习。这种方法将问题转化为一个标准的音频到序列的任务，可以使用经典的卷积循环神经网络（CRNN）架构。

以下是基于 PyTorch 的 CRNN 模型修改步骤，重点使用 GRU 来处理时序信息：

1.  **输入调整**:

    - 模型的输入不再是 `z` (`[B, C, T]`)，而是音频的梅尔频谱图。
    - 输入形状为 `[B, 1, n_mels, n_frames]` (分别对应 批次, 通道, 频率, 时间)。这里我们以 `Conv2D` 为例，因为它能同时处理时间和频率维度上的局部特征。

2.  **卷积特征提取 (CNN Front-end)**:

    - 在模型前端添加几层 2D 卷积层 (`nn.Conv2d`) 来从频谱图中提取高级特征。
    - 典型的结构是 `Conv2d` -> `BatchNorm2d` -> `LeakyReLU` -> `MaxPool2d` 的堆叠。这可以有效降低特征图的维度，同时扩大感受野。

    ```python
    # 伪代码
    self.cnn = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(2), # (B, 32, n_mels/2, n_frames/2)

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(2) # (B, 64, n_mels/4, n_frames/4)
    )
    ```

3.  **Reshape**:

    - 将 CNN 模块提取的特征图从 `[B, C, H, W]` 调整为适合循环层处理的 `[B, T, F]` 格式。
    - `T` 代表时间序列长度, `F` 代表每个时间步的特征维度。

    ```python
    # x is output from self.cnn
    B, C, H, W = x.shape
    x = x.permute(0, 3, 1, 2) # [B, W, C, H]
    x = x.reshape(B, W, C * H) # [B, T, F], where T=W, F=C*H
    ```

4.  **时序特征建模 (GRU)**:

    - 使用 `nn.GRU` 层来捕捉特征序列中的时间依赖关系。
    - 使用 `bidirectional=True` 的双向 GRU 通常能获得更好的性能，因为它能同时考虑过去和未来的上下文。
    - GRU 的输出是每个时间步的隐藏状态。如果只需要一个最终的特征向量来代表整个序列（例如，用于分类或单帧表情预测），可以只取最后一个时间步的输出，或者对所有时间步的输出进行池化（如 `GlobalAveragePooling1D`）。

    ```python
    # 伪代码
    self.gru = nn.GRU(input_size=C*H, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
    # x from reshape
    x, _ = self.gru(x)
    ```

5.  **输出层 (Output Head)**:
    _ 在 GRU 之后，连接一个或多个全连接层 (`nn.Linear`)，将 GRU 的输出映射到最终的 ARKit 表情参数维度。
    _ 最后一层使用 `Sigmoid` 激活函数，将输出值归一化到 `[0, 1]` 范围。

        ```python
        # 伪代码
        # x is output from self.gru
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128), # *2 because of bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 61), # 61 is n_arkit_outputs
            nn.Sigmoid()
        )
        # 如果GRU的return_sequences=True，则需要对时间维度处理
        # 例如取最后一个时间步的输出: x = x[:, -1, :]
        y = self.fc(x)
        ```

    这种 CRNN 架构的优势在于它解耦了特征提取（CNN）和时序建模（GRU），使得模型结构更清晰，并且能更有效地从原始音频信号中学习面部动画。
