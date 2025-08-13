import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return out.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 使用CBAM代替单纯的通道注意力
        self.att = CBAM(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.att(out)  # 应用注意力

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return F.relu(out)


class EnhancedAudioClassifier(nn.Module):
    def __init__(self, frames, bins, class_num,
                 enable_multimodal=False, num_locations=10, location_embedding_dim=16,
                 time_feature_dim=8, time_mapping_dim=16):
        super(EnhancedAudioClassifier, self).__init__()

        self.enable_multimodal = enable_multimodal
        self.frames = frames
        self.bins = bins

        # 增强初始特征提取
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 增强残差块结构
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 改进池化机制
        self.attention_pool = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )

        # 多模态处理（如果启用）
        if enable_multimodal:
            self.location_embedding = nn.Embedding(num_locations, location_embedding_dim)
            self.time_projection = nn.Sequential(
                nn.Linear(time_feature_dim, time_mapping_dim),
                nn.ReLU(),
                nn.Linear(time_mapping_dim, time_mapping_dim)
            )
            # 增强特征融合
            self.feature_fusion = nn.Sequential(
                nn.Linear(512 + location_embedding_dim + time_mapping_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        else:
            self.location_embedding = None
            self.time_projection = None
            self.feature_fusion = None

        # 增强分类器
        self.fc = nn.Sequential(
            nn.Linear(512 if not enable_multimodal else 256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, class_num)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, locations=None, time_features=None):
        # 输入x的形状: [batch_size, 1, frames, bins]

        # 增强的前端处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 通过深度残差网络
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 改进的池化机制
        attn_weights = self.attention_pool(x)
        attn_pooled = torch.sum(x * attn_weights, dim=(2, 3))
        avg_pooled = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        max_pooled = F.adaptive_max_pool2d(x, (1, 1)).squeeze()

        # 三重特征融合
        audio_features = attn_pooled + avg_pooled + max_pooled

        # 多模态融合
        if self.enable_multimodal and locations is not None and time_features is not None:
            loc_emb = self.location_embedding(locations)
            time_emb = self.time_projection(time_features)
            combined = torch.cat([audio_features, loc_emb, time_emb], dim=1)
            features = self.feature_fusion(combined)
        else:
            features = audio_features

        # 分类
        out = self.fc(features)

        return out


if __name__ == "__main__":
    # 测试用例
    model = EnhancedAudioClassifier(frames=500, bins=64, class_num=10,
                            enable_multimodal=True, num_locations=10,
                            location_embedding_dim=16, time_feature_dim=8,
                            time_mapping_dim=16)

    # 音频输入
    audio_input = torch.randn(16, 1, 500, 64)  # [batch, 1, frames, bins]
    # 多模态输入
    locations = torch.randint(0, 10, (16,))  # 16个样本，每个样本一个位置ID
    time_features = torch.randn(16, 8)  # 16个样本，每个样本8维时间特征

    output = model(audio_input, locations, time_features)
    print(output.size())  # 应该输出 torch.Size([16, 10])