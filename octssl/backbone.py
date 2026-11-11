import copy
import math
import torch
from torch import nn
ACT2FN={"gelu":torch.nn.functional.gelu,"relu":torch.nn.functional.relu}
def swish(x):
    return x * torch.sigmoid(x)


class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.two_conv3x3(1, 64, 64)

        self.conv2 = self.two_conv3x3(64, 128, 128)

        self.conv3 = self.two_conv3x3(128, 256, 256)

        self.conv4 = self.two_conv3x3(256, 512, 512)

        self.mp = nn.MaxPool2d(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        features = []
        conv1 = self.conv1(x)
        pool1 = self.mp(conv1)
        features.append(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.mp(conv2)
        features.append(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.mp(conv3)
        features.append(conv3)

        conv4 = self.conv4(pool3)
        features.append(conv4)

        return conv4, features[::-1]

    def two_conv3x3(self, channels1, channels2, channels3):
        return nn.Sequential(
            nn.Conv2d(channels1, channels2, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(channels2),
            nn.ReLU(True),
            nn.Conv2d(channels2, channels3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(channels3),
            nn.ReLU(True),
        )


class Attention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):


    def __init__(self, config, img_size, in_channels=512):
        super().__init__()
        self.hybrid = None
        self.config = config

        patch_size = (2, 2)
        n_patches = int((img_size[0]/patch_size[0]**4)*(img_size[1]/patch_size[1]**4))
        self.hybrid = True

        self.hybrid_model = UNetEncoder()
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, skip_channels=0, use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, upsampling=1
    ):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True,
            ):
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super().__init__(conv, bn, relu)


class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.img_size = img_size
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True,
            )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):
                skip_channels[3-i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(self.img_size[0]/2**4), int(self.img_size[1]/2**4)
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


class TransUNet_decode(nn.Module):
    def __init__(self, config, img_size=(128,1024), num_classes=9, zero_head=False, vis=False,in_channels=512):
        super().__init__()
        self.config = config
        patch_size = (2, 2)
        n_patches = int((img_size[0] / patch_size[0] ** 4) * (img_size[1] / patch_size[1] ** 4))
        self.hybrid_model = UNetEncoder()
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        self.encoder = Encoder(config, vis)
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config, img_size)
        self.feature_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['feature_num'],
            kernel_size=3,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=config['feature_num'],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,features):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        encoded, attn_weights = self.encoder(embeddings)
        x = self.decoder(encoded, features)
        feature1 = self.feature_head(x)
        feature2 = self.segmentation_head(feature1)

        return feature1,feature2


class TransUNet_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.segmentation_head = SegmentationHead(
            in_channels=config['feature_num'],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        result = self.segmentation_head(x)
        return result


class UNet_encode(nn.Module):
    def __init__(self, config, img_size=(1024, 128), num_classes=4, zero_head=False, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.unet_encode = UNetEncoder()

    def forward(self, x):
        conv4, features = self.unet_encode(x)
        return conv4, features
