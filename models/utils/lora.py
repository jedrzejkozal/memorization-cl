import torch
import torch.nn as nn
import math


def replace_conv_with_lora(model: nn.Module, lora_r: int, lora_alpha: float, lora_type='lora', skip_layers: list[str] = ('conv1',)):
    conv_names = list()
    out_channels = set()
    for name, module in model.named_modules():
        if type(module) == nn.Conv2d and name not in skip_layers:
            conv_names.append(name)
            out_channels.add(module.out_channels)
    min_out = min(list(out_channels))

    lora_types = {
        'lora': ConvLoRA,
        'lora_tucker': ConvLoRATucker,
        'loha': ConvLoHa,
    }
    layer_type = lora_types[lora_type]
    for name in conv_names:
        module = get_module(model, name)
        r = int(module.out_channels / min_out * lora_r)
        set_module(model, name, layer_type(module, r=r, alpha=lora_alpha))


def replace_lora_with_conv(model: nn.Module, lora_type='lora'):
    lora_types = {
        'lora': ConvLoRA,
        'lora_tucker': ConvLoRATucker,
        'loha': ConvLoHa,
    }
    layer_type = lora_types[lora_type]

    conv_names = list()
    for name, module in model.named_modules():
        if type(module) == layer_type:
            conv_names.append(name)

    for name in conv_names:
        module = get_module(model, name)
        set_module(model, name, module.get_merged())


def get_module(module: nn.Module, name: str):
    """getattr for pytorch modules, that takes nn.Sequntial into account"""
    layer_names = name.split('.')
    if len(layer_names) == 1:
        if type(module) == nn.Sequential:
            return module[0]
        else:
            return getattr(module, layer_names[0])

    remaining_names = '.'.join(layer_names[1:])
    if type(module) == nn.Sequential:
        layer_index = int(layer_names[0])
        return get_module(module[layer_index], remaining_names)
    else:
        return get_module(getattr(module, layer_names[0]), remaining_names)


def set_module(module: nn.Module, name: str, new_value: nn.Module):
    """setattr for pytorch modules, that takes nn.Sequntial into account"""
    layer_names = name.split('.')
    if len(layer_names) == 1:
        if type(module) == nn.Sequential:
            layer_index = int(layer_names[0])
            module[layer_index] = new_value
        else:
            setattr(module, layer_names[0], new_value)
        return

    remaining_names = '.'.join(layer_names[1:])
    if type(module) == nn.Sequential:
        layer_index = int(layer_names[0])
        next_module = module[layer_index]
        set_module(next_module, remaining_names, new_value)
    else:
        next_module = get_module(module, layer_names[0])
        set_module(next_module, remaining_names, new_value)


class ConvLoRA(nn.Module):
    """wrapper for Conv2d layer"""

    def __init__(self, conv_layer: nn.Conv2d, r=5, alpha=1) -> None:
        super().__init__()
        self.conv = conv_layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        if conv_layer.kernel_size[0] != conv_layer.kernel_size[1]:
            raise ValueError('kernel size not equal')
        self.kernel_size = conv_layer.kernel_size[0]

        self.r = r
        self.alpha = alpha

        device = conv_layer.weight.device
        self.conv_A = nn.Conv2d(self.in_channels, r, self.kernel_size, bias=False).to(device)
        self.use_bias = self.conv.bias is not None
        self.conv_B = nn.Conv2d(r, self.out_channels//self.conv.groups, kernel_size=1, bias=self.use_bias).to(device)
        self.scaling = self.alpha / self.r
        self.initialize()

        self.conv.weight.requires_grad = False

    def initialize(self):
        nn.init.kaiming_uniform_(self.conv_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_B.weight)

    def get_merged(self):
        A_matrix = self.conv_A.weight.data.view(self.r, self.in_channels * self.kernel_size * self.kernel_size)
        B_matrix = self.conv_B.weight.data.view(self.out_channels, self.r)
        self.conv.weight.data += self.scaling * (B_matrix @ A_matrix).view(self.conv.weight.shape)
        if self.use_bias:
            self.conv.bias.data += self.scaling * self.conv_B.bias.data
        return self.conv

    def forward(self, x):
        x_out = self.conv(x)
        x_out += self.scaling * self.conv_B(self.conv._conv_forward(x, self.conv_A.weight, bias=None))
        return x_out


def rebuild_tucker(t, wa, wb):
    rebuild2 = torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
    return rebuild2


class ConvLoRATucker(nn.Module):
    """wrapper for Conv2d layer"""

    def __init__(self, conv_layer: nn.Conv2d, r=5, alpha=1) -> None:
        super().__init__()
        self.conv = conv_layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        if conv_layer.kernel_size[0] != conv_layer.kernel_size[1]:
            raise ValueError('kernel size not equal')
        self.kernel_size = conv_layer.kernel_size[0]

        self.r = r
        self.alpha = alpha

        device = conv_layer.weight.device
        self.conv_A = nn.Conv2d(self.in_channels, r, 1, bias=False).to(device)
        self.conv_B = nn.Conv2d(r, self.out_channels//self.conv.groups, kernel_size=1, bias=False).to(device)
        self.conv_G = nn.Conv2d(r, r, kernel_size=self.kernel_size, bias=False).to(device)
        self.scaling = self.alpha / self.r
        self.initialize()

        self.conv.weight.requires_grad = False

    def initialize(self):
        nn.init.kaiming_uniform_(self.conv_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_B.weight)

    def get_merged(self):
        A_matrix = self.conv_A.weight.data.view(self.r, self.in_channels)
        B_matrix = self.conv_B.weight.data.view(self.out_channels, self.r)
        G_matrix = self.conv_G.weight.data

        self.conv.weight.data += rebuild_tucker(G_matrix, B_matrix.transpose(0, 1), A_matrix)
        return self.conv

    def forward(self, x):
        x_out = self.conv(x)
        x_out += self.scaling * self.conv_B(self.conv._conv_forward(self.conv_A(x), self.conv_G.weight, bias=None))
        # x_out += self.scaling * self.conv_B(self.conv._conv_forward(x, self.conv_A.weight, bias=None))
        return x_out


class ConvLoHa(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, r=5, alpha=1) -> None:
        super().__init__()
        self.conv = conv_layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        if conv_layer.kernel_size[0] != conv_layer.kernel_size[1]:
            raise ValueError('kernel size not equal')
        self.kernel_size = conv_layer.kernel_size[0]

        self.r = r
        self.alpha = alpha

        device = conv_layer.weight.device
        self.conv_A1 = nn.Conv2d(self.in_channels, r, self.kernel_size, bias=False).to(device)
        self.conv_B1 = nn.Conv2d(r, self.out_channels//self.conv.groups, kernel_size=1, bias=False).to(device)
        self.conv_A2 = nn.Conv2d(self.in_channels, r, self.kernel_size, bias=False).to(device)
        self.conv_B2 = nn.Conv2d(r, self.out_channels//self.conv.groups, kernel_size=1, bias=False).to(device)
        self.scaling = self.alpha / self.r
        self.initialize()

        self.conv.weight.requires_grad = False

    def initialize(self):
        nn.init.kaiming_uniform_(self.conv_A1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_B1.weight)
        nn.init.kaiming_uniform_(self.conv_A2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_B2.weight)

    def get_merged(self):
        A1_matrix = self.conv_A1.weight.data.view(self.r, self.in_channels * self.kernel_size * self.kernel_size)
        B1_matrix = self.conv_B1.weight.data.view(self.out_channels, self.r)
        A2_matrix = self.conv_A1.weight.data.view(self.r, self.in_channels * self.kernel_size * self.kernel_size)
        B2_matrix = self.conv_B1.weight.data.view(self.out_channels, self.r)
        self.conv.weight.data += self.scaling * ((B1_matrix @ A1_matrix) * (B2_matrix @ A2_matrix)).view(self.conv.weight.shape)
        return self.conv

    def forward(self, x):
        x_out = self.conv(x)
        x1 = self.conv_B1(self.conv._conv_forward(x, self.conv_A1.weight, bias=None))
        x2 = self.conv_B2(self.conv._conv_forward(x, self.conv_A2.weight, bias=None))
        x_out += self.scaling * x1 * x2
        return x_out
