import torch
import torch.nn.functional as F

class StrideHD:
    def __init__(self, window_size=(4,4), stride=2, pool_size=(2,2)):
        self.window_size = window_size
        self.stride = stride
        self.pool_size = pool_size

    def extract_and_pool(self, x):
        """
        x: [B, C, H, W]
        Retorna: [B, num_windows, C, pooled_H, pooled_W]
        """
        B, C, H, W = x.shape

        # 1) Extrai as janelas diretamente usando unfold
        windows = x.unfold(2, self.window_size[0], self.stride) \
                   .unfold(3, self.window_size[1], self.stride)
        # windows: [B, C, num_h, num_w, win_H, win_W]

        # 2) Combina dimensões num_h e num_w para formar num_windows
        B, C, num_h, num_w, win_H, win_W = windows.shape
        windows = windows.permute(0, 2, 3, 1, 4, 5)  # [B, num_h, num_w, C, win_H, win_W]
        windows = windows.contiguous().view(B, num_h*num_w, C, win_H, win_W)  # [B, num_windows, C, win_H, win_W]

        # 3) Aplicar max pooling de forma eficiente
        # Transformamos [B*num_windows, C, win_H, win_W] para aplicar F.max_pool2d
        pooled = F.max_pool2d(windows.view(-1, C, win_H, win_W), self.pool_size)
        pooled_H, pooled_W = pooled.shape[2], pooled.shape[3]
        pooled = pooled.view(B, num_h*num_w, C, pooled_H, pooled_W)

        return pooled

# --- TESTE ---
if __name__ == "__main__":
    # Batch simulado CIFAR-10
    x = torch.randn(8, 3, 32, 32)
    stride_hd = StrideHD(window_size=(4,4), stride=2, pool_size=(2,2))
    pooled_windows = stride_hd.extract_and_pool(x)
    
    print("Shape após extração + pooling:", pooled_windows.shape)
