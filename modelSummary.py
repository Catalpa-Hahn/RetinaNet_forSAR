import torch, time, sys, platform
from torchsummary import summary
from thop import profile
from tqdm import tqdm
from contextlib import contextmanager

@contextmanager
def hidden_output():
    """
    用于隐藏输出的上下文管理器
    """
    original_stdout = sys.stdout
    # 打开一个空的文件对象（在 Windows 上使用 'nul'，在 Unix/Linux/MacOS 上使用 '/dev/null'）
    plat = platform.system()  # 判断操作系统类型
    if plat == 'Windows':
        empty_file = 'nul'
    elif plat == 'Linux' or plat == 'Darwin':
        empty_file = '/dev/null'
    else:
        raise NotImplementedError('无法识别的操作系统！')
    with open(empty_file, 'w') as f:
        try:
            sys.stdout = f  # 重定向标准输出到空文件
            yield
        finally:
            sys.stdout = original_stdout


def test_model(model, input_size, device):
    """
    测试模型参数
    :param model: torch模型
    :param input_size: 输入图像大小[C, H, W]
    :return: None
    """
    # 准备
    model.eval()
    input_tensor = torch.randn(input_size).to(device)

    # 输出模型结构（含参数量和模型大小）
    summary(model,
            # input_size=(32, 3, 512, 512),
            # batch_size=32,
            input_data=input_size[1:],
            col_names=("output_size", "num_params", "kernel_size"),
            depth=1,
            verbose=1,
            branching=True,)

    # 使用 thop 计算 FLOPs
    # print('\nthop结果：')
    with hidden_output():
        macs, params = profile(model, inputs=(input_tensor,))
    print(f"MACs: {macs / 1e9:.2f} GMACs")
    print(f"FLOPs: {macs / 2 / 1e9:.2f} GFLOPs (取FLOPs = MACs / 2)")
    print(f"params: {params / 1e6:.2f} Mparams")

    # 测试 FPS 和每张图像的推理时间
    time.sleep(1.0) # 暂停1s，避免输出混乱
    num_images = 100  # FPS 测试的图像数量
    start_time = time.time()

    with torch.no_grad():
        for _ in tqdm(range(num_images), desc="Testing FPS and Inference time for each image", leave=False):
            _ = model(input_tensor)

    end_time = time.time()
    total_time = end_time - start_time
    fps = num_images / total_time
    inf_time = total_time / num_images

    print(f"FPS: {fps:.2f}")
    print(f"每张图像推理时间: {inf_time:.2f} s")


def main(model_path, input_size):
    """
    主函数
    :param model_path: 模型路径
    :param input_size: 输入图像大小[B, C, H, W]
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Device:', torch.cuda.get_device_name())
    else:
        device = torch.device('cpu')
        print('Device:', torch.device('cpu'))

    model_test = torch.load(model_path)
    model_test.to(device)

    test_model(model_test, input_size, device)


if __name__ == '__main__':

    main(model_path= './weights/R50_154.pt',    # 模型路径
         input_size= (1, 3, 512, 512),          # 输入图像大小[B, C, H, W]
         )

