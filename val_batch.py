import os
import subprocess
import platform
import pathlib

def main():
    models = ['R101_138.pt']    # 要评估的模型'R50_154.pt',
    depth = ['101']   # ResNet网络深度（与models一一对应）'50',

    # 选择dataset文件
    plt = platform.system() # 判断操作系统类型
    if plt == 'Windows':
        dataset_path = '../Datasets/SARDet-100K'
    elif plt == 'Linux':
        dataset_path = '/root/autodl-tmp/yolo'
    else:
        raise RuntimeError('Unsupported platform')

    # 运行测试
    for i in range(len(models)):
        print('\n\n正在测试模型{}...'.format(models[i]))
        os.makedirs(os.path.join('./runs/val', models[i].split('.')[0]), exist_ok=True)
        subprocess.run(['python', 'SARDet_validation.py',
                        '--coco_path', dataset_path,
                        '--model_path', './weights/{}'.format(models[i]),
                        '--save_path', './runs/val/{}'.format(models[i].split('.')[0]),
                        '--depth', depth[i],
                        ])
    print('测试结束！')

if __name__ == '__main__':
    main()