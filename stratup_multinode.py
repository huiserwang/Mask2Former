import ast
import os
import moxing as mox
import argparse
import logging
parser = argparse.ArgumentParser()

# common parameters
parser.add_argument('--train_url', type=str, default=None, help='the output path')
parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--batch_size_per_gpu', type=int, default=64, help='the batch size per gpu')
parser.add_argument('--batch_size_mode', type=str, default='all', choices=['all', 'single'], help='the batch size mode. If [all], it means batch size for all gpus. if [single], it means batch size for one gpu. [all]=[single]*num_gpus.')
parser.add_argument('--epochs', type=int, default=100, help='the batch size per gpu')
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--data_url', type=str, default=None)
parser.add_argument('--ckpt_path_s3', type=str, default=None, help='the ckpt path at s3')
parser.add_argument('--code_dir', type=str, help='path of code to be copied to output dir')
parser.add_argument('--test_mode', type=ast.literal_eval, default='False', help='testing for the cluster with 2 gpus')
    
# test parameter
parser.add_argument("--is_shanghai1", default=False, type=bool,
                    help="number of prototypes - it can be multihead")

args, unparsed = parser.parse_known_args()

os.system('cat /usr/local/cuda/version.txt')
os.system('nvcc --version')
print(args.train_url)


# ############# preparation stage ####################
master_host = os.environ['VC_WORKER_HOSTS'].split(',')[0]
master_addr = master_host.split(':')[0]
master_port = '8524'
# FLAGS.worldsize will be re-computed follow as FLAGS.ngpu*FLAGS.nodes_num
# FLAGS.rank will be re-computed in main_worker
modelarts_rank = args.rank  # ModelArts receive FLAGS.rank means node_rank
modelarts_world_size = args.world_size  # ModelArts receive FLAGS.worldsize means nodes_num
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port
print(f'IP: {master_addr}, PORT: {master_port}')
print(f'ModelArts rank: {modelarts_rank}, world_size: {modelarts_world_size}')

#### backup code and enter code dir
mox.file.copy_parallel('./{}/'.format(args.code_dir), args.train_url + '/{}_code/'.format(args.code_dir))
print('Current path: ' + os.getcwd())
print('Current dirs: ' + str(list(os.listdir())))
os.chdir('./{}/'.format(args.code_dir))
print('Current path changed to: ' + os.getcwd())

#### install some customized packages
mox.file.copy_parallel('s3://bucket-3690/wangxuehui/packages_zips/detectron2-0.6+cu102-cp37-cp37m-linux_x86_64.whl', '/cache/detectron2-0.6+cu102-cp37-cp37m-linux_x86_64.whl')
os.system("pip install /cache/detectron2-0.6+cu102-cp37-cp37m-linux_x86_64.whl")
os.system("pip install cython scipy shapely timm h5py yacs diffdist termcolor lmdb tensorboard einops")
os.system("pip install --ignore-installed PyYAML")


#### copy dataset from s3
print('Start copying dataset')

print("start copy coco-stuff-10k and dataset from bucket-3690 (Shanghai)")
mox.file.copy_parallel('s3://bucket-3690/wangxuehui/02data/coco/cocostuff-10k-v1.0.zip', '/cache/cocostuff-10k-v1.0.zip')
mox.file.copy_parallel('s3://bucket-3690/wangxuehui/packages_zips/pretrained/MaskFormerR50.pkl', '/cache/MaskFormerR50.pkl')
os.system('tar xf /cache/cocostuff-10k-v1.0.zip -C /cache/')
data_dir = "/cache/dataset/"

print('Finish copying dataset, data_dir:{}'.format(data_dir))

#### calculate batchsize
if args.batch_size_mode == 'all':
    args.batch_size = args.batch_size_per_gpu * args.num_gpus
elif args.batch_size_mode == 'single':
    args.batch_size = args.batch_size_per_gpu

#### cmd config
#cmd_str = 'python main_moco_official.py --mlp --aug-plus --cos --moco-t {} --world-size {} --rank {} --multiprocessing-distributed --data {} --epochs {} --batch-size {} --lr {} --use_mox True --output_dir {} > {}/train_log.txt'.format(args.moco_t, modelarts_world_size, modelarts_rank, data_dir, args.epochs, args.batch_size, args.lr, args.train_url, args.train_url)
cmd_str = f'python -m torch.distributed.launch --nproc_per_node {args.num_gpus} \
                --nnodes {modelarts_world_size} \
                --node_rank {modelarts_rank} \
                --master_addr {master_addr} \
                --master_port {master_port} train_net.py \
                --num-gpus 8 \
                --config-file configs/coco-stuff-10k-171/maskformer_R50_bs32_60k.yaml'
    
print('The running command is: ' + cmd_str)
os.system(cmd_str)