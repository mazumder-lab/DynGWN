#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=64G
#SBATCH --time=4-00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shibal@mit.edu
#SBATCH --output=dyngwn_%A_%a.out
#SBATCH --error=dyngwn_%A_%a.err
#SBATCH --array=0-1

TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo $TASK_ID
echo $EXP_ID

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((20543 + TASK_ID))
export WORLD_SIZE=1
echo $MASTER_PORT


### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2022b
#module load gurobi/gurobi-903

# Call your script as you would from your command line
source activate additive2

export HDF5_USE_FILE_LOCKING=FALSE

cd /home/gridsan/shibal/projects/DynGWN

# METR-LA
# srun /home/gridsan/shibal/.conda/envs/dyngwn/bin/python -u main_dyngwn.py --data './data/METR-LA' --adjdata './data/METR-LA/adj_mx.pkl' --n_train 50000 --batch_size 8 --num_nodes 207 --epochs 150 --layers 2 --blocks 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed 0 --workers 8 --learning_rate 0.0001 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/METR-LA/Nodes207/Dynamic:True/N50000/Blocks4/partial-correlation/absolute/LR0.0001/8.0/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/dyngwn/bin/python -u main_dyngwn.py --data './data/METR-LA' --adjdata './data/METR-LA/adj_mx.pkl' --n_train 50000 --batch_size 8 --num_nodes 207 --epochs 150 --layers 2 --blocks 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed 0 --workers 8 --learning_rate 0.0005 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/METR-LA/Nodes207/Dynamic:True/N50000/Blocks4/partial-correlation/absolute/LR0.0005/8.0/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/dyngwn/bin/python -u main_dyngwn.py --data './data/METR-LA' --adjdata './data/METR-LA/adj_mx.pkl' --n_train 50000 --batch_size 8 --num_nodes 207 --epochs 150 --layers 2 --blocks 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed 0 --workers 8 --learning_rate 0.001 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/METR-LA/Nodes207/Dynamic:True/N50000/Blocks4/partial-correlation/absolute/LR0.001/8.0/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/dyngwn/bin/python -u main_dyngwn.py --data './data/METR-LA' --adjdata './data/METR-LA/adj_mx.pkl' --n_train 50000 --batch_size 8 --num_nodes 207 --epochs 150 --layers 2 --blocks 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed 0 --workers 8 --learning_rate 0.005 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/METR-LA/Nodes207/Dynamic:True/N50000/Blocks4/partial-correlation/absolute/LR0.005/8.0/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/dyngwn/bin/python -u main_dyngwn.py --data './data/METR-LA' --adjdata './data/METR-LA/adj_mx.pkl' --n_train 50000 --batch_size 8 --num_nodes 207 --epochs 150 --layers 2 --blocks 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed 0 --workers 8 --learning_rate 0.01 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/METR-LA/Nodes207/Dynamic:True/N50000/Blocks4/partial-correlation/absolute/LR0.01/8.0/tuning.txt'

# PEMS-Bay
########## Dynamic
# srun /home/gridsan/shibal/.conda/envs/dyngwn/bin/python -u main_dyngwn.py --data '/home/gridsan/shibal/traffic-data/data/PEMS-BAY' --adjdata '/home/gridsan/shibal/traffic-data/data/PEMS-BAY/adj_mx.pkl' --domain "traffic" --n_train 5000 --num_nodes 80 --epochs 200 --layers 2 --blocks 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --n_trials 10 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --learning_rate 0.01 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/PEMS-BAY/Nodes80/Dynamic:True/N5000/Blocks4/partial-correlation/absolute/LR0.01/8.$SLURM_ARRAY_TASK_ID/tuning.txt'

######### Stock volatilities
# srun /home/gridsan/shibal/.conda/envs/additive2/bin/python -u main_dyngwn.py --data './data/stocks' --adjdata './data/stocks/adj_mx.pkl' --domain "stocks" --n_train 2500 --batch_size 8 --num_nodes 83 --in_dim 10 --input_seq_length 50 --epochs 150 --layers 2 --blocks 2 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --workers 8 --learning_rate 0.01 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/stocks/Nodes83/Dynamic:True/N2500/Blocks2/partial-correlation/absolute/LR0.01/8.$SLURM_ARRAY_TASK_ID/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/additive2/bin/python -u main_dyngwn.py --data './data/stocks' --adjdata './data/stocks/adj_mx.pkl' --domain "stocks" --n_train 2500 --batch_size 8 --num_nodes 83 --in_dim 10 --input_seq_length 50 --epochs 150 --layers 2 --blocks 2 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --workers 8 --learning_rate 0.005 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/stocks/Nodes83/Dynamic:True/N2500/Blocks2/partial-correlation/absolute/LR0.005/8.$SLURM_ARRAY_TASK_ID/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/additive2/bin/python -u main_dyngwn.py --data './data/stocks' --adjdata './data/stocks/adj_mx.pkl' --domain "stocks" --n_train 2500 --batch_size 8 --num_nodes 83 --in_dim 10 --input_seq_length 50 --epochs 150 --layers 2 --blocks 2 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --workers 8 --learning_rate 0.001 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/stocks/Nodes83/Dynamic:True/N2500/Blocks2/partial-correlation/absolute/LR0.001/8.$SLURM_ARRAY_TASK_ID/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/additive2/bin/python -u main_dyngwn.py --data './data/stocks' --adjdata './data/stocks/adj_mx.pkl' --domain "stocks" --n_train 2500 --batch_size 8 --num_nodes 83 --in_dim 10 --input_seq_length 50 --epochs 150 --layers 2 --blocks 2 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --workers 8 --learning_rate 0.0005 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/stocks/Nodes83/Dynamic:True/N2500/Blocks2/partial-correlation/absolute/LR0.0005/8.$SLURM_ARRAY_TASK_ID/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/additive2/bin/python -u main_dyngwn.py --data './data/stocks' --adjdata './data/stocks/adj_mx.pkl' --domain "stocks" --n_train 2500 --batch_size 8 --num_nodes 83 --in_dim 10 --input_seq_length 50 --epochs 150 --layers 2 --blocks 2 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --workers 8 --learning_rate 0.0001 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/stocks/Nodes83/Dynamic:True/N2500/Blocks2/partial-correlation/absolute/LR0.0001/8.$SLURM_ARRAY_TASK_ID/tuning.txt'


# srun /home/gridsan/shibal/.conda/envs/additive2/bin/python -u main_dyngwn.py --data './data/exchange' --adjdata './data/exchange/adj_mx.pkl' --domain "stocks" --n_train 4340 --batch_size 8 --num_nodes 80 --in_dim 1 --input_seq_length 50 --epochs 150 --layers 2 --blocks 2 --kernel_size 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --workers 8 --learning_rate 0.001 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/exchange/Nodes80/Dynamic:True/N4340/Blocks2/partial-correlation/absolute/LR0.001/8.$SLURM_ARRAY_TASK_ID/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/additive2/bin/python -u main_dyngwn.py --data './data/exchange' --adjdata './data/exchange/adj_mx.pkl' --domain "stocks" --n_train 4340 --batch_size 8 --num_nodes 80 --in_dim 1 --input_seq_length 50 --epochs 150 --layers 2 --blocks 2 --kernel_size 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --workers 8 --learning_rate 0.0005 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/exchange/Nodes80/Dynamic:True/N4340/Blocks2/partial-correlation/absolute/LR0.0005/8.$SLURM_ARRAY_TASK_ID/tuning.txt'
# srun /home/gridsan/shibal/.conda/envs/additive2/bin/python -u main_dyngwn.py --data './data/exchange' --adjdata './data/exchange/adj_mx.pkl' --domain "stocks" --n_train 4340 --batch_size 8 --num_nodes 80 --in_dim 1 --input_seq_length 50 --epochs 150 --layers 2 --blocks 2 --kernel_size 4 --gcn_bool --adjtype 'doubletransition' --addaptadj  --randomadj --dynamic_gcn_bool --dynamic_supports_len 1 --dynamic_graph 'partial-correlation' --dynamic_graph_transform 'absolute' --dynamic_graph_window 48 --expid 8 --tuning_seed $SLURM_ARRAY_TASK_ID --workers 8 --learning_rate 0.0001 --save_directory './logs/dyngwn' |& tee -a './logs/dyngwn/exchange/Nodes80/Dynamic:True/N4340/Blocks2/partial-correlation/absolute/LR0.0001/8.$SLURM_ARRAY_TASK_ID/tuning.txt'

########### For No static graph: --gcn_bool can be omitted.
