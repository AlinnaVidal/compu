Para probarlo local:

seguir instrucciones.png
mpiexec -n 3 python3 kmeans_distributed.py


Para acceder al Cluster:

ingrese con ssh usuario@kraken.ing.puc.cl
donde usuario corresponde a: Inicial nombre + primer apellido.


en el run_kmeans.slurm:
usamos los dos nodos, como nuestro codigo funciona con mem ditribuida podemos separalo, considerando la memoria del cluster y el tiempo de ejecucion maximo

#!/bin/bash
#SBATCH --job-name=kmeans_mio
#SBATCH --partition=hpc-iic3533
#SBATCH --nodes=2              # usar Ahsoka y Ventress
#SBATCH --ntasks=16            # 8 por nodo
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G               
#SBATCH --time=00:15:00
#SBATCH --output=kmeans_%j.out

module load python/3.10 mpi/openmpi

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpirun -n $SLURM_NTASKS python kmeans_distributed.py
