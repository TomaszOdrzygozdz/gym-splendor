#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J SplendorAgentsArena
## Liczba alokowanych wezlow
#SBATCH -N 1
## Liczba zadan per wezel (domyslnie jest to liczba alokowanych rdzeni na wezle)
#SBATCH --ntasks-per-node=4
## Ilosc pamieci przypadajacej na jeden rdzen obliczeniowy (domyslnie 5GB na rdzen)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:02:00
## Nazwa grantu do rozliczenia zuzycia zasobow
#SBATCH -A planningrl
## Specyfikacja partycji
#SBATCH -p plgrid
## Plik ze standardowym wyjsciem
#SBATCH --output="output.out"
## Plik ze standardowym wyjsciem bledow
#SBATCH --error="error.err"

## Zaladowanie modulu mpi4py
module load plgrid/libs/python-mpi4py

## Zaladowanie modulu Python
module load plgrid/tools/python

## Zaladowanie modulu IntelMPI
module add plgrid/tools/impi


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on Dir: $SLURM_SUBMIT_DIR"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

mpiexec python3 ./run_experiment.py

echo Job finished at `date`