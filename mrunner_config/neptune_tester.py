from mpi4py import MPI


comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

print('My rank = {}'.format(myrank))