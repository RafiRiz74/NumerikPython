from mpi4py import MPI
import numpy as np

def parallel_sum(data, comm):
    local_sum = np.sum(data)
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    return total_sum

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Inisialisasi data pada prosesor utama
        data_size = 1000000
        data = np.random.rand(data_size)
    else:
        data = None

    # Sebar data ke semua prosesor
    data = comm.bcast(data, root=0)

    # Hitung jumlah elemen secara paralel
    total_sum = parallel_sum(data, comm)

    if rank == 0:
        print("Total sum:", total_sum)
