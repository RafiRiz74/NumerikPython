from mpi4py import MPI
import numpy as np
import time

def parallel_sum(arr):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Menentukan ukuran blok untuk setiap prosesor
    local_size = len(arr) // size
    remainder = len(arr) % size

    # Menentukan batas awal dan akhir untuk setiap prosesor
    start = rank * local_size + min(rank, remainder)
    end = start + local_size + (1 if rank < remainder else 0)

    # Menghitung jumlah lokal di setiap prosesor
    local_sum = np.sum(arr[start:end])

    # Mengumpulkan hasil dari setiap prosesor
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        return total_sum

if __name__ == "__main__":
    # Membuat larik NumPy besar untuk dijumlahkan
    array_size = 1000000
    data = np.arange(array_size)

    # Waktu eksekusi
    start_time = time.time()

    # Menjalankan fungsi parallel_sum
    result = parallel_sum(data)

    # Waktu eksekusi selesai
    end_time = time.time()

    # Menampilkan hasil dan waktu eksekusi
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Total sum:", result)
        print("Execution time:", end_time - start_time, "seconds")
