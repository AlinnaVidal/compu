import numpy as np
from mpi4py import MPI
import numba
from numba import njit, prange

def generate_distributed_data(n_total, d, k, seed): # este me da data
    
    rank = MPI.COMM_WORLD.Get_rank()
    num_procesos = MPI.COMM_WORLD.Get_size()

    semilla = seed + rank
    rng = np.random.default_rng(semilla)
    #print(semilla)

    tareas_por_proceso = n_total//num_procesos
    resto = n_total % num_procesos

    if rank < resto:
        tareas_por_proceso = tareas_por_proceso + 1


    lista_numeros = rng.random((tareas_por_proceso, d))
    #print(lista_numeros)

    return lista_numeros

@njit(parallel=True)
def compute_distances(data, centroids):
    matriz_distancias = np.zeros((centroids.shape[0], data.shape[0] ), dtype=np.float64)

    for punto in prange(data.shape[0]): #uso prange

        for centroide in range(centroids.shape[0]):
            distancia = 0
            for dimension in range(data.shape[1]):
                r = data[punto, dimension] - centroids[centroide, dimension]
                distancia += r * r

            matriz_distancias[centroide, punto] = distancia
    return matriz_distancias

@njit(parallel=True) #este de da labels
def assign_labels(distances):
    n_puntos = distances.shape[1] 
    n_centroides = distances.shape[0]

    lista_cluster_asignado = np.zeros(n_puntos, dtype=np.int64)
    for i in prange(n_puntos): # no se si usar prange
        distancia_minima = 100000000000000
        centroide_minimo = -1
        for j in range(n_centroides):
            if distancia_minima > distances[j, i]:
                distancia_minima = distances[j, i]
                centroide_minimo = j
        lista_cluster_asignado[i] = centroide_minimo
    return lista_cluster_asignado

@njit(parallel=True)
def compute_local_sums(data, labels, k):
    d = data.shape[1]
    suma = np.zeros((k, d), dtype=np.float64) # data.shape[1] es la dimensión d
    lista_cluster_cantidad = np.zeros(k, dtype=np.int64)
    for n_centroide in prange(k):
        numero_puntos_en_centroide = 0
        for n_punto in range(data.shape[0]):
            if n_centroide == labels[n_punto]:
                numero_puntos_en_centroide = numero_puntos_en_centroide + 1
                for dimension in range(d):
                    suma[n_centroide, dimension] += data[n_punto, dimension]

        lista_cluster_cantidad[n_centroide] = numero_puntos_en_centroide
    return suma, lista_cluster_cantidad

def clustering_kmeans(n_total, d, k, seed, centroids, iteracion_maxima=100, tolerance=0.000001):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = generate_distributed_data(n_total, d, k, seed)
    print("__________________PUNTOS GENERADOS______________________")
    print(data)
    print("________________________________________________________________")
    comm.Bcast(centroids, root=0)
    for i in range(iteracion_maxima):
        print("iteración:", i)
        parar = False
        centroide_pasado = centroids.copy()

        distances = compute_distances(data, centroids)
        labels = assign_labels(distances)
        local_sums, cantidad_por_cluster = compute_local_sums(data, labels, k)

        global_sums = np.zeros_like(local_sums)
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        ### Calcular los centroides nuevos ### 
        global_counts = np.zeros_like(cantidad_por_cluster)
        comm.Allreduce(cantidad_por_cluster, global_counts, op=MPI.SUM)
        if rank == 0:
            for centroide in range(k):
                if global_counts[centroide] > 0:  
                    centroids[centroide] = global_sums[centroide] / global_counts[centroide]
            moved = np.linalg.norm(centroids - centroide_pasado)
            if moved < tolerance:
                parar = True
        parar = comm.bcast(parar, root=0)


        if parar:
            break
        
        comm.Bcast(centroids, root=0)
        #######################################}
    return centroids
        
# x = generate_distributed_data(4, 1, 1, 666 )
# print(x)
# centroids = np.array([[0.0]])
# y = compute_distances(x, centroids)
# print(y)

n_total = 4000000
d = 2
k = 20
seed = 100

centroids = np.array([
    [0.1, 0.9],   # Cerca del primer punto
    [0.6, 0.3]    # Cerca del tercero
])


centroidee = clustering_kmeans(n_total,d,k,seed, centroids)
print("centroide final", centroidee)


#generate_distributed_data(10,2,3,4)
