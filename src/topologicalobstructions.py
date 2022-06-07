from ripser import ripser
from scipy import sparse
import numpy as np
from src.linearsystemsmodp import *


def sparse_weighted_graph_persistence(sparse_matrix, maxdim, suplevelset, do_cocycles=False):
	cocycles = []
	if suplevelset:
		r = ripser(-sparse_matrix, maxdim=maxdim, distance_matrix=True,do_cocycles=do_cocycles)
	else:
		r = ripser(sparse_matrix, maxdim=maxdim, distance_matrix=True,do_cocycles=do_cocycles)
	dgms = r['dgms']
	if do_cocycles:
		cocycles = r['cocycles']
	if suplevelset:
		dgms = [-d[:,[1,0]] for d in dgms]
	return dgms, cocycles


def functional_persistence(vertices, edges, fvertices, fedges, maxdim=0, suplevelset=False, do_cocycles=False):
	N = vertices.shape[0]
	vertex_label_to_index = { vertices[i]:i for i in range(N) }
	vertices_indices = np.array(list(range(N)))
	edges_indices = np.array([[vertex_label_to_index[a], vertex_label_to_index[b]] for a,b in edges ])
	I = np.concatenate((edges_indices[:,0],vertices_indices))
	J = np.concatenate((edges_indices[:,1],vertices_indices))
	V = np.concatenate((fedges,fvertices))
	D = sparse.coo_matrix((V, (I,J)), shape=(N,N)).tocsr()

	return sparse_weighted_graph_persistence(D, maxdim, suplevelset, do_cocycles=do_cocycles)




def ordered_neighbors_dist_mat(dist_mat) :
    return np.argsort(dist_mat, axis=1)

def approx_cocycle_death(dist_mat, cocycle, tolerance = 0.5, initial_death = 0.99) :

    ordered_neighbors_index = ordered_neighbors_dist_mat(dist_mat)

    death = initial_death
    
    n = len(dist_mat)
    for i in range(n) :

        close_neighbors_i = ordered_neighbors_index[i]
        for j in close_neighbors_i :
            if j <= i :
                continue
            if dist_mat[i,j] > death :
                break
            ij_omega = cocycle[i,j]

            close_neighbors_j = ordered_neighbors_index[j]

            for k in close_neighbors_j :

                if k <= j:
                    continue
                if dist_mat[j,k] > death or dist_mat[i,k] > death :
                    continue

                jk_omega = cocycle[j,k]
                ik_omega = cocycle[i,k]
    

                #if np.linalg.norm(ij_omega @ jk_omega - ik_omega) != 0 :
                #    print(np.linalg.norm(ij_omega @ jk_omega - ik_omega))
                if np.linalg.norm(ij_omega @ jk_omega - ik_omega) >= tolerance :
                    death_candidate = max(dist_mat[i,j], dist_mat[j,k], dist_mat[i,k])
                    if death_candidate < death :
                        death = death_candidate

    return death


def approx_sw1(dist_mat, approx_cocycle, max_eps) :
    cocycle = []
    n = dist_mat.shape[0]

    for i in range(n):
        for j in range(n) :
            if dist_mat[i,j] < max_eps and np.linalg.det(approx_cocycle[i,j]) < 0 :
                cocycle.append([i,j,1])

    return np.array(cocycle)


def matrix_from_vertices_gen_cocycle(dist_mat, coh_gen, deaths, cocycle, co_death) :

    N = dist_mat.shape[0]

    #kd_tree = KDTree(pointcloud, leaf_size=2)
    #close_neighbors = kd_tree.query_radius(pointcloud, r = co_death)
    #close_neighbors = [ np.array([j for j in close_neighbors[i]
    #                              if j != i and np.linalg.norm(pointcloud[i] - pointcloud[j]) < co_death])
    #                              for i in range(N) ]
    #close_neighbors = np.array(close_neighbors)

    close_neighbors = [ np.array([ j for j in range(N) if j != i and dist_mat[i,j] < co_death ]) for i in range(N)]


    # start by having as generators only the coboundaries of vertices
    gens = []

    for i in range(N) :
        gen = []
        for j in close_neighbors[i] :
            gen.append([i,j,-1])
        if len(gen) > 0 :
            gens.append(gen)

    true_coh_gen = []
    for g, d in zip(coh_gen, deaths) :
        #print(d)
        #print(co_death)
        g_ = []
        for e in g :
            i, j, v = e
            if dist_mat[i,j] < co_death and dist_mat[i,j] < d  :
                g_.append(e)
        true_coh_gen.append(g_)

    gens = list(true_coh_gen) + list(gens) + [cocycle]


    ### rows should be indexed by pairs of ordered and distinct points within distance co_death 
    rows = [(i,j) for i in range(N) for j in close_neighbors[i] if i < j]
    #print(dist_mat[13,15])
    #print(close_neighbors[i])
    #print(dist_mat[13,15] < co_death)
    edge_to_row = dict([])
    for n,p in enumerate(rows) :
        edge_to_row[p] = n


    M = np.zeros((len(rows), len(gens)), dtype=int)

    for col, g in enumerate(gens) :
        for e in g :
            i,j,v = e
            if i < j :
                M[edge_to_row[(i,j)],col] = v
            else :
                M[edge_to_row[(j,i)],col] = -v

    return M


def sw1_obstruction(X_dm, nerve_base, centers_base, omegas):

    vertices = centers_base
    edges = list(nerve_base.keys())
    fvertices =  np.full(vertices.shape[0], 0)
    N = X_dm.shape[0]
    fedges = np.array([ 1-len(nerve_base[p])/N for p in edges ])
    dgms, ripser_cocycle = functional_persistence(vertices, edges, fvertices, fedges, maxdim = 1, suplevelset=False, do_cocycles=True)
    basis = ripser_cocycle[1]

    nerve_weights = np.full((centers_base.shape[0], centers_base.shape[0]), 1.)
    cocycle = {}
    for i in range(centers_base.shape[0]):
        for j in range(centers_base.shape[0]):
            p, q = centers_base[i], centers_base[j]
            if (p,q) in nerve_base:
                cocycle[(i,j)] = omegas[p,q]
                cocycle[(j,i)] = omegas[p,q]
                nerve_weights[i,j] = 1 - len(nerve_base[(p,q)])/N
                nerve_weights[j,i] = 1 - len(nerve_base[(p,q)])/N
    
    dth = approx_cocycle_death(nerve_weights, cocycle, tolerance = 1.99, initial_death = 0.999)
    #print(dth)

    cocycle1 = approx_sw1(nerve_weights,cocycle,dth)


    a = matrix_from_vertices_gen_cocycle(nerve_weights, basis,dgms[1][:,1],cocycle1,dth)
    return solve_system_mod(a,mod=2), basis, dth, dgms