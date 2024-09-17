import numpy as np
import scipy as sp
from scipy.stats import ortho_group, special_ortho_group
from scipy.linalg import null_space
from scipy.spatial import distance_matrix
import random
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from procrustes.generic import generic as procrustes
from procrustes.orthogonal import orthogonal as orth_procrustes


# distance

def geodesicDistance(X, k = 15):
    iso = Isomap(n_components = 2,n_neighbors=k)
    return iso.fit(X).dist_matrix_


# greedy permutation

def getGreedyPerm(D):
    # Author: Chris Tralie
    # https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8
    if D.shape[0] != D.shape[1]:
        D = distance_matrix(D,D)
    N = D.shape[0]
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)


# given distance matrix D and natural number J, returns a cover with J elements
def greedyPermutationCover(D, J, overlap):
    perm, lambdas = getGreedyPerm(D)
    eps = lambdas[J-1]
    #print(lambdas)
    N = D.shape[0]
    cover = {}
    for i in perm[:J]:
        cover[i] = []
        for j in range(N):
            if D[i,j] < overlap * eps :
                cover[i].append(j)
        #cover[i] = np.array(cover[i])
    return perm[:J], cover, overlap * eps


def cmds(D,dim):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n,n))/n
    B = -H.dot(D**2).dot(H)/2

    if dim > B.shape[0]:
        return np.zeros((n,dim)), None

    evals, evecs = sp.linalg.eigh(B,subset_by_index=[n-dim,n-1])
    evals, evecs = evals[::-1], evecs[:,::-1]

    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    res = np.zeros((n,dim))
    res[:Y.shape[0],:Y.shape[1]] = Y

    return res, evals


def normalizedEuclideanModelsDim(D, centers, cover, dim):
    F = {}
    for i in centers:
        Y, evals = cmds(D[cover[i],:][:,cover[i]], dim) 
        F[i] = Y[:,:dim]
        F[i] -= np.mean(F[i],axis=0)
    return F


# nerve


def computeNerveBase(dm_base, cover_size, overlap) : 
    centers_base, cover_base, epsilon_base = greedyPermutationCover(dm_base, cover_size, overlap)
    nerve_base = coverDistinctIntersections(cover_base)
    #nerve_base = filter_nerve(centers_base, nerve_base_, n_neighbors)
    partition_unity_base = lambda center, x : np.exp( -1/(1 - (dm_base[x,center]/epsilon_base)**2) )

    return cover_base, nerve_base, centers_base, partition_unity_base

def coverDistinctIntersections(cover,minimum_intersection_size = 1):
    vertices = list(cover.keys())
    J = len(vertices)
    edges = {}
    for i in range(J):
        for j in range(i+1,J):
            intersect = list(set(cover[vertices[i]]).intersection(cover[vertices[j]]))
            size_intersection = len(intersect)
            if size_intersection >= minimum_intersection_size:
                edges[ (vertices[i],vertices[j]) ] = intersect
    return edges


# tangent structure 

def computeTangentStructureBase(base_pointcloud, cover_base, centers_base, local_dimension_base):
    psis, base_coordinates = localPCA(base_pointcloud, cover_base, centers_base, local_dimension_base)
    alphas = baseOrthogonalFrames(centers_base, psis)
    taus_base = { c:np.array([np.average(base_pointcloud[cover_base[c]], axis=0)]).T for c in centers_base }

    return psis, base_coordinates, alphas, taus_base


# fiberwise alignment

def fiberwiseEmbedding(local_dimension_preimage, local_dimension_base, target_dim, centers_base, cover_base, nerve_base, fiber_coordinates, alphas, n_iterations_align, seed) : 
    source_dims = { c:local_dimension_preimage - local_dimension_base for c in centers_base }
    target_dims = { c:target_dim - local_dimension_base for c in centers_base }

    omegas, _ = localAlignments(cover_base, nerve_base, fiber_coordinates)
    if source_dims == target_dims:
        chi_primes = initialChiPrimesSpecial(source_dims, target_dims, centers_base, seed = seed)
        chi_primes, deltas, errs, synchronization_hist = alignChiPrimesSpecial(centers_base, chi_primes, alphas, cover_base, nerve_base, n_iterations_align, omegas, weighted=True, seed=seed)
    else :
        chi_primes = initialChiPrimes(source_dims, target_dims, centers_base, seed = seed)
        chi_primes, deltas, errs = alignChiPrimes(chi_primes, alphas, nerve_base, n_iterations_align, omegas, weighted=True, seed=seed)
        synchronization_hist = None
    chis = buildChis(centers_base, chi_primes, alphas)

    return chis, deltas, errs, synchronization_hist


def localAlignments(cover,nerve,F):
    omega = {}
    mu = {}
    for edge, intersect in nerve.items():
        i = edge[0]
        j = edge[1]
        fij = np.array([ F[i][cover[i].index(k)] for k in intersect ])
        fji = np.array([ F[j][cover[j].index(k)] for k in intersect ])
        muij = np.mean(fij, axis=0)
        muji = np.mean(fji, axis=0)
        mu[(i,j)] = [muij,muji]
        fij -= muij
        fji -= muji
        Mij = fij.T @ fji
        omegaij = orthogonalize(Mij)
        omega[(i,j)] = omegaij

    return omega, mu


def assemblePart(dm, part, centers, cover, fiber_coordinates, chi, pi, reach, fib_scale):
    points = []
    for x in range(len(dm)):
        versions = []
        accum = 0
        for i in centers :
            if x in cover[i] :
                belongs = part(i,x)
                accum += belongs
                idx = cover[i].index(x)
                version = ((reach * fib_scale) * chi[i] @ np.array([fiber_coordinates[i][idx]]).T + np.array([pi[x]]).T) * belongs
                versions.append(version)
        versions = np.array(versions)
        fx = np.sum(versions, axis = 0) / accum
        points.append(fx)

    return np.array(points)[:,:,0]


def nonorthogonalProcrustes(A,B,use_svd=False):
    n_cols_res = A.shape[0]
    solT = procrustes(A.T,B.T,use_svd=use_svd).t
    return solT.T[:,:n_cols_res]

def orthogonalProcrustes(A,B):
    n_cols_res = A.shape[0]
    solT = orth_procrustes(A.T,B.T).t
    return solT.T[:,:n_cols_res]

def localPCA(X, cover, centers, local_dimension):

    local_bases = {}
    local_projections = {}

    for c in centers:
        pca = PCA()
        pca.fit(X[cover[c],:])
        local_bases[c] = pca.components_[:local_dimension,:].T
        local_projections[c] = pca.transform(X[cover[c]])[:,:local_dimension]
    
    return local_bases, local_projections

# orthogonal complement of a partial basis
def basisOrthogonalComplement(M):
    n, m = M.shape
    x, _, _ = np.linalg.svd(M)
    x = x[:, :m]

    u, s, v = np.linalg.svd(x)
    y = u[:, m:]
    return y


def orthogonalize(M) :
    u, s, vh = np.linalg.svd(M)
    return u @ np.eye(u.shape[1],vh.shape[0]) @ vh

def specialOrthogonalize(M) :
    if M.shape[1] == 0 :
        return np.array([[1]])
    u, s, vh = np.linalg.svd(M)
    if np.isclose(np.linalg.det(u), np.linalg.det(vh)):
        return u @ vh
    else :
        u[:,-1] = -u[:,-1]
        return u @ vh


def interpolate(X, Y, a):
    n,p = X.shape
    I = np.eye(n)
    V = 2*Y @ np.linalg.inv(np.eye(p) + X.T @ Y)
    K = ((a*V) @ X.T) - (X @ (a*V).T)
    return np.linalg.inv(I - 0.5*K) @ (I + 0.5*K) @ X


def z2Synchronization(vertices, edges, weights):
    n = len(vertices)
    connection_laplacian = np.zeros((n,n))
    for i in range(n):
        nonzero_elems = 0
        for j in range(n):
            if i==j :
                nonzero_elems += weights[vertices[i],vertices[j]]
                connection_laplacian[i,j] = weights[vertices[i],vertices[j]]
            else :
                if (vertices[i],vertices[j]) in edges.keys():
                    nonzero_elems += weights[vertices[i],vertices[j]]
                    connection_laplacian[i,j] = edges[(vertices[i],vertices[j])] * weights[vertices[i],vertices[j]]
                if (vertices[j],vertices[i]) in edges.keys():
                    nonzero_elems += weights[vertices[j],vertices[i]]
                    connection_laplacian[i,j] = edges[(vertices[j],vertices[i])] * weights[vertices[j],vertices[i]]
 
        connection_laplacian[i] /= nonzero_elems
    
    eigen_solution = np.linalg.eigh(connection_laplacian)

    cocycle = np.sign(eigen_solution[1][:,-1])

    return { vertices[i]:cocycle[i] for i in range(n)}, eigen_solution[1][:,-1]

def alignChiPrimesSpecial(centers, chi_primes, alphas, cover, nerve, n_iterations, omegas, weighted=True, seed=0):

    signs_omegas = { p:np.sign(np.linalg.det(omega)) for p,omega in omegas.items() }
    signs_alphas = { (i,j):np.linalg.det(alphas[i].T @ alphas[j]) for i,j in nerve.keys() }
    signs = { p:signs_omegas[p] * signs_alphas[p] for p in nerve.keys() }

    weights_synchronization = { p:len(nerve[p]) for p in nerve.keys() } | { (i,i):len(cover[i]) for i in centers}

    correction_signs, synch_hist = z2Synchronization(centers, signs, weights_synchronization)

    # correction to alphas
    for c in centers:
        alphas[c][:,0] *= correction_signs[c]


    alphas_cocycle = { (i,j):orthogonalize(alphas[i].T @ alphas[j]) for i,j in nerve.keys() }

    random.seed(seed)

    nerve_pairs = list(nerve)

    if weighted:
        weights = [ len(nerve[pair]) for pair in nerve_pairs ]

    if weighted:
        pairs = random.choices(nerve_pairs, weights=weights, k = n_iterations)
    else :
        pairs = random.choices(nerve_pairs, k = n_iterations)

    deltas = []
    errs = []
    a = 1

    for n in range(n_iterations):
        one, two = pairs[n]

        chi_prime_one = chi_primes[one]
        chi_prime_two = chi_primes[two]
        M = orthogonalProcrustes( omegas[(one,two)], alphas_cocycle[(one,two)] @ chi_prime_two)

        errs.append(np.linalg.norm(M - chi_prime_one))

        chi_primes[one] = specialOrthogonalize((1-a) * chi_prime_one + a * M)

        deltas.append(np.linalg.norm(chi_primes[one]-chi_prime_one))

        a = 1 - (n+1)/n_iterations

    return chi_primes, deltas, errs, synch_hist


def alignChiPrimes(chi_primes, alphas, nerve, n_iterations, omegas, weighted=True, seed=0):

    alphas_cocycle = { (i,j):orthogonalize(alphas[i].T @ alphas[j]) for i,j in nerve.keys() }

    nerve_pairs = list(nerve)

    if weighted:
        weights = [ len(nerve[pair]) for pair in nerve_pairs ]

    random.seed(seed)

    if weighted:
        pairs = random.choices(nerve_pairs, weights=weights, k = n_iterations)
    else :
        pairs = random.choices(nerve_pairs, k = n_iterations)

    deltas = []
    errs = []
    a = 1

    for n in range(n_iterations):
        one, two = pairs[n]

        chi_prime_one = chi_primes[one]
        chi_prime_two = chi_primes[two]
        M = orthogonalProcrustes( omegas[(one,two)], alphas_cocycle[(one,two)] @ chi_prime_two)

        errs.append(np.linalg.norm(M - chi_prime_one))

        chi_primes[one] = orthogonalize((1-a) * chi_prime_one + a * M)

        deltas.append(np.linalg.norm(chi_primes[one]-chi_prime_one))

        a = 1 - (n+1)/n_iterations

    return chi_primes, deltas, errs


def initialChiPrimes(source_dims, target_dims, centers, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    chi_primes = {}
    for c in centers:
        chi_prime = ortho_group.rvs(dim=target_dims[c])[:,:source_dims[c]]
        chi_primes[c] = chi_prime

    return chi_primes

def initialChiPrimesSpecial(source_dims, target_dims, centers, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    chi_primes = {}
    for c in centers:
        if target_dims[c] > 1:
            sp_orth_mat = special_ortho_group.rvs(dim=target_dims[c])
        else :
            sp_orth_mat = np.array([[1]])
        chi_prime = sp_orth_mat[:,:source_dims[c]]
        chi_primes[c] = chi_prime

    return chi_primes


def buildChis(centers, chi_primes, alphas):
    chis = {}

    for c in centers:
        chis[c] = alphas[c] @ chi_primes[c]

    return chis

def baseOrthogonalFrames(centers, psis):

    frames = {}

    for c in centers:
        frames[c] = basisOrthogonalComplement(psis[c])

    return frames


def buildFiberCoordinates(centers, base_coordinates, local_models, not_intersect = False, center_fiber = False):

    if center_fiber :
        import miniball

    fiber_models = {}

    for c in centers:
        if not_intersect : 
            fiber_models[c] = local_models[c]
        else :
            align_base_preimage = nonorthogonalProcrustes(local_models[c].T, base_coordinates[c].T)
            fiber_directions = null_space(align_base_preimage)

            fiber_models[c] = (fiber_directions.T @ local_models[c].T).T

    for c in centers:
        if center_fiber == True:
            center, _ = miniball.get_bounding_ball(fiber_models[c])
            fiber_models[c] -= center
        fiber_models[c] = fiber_models[c]/np.max(np.linalg.norm(fiber_models[c], axis=1))

    return fiber_models

def computeReach(centers, pi, psis):
    local_reaches = []
    n = len(centers)
    for i in range(n):
        for j in range(i+1,n):
            a = np.linalg.norm(pi[centers[i]] - pi[centers[j]])**2
            b = np.linalg.norm(psis[centers[i]].T @ (pi[centers[j]] - pi[centers[i]]))**2
            if np.isclose(a,b) :
                local_reaches.append(1)
            else :
                local_reaches.append(a/(2*np.sqrt(a-b)))

    return min(local_reaches)


def obstructionsData(X_dm, pi, k, e, d, overlap_cover=3, center_fiber=False):
    D = pi.shape[1]
    dm_base = distance_matrix(pi,pi)
    cover_base, nerve_base, centers_base, part_base = computeNerveBase(dm_base, k, overlap=overlap_cover)

    local_models = normalizedEuclideanModelsDim(X_dm,centers_base, cover_base, d)

    # compute tangent structure base space
    psis, base_coordinates, alphas, taus_base = computeTangentStructureBase(pi, cover_base, centers_base, e)

    # compute fiber coordinates
    fiber_coordinates = buildFiberCoordinates(centers_base, base_coordinates, local_models, center_fiber=center_fiber)

    omegas, _ = localAlignments(cover_base, nerve_base, fiber_coordinates)

    ## compute fiberwise embedding
    #chis, deltas, errs, synchronization_hist = fiberwiseEmbedding(d, e, D, centers_base, cover_base, nerve_base, fiber_coordinates, alphas, n_iter, seed)

    return cover_base, nerve_base, centers_base, omegas


def fibered(X_dm, pi, k, e, d, fib_scale=1/2, overlap_cover=3, n_iter = 1000, seed = 0, center_fiber=False):
    D = pi.shape[1]
    dm_base = distance_matrix(pi,pi)
    cover_base, nerve_base, centers_base, part_base = computeNerveBase(dm_base, k, overlap=overlap_cover)
    local_models = normalizedEuclideanModelsDim(X_dm,centers_base, cover_base, d)

    # compute tangent structure base space
    psis, base_coordinates, alphas, taus_base = computeTangentStructureBase(pi, cover_base, centers_base, e)

    reach = computeReach(centers_base,pi,psis)

    # compute fiber coordinates
    fiber_coordinates = buildFiberCoordinates(centers_base, base_coordinates, local_models, center_fiber=center_fiber)

    # compute fiberwise embedding
    chis, deltas, errs, synchronization_hist = fiberwiseEmbedding(d, e, D, centers_base, cover_base, nerve_base, fiber_coordinates, alphas, n_iter, seed)

    return assemblePart(X_dm, part_base, centers_base, cover_base, fiber_coordinates, chis, pi, reach, fib_scale)