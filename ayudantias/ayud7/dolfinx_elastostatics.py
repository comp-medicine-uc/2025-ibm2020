import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl
import numpy as np

# Parameters
L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma
N_nodes_x = 20
N_nodes_y = N_nodes_z = 6

# Domain
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
                         [N_nodes_x, N_nodes_y, N_nodes_z], cell_type=mesh.CellType.tetrahedron)

# Vector function space
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # (domain.geometry.dim,) = (3,)

# Boundary conditions
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

# xyz = ufl.SpatialCoordinate(domain)
# rho_variable = rho * xyz[0]

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# External traction
T = fem.Constant(domain, default_scalar_type((0.05, 0, 0)))

# Surface measure
ds = ufl.Measure("ds", domain=domain)

# Weak form
def epsilon(u):
    return ufl.sym(ufl.grad(u)) 

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g))) 
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

# Instance problem and solve
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "gmres", "pc_type": "lu"})
uh = problem.solve()

# Write file
with XDMFFile(domain.comm, "elastic_displacement.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(uh)