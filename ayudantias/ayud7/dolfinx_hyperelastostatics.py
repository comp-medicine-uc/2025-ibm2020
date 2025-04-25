from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from petsc4py import PETSc
import pyvista
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, mesh, plot

# Geometry and function space

L = 20.0
domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, 1, 1]], [10,3,3], mesh.CellType.hexahedron)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))
Vdeg1 = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

# Boundary conditions
def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], L)


fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)

left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# Functions
v = ufl.TestFunction(V)
u = fem.Function(V)

# Hyperelasticity terms

# Spatial dimension
d = len(u)

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(u))

# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)

# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

# Elasticity parameters
E = default_scalar_type(1.0e4)
nu = default_scalar_type(0.3)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# Stress
# Hyper-elasticity
P = ufl.diff(psi, F)

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Define form F (we want to find u such that F(u) = 0)
F = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)

# Instance nonlinear problem and nonlinear solver
problem = NonlinearProblem(F, u, bcs)
solver = NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
# opts[f"{option_prefix}pc_type"] = "ilu"
ksp.setFromOptions()

log.set_log_level(log.LogLevel.INFO)
tval0 = -1.5
for n in range(1, 3):
    T.value[2] = n * tval0
    num_its, converged = solver.solve(u)
    assert (converged)
    u.x.scatter_forward()
    print(f"Time step {n}, Number of iterations {num_its}, Load {T.value}")

# Interpolate solution to degree 1 
u_exp = fem.Function(Vdeg1)
u_exp.interpolate(u)

# Write file
with XDMFFile(domain.comm, "hyperelastic_displacement.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(u_exp)