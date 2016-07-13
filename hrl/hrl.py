from dolfin import *
import numpy as np
set_log_level(ERROR)

class TopBoundary(SubDomain):
         def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1.0)

class BottomBoundary(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
       return on_boundary and near(x[0], 1.0)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
       return on_boundary and near(x[0], 0.0)

class u_bc_expression(Expression):
    def __init__(self, t, Ra):
        self.t = t
        self.Ra = Ra
    def eval(self, values, x):
        if x[1] == 1:
            values[0] = 0.0
        else:
            if self.Ra == 50:
                if self.t < 1.0e-2:
                    values[0] = 1.0 + 1.0e-6*sin(2*pi*x[0])
                else:
                    values[0] = 1.0
            else:
                values[0] = (1.0 - x[1])*1.0

class zero_normal_flux_bc_expression(Expression):
    def __init__(self, mesh):
        self.mesh = mesh
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = 0.0
        values[0] = g*n[0]
        values[1] = g*n[1]
    def value_shape(self):
        return (2,)

def hrl_solver(Ra, dt, nx, plot_slu=False, porder=0):

    # constant
    tol = 1.0e-5
    cfl = 0.5
    T = 1.0e10

    # t stores current time
    t = dt
    count = 1

    # create unit square mesh (nx * nx)
    mesh = UnitSquareMesh(nx, nx)

    # mark boundary
    boundaries = FacetFunction('size_t', mesh, 0)
    TopBoundary().mark(boundaries, 4)
    BottomBoundary().mark(boundaries, 2)
    RightBoundary().mark(boundaries, 3)
    LeftBoundary().mark(boundaries, 1)

    # function spaces
    BDM = FunctionSpace(mesh, "BDM", porder+1)
    DG0 = FunctionSpace(mesh, "DG", porder)
    DG1 = FunctionSpace(mesh, "DG", porder+1)
    W = MixedFunctionSpace([BDM, DG0, DG1])
    # function space to project velocity
    VEL = VectorFunctionSpace(mesh, 'CG', 1)

    if Ra == 50:
        w = Function(W)
        w0= Function(W)
    else:
        w = Function(W, 'hrl_ra%d_nx%d.xml' % (Ra-10, nx))
        w0= Function(W, 'hrl_ra%d_nx%d.xml' % (Ra-10, nx))

    # test functions
    (uu, p, u) = split(w)
    (uu0, p0, u0) = split(w0)
    (vv, q, v) = TestFunctions(W)

    # define boundary expressions
    u_bc = u_bc_expression(t, Ra)
    zero_normal_flux_bc = zero_normal_flux_bc_expression(mesh)

    # define measure ds
    ds = Measure("ds")[boundaries]

    # cell size
    h = CellSize(mesh)

    # normal direction
    n = FacetNormal(mesh)

    # temperature-dependent source term
    def gravity(u):
        val = as_vector([0.0, Ra*u])
        return val

    # weak form of darcy velocity equation
    F_vel = (dot(uu, vv) - div(vv)*p - inner(gravity(u), vv) )*dx

    # weak form of flow equation
    F_flo = nabla_div(uu)*q*dx

    alpha = Constant(5.0)
    if porder == 1:
        alpha = Constant(100.0)
    # un = un on outflow facet, otherwise 0
    un = (dot(uu, n) + abs(dot(uu, n)))/2.0

    # weak form of time gradient term
    a_tim = (u - u0)/dt*v*dx

    # internal
    a_int = dot(grad(v), grad(u) - uu*u)*dx 
    #facet
    a_fac = (alpha('+')/h('+'))*dot(jump(v, n), jump(u, n))*dS \
          - dot(avg(grad(v)), jump(u, n))*dS \
          - dot(jump(v, n), avg(grad(u)))*dS
    #velocity
    a_vel = dot(jump(v), un('+')*u('+') - un('-')*u('-') )*dS + dot(v, un*u)*ds

    a = a_int + a_fac + a_vel
    # weak form of advection-diffusion equation
    F_a_d = a_tim + a

    F = F_vel + F_flo + F_a_d

    # boundary condition
    def all_domain(x, on_boundary):
        return on_boundary
    bc1 = [DirichletBC(W.sub(0), zero_normal_flux_bc, all_domain)]
    bc2 = [DirichletBC(W.sub(2), u_bc, boundaries, 2, "geometric"), \
           DirichletBC(W.sub(2), u_bc, boundaries, 4, "geometric")]
    bc = bc1 + bc2

    # assign problem and solver
    dw   = TrialFunction(W)
    J = derivative(F, w, dw)
    problem = NonlinearVariationalProblem(F, w, bc, J)
    solver  = NonlinearVariationalSolver(problem)

    solver.parameters["newton_solver"]["linear_solver"] = "gmres"
    # solver.parameters["newton_solver"]["preconditioner"] = "ilu"

    # plot
    if plot_slu:
        fig_u = plot(w.sub(0), axes=True, title = 'Velocity')
        fig_p = plot(w.sub(1), axes=True, title = 'Pressure')
        fig_ut = plot(w.sub(2), axes=True, title = 'Temperature')

    while t < T:

        # update time-dependent expression
        u_bc.t = t

        # solve system
        solver.solve()

        # calculate 2-norm difference
        warray = w.vector().array()
        w0array= w0.vector().array()
        w2norm = np.linalg.norm((warray - w0array))
        if w2norm < tol:
            print 'Steady state reached'
            break

        # update w0 with w
        w0.vector()[:] = w.vector()

        # plot solutions
        if plot_slu:
            fig_u.plot()
            fig_p.plot()
            fig_ut.plot()
 
        # adjust time step length
        velo = interpolate(w.sub(0), VEL)
        max_velocity = np.max(np.abs(velo.vector().array()))
        hmin = mesh.hmin()
        dt_cfl = cfl*hmin/max_velocity
        if dt_cfl < dt:
            dt = dt_cfl

        # calculate Nu
        grad_norm = inner(grad(u),grad(u))*dx
        Nu = assemble(grad_norm)

        # print out output
        output = 'nx=%d, Ra=%d, t=%.4e, dt=%.4e, Nu=%.3f' %(nx, Ra, t, dt, Nu)
        print output

        # update time
        t += dt
        count += 1

    File('hrl_ra%d_nx%d.xml' % (Ra, nx)) << w 
    return [nx, Ra, Nu]

nxs = [32, 48, 64]
Ras = range(50,310,10)
dt = 1.0e-3
output_handle = file('hrl_table.txt', 'w')
for nx in nxs:
    output ="-"*30 + "\n%8s | %8s | %8s\n" % ('nx', 'Ra', 'Nu')
    output_handle.write(output)
    for Ra in Ras:
        if nx == 64:
            dt = 5.0e-4
        temp = hrl_solver(Ra, dt, nx, True)
        output = "%8d | %8d | %8f\n" % (temp[0], temp[1], temp[2])
        print output
        output_handle.write(output)
output_handle.close()
print 'DONE!'
