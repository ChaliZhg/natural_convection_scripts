from dolfin import *
set_log_level(ERROR)
from math import log as ln
import time
import numpy as np

def calerrornorm(u_e, u, Ve):
    '''
    This function calculates L2 norm of of difference between two solutions.
    '''
    u_Ve = interpolate(u, Ve)
    u_e_Ve = interpolate(u_e, Ve)
    e_Ve = Function(Ve)
    e_Ve.vector()[:] = u_e_Ve.vector().array() - \
                       u_Ve.vector().array()
    error = e_Ve**2*dx
    return sqrt(assemble(error))

def gravity(u):
    '''
    This function defines the right-hand-side source term in Eq. 7, and Ra is set to be 1.
    u: temperature
    '''
    Ra = 1.0
    val = as_vector([0.0, Ra*u])
    return val

def mms_solver(nx, order, t, dt, T):
    '''
    This function is to solve the manufactured solution problem descirbed in Section 5.1.
    nx: number of vertices when creating a unit square mesh
    order: approximation order of pressure field
    t: simulation start time
    dt: fixed time step
    T: simulation stop time
    '''
    cost0 = time.time()
    # create a unit saure mesh (nx * nx)
    mesh = UnitSquareMesh(nx, nx)

    # define function spaces
    # for velocity
    BDM = FunctionSpace(mesh, "BDM", order+1)
    # for pressure
    DG0 = FunctionSpace(mesh, "DG", order)
    # for temperature
    DG1 = FunctionSpace(mesh, "DG", order+1)
    W = MixedFunctionSpace([BDM, DG0, DG1])
    # spaces to interpolate exact solutions
    VCG = VectorFunctionSpace(mesh, "CG", order+3)
    CG = FunctionSpace(mesh, "CG", order+3)

    # manufactured source terms
    f_1 = Expression('(sin(x[0]) + cos(x[1]))*cos(t)', t=t)
    f_2 = Expression(('0.0', '(-sin(x[0]) - sin(x[1]))*cos(t)'), t=t)
    f_3 = Expression('-(sin(x[0]) + sin(x[1]))*sin(t) +(-cos(x[0])*cos(x[0]) \
        + sin(x[1])*cos(x[1]))*cos(t)*cos(t) + (sin(x[0]) + sin(x[1]))*cos(t)', t=t)

    # exact solution expressions
    u_exact = Expression('(sin(x[0])+ sin(x[1]))*cos(t)', t=t)
    p_exact = Expression('(sin(x[0]) + cos(x[1]))*cos(t)', t=t)
    uu_exact = Expression(('-cos(x[0])*cos(t)', 'sin(x[1])*cos(t)'), t=t)

    # w store current solution, and w0 stores solution from the previous step
    w = Function(W)
    w0= Function(W)

    # initialize solution according to exact solution expressions
    assign(w.sub(0), interpolate(uu_exact, BDM))
    assign(w.sub(1), interpolate(p_exact, DG0))
    assign(w.sub(2), interpolate(u_exact, DG1))
    assign(w0.sub(0), interpolate(uu_exact, BDM))
    assign(w0.sub(1), interpolate(p_exact, DG0))
    assign(w0.sub(2), interpolate(u_exact, DG1))

    # split w into uu(velocity), p(pressure), and u(temperature)
    (uu, p, u) = split(w)
    (uu0, p0, u0) = split(w0)

    # define test functions
    (vv, q, v) = TestFunctions(W)

    # n is unit normal vector to facet
    n = FacetNormal(mesh)

    # penalty terms on interior and boundary faces
    alpha = Constant(500000.0)
    gamma = Constant(1000000.0)

    # cell size
    h = CellSize(mesh)

    # weak form of flow equation
    F_flo = nabla_div(uu)*q*dx - f_1*q*dx
    # weak form of darcy velocity equation
    F_vel = (dot(uu, vv) - div(vv)*p - inner(gravity(u), vv) - inner(f_2, vv) )*dx + dot(n, vv)*p_exact*ds

    # un = un on outflow facet, otherwise 0
    un = (dot(uu, n) + abs(dot(uu, n)))/2.0

    # weak form of advectin term (Note uu is not divergence free)
    a_a = dot(grad(v), -uu*u)*dx - u*v*nabla_div(uu)*dx + dot(jump(v), un('+')*u('+') - un('-')*u('-') )*dS\
          + dot(v, un*u)*ds - f_3*v*dx
    # weak form of diffusion term
    a_d = dot(grad(v), grad(u))*dx\
          - dot(avg(grad(v)), jump(u, n))*dS - dot(jump(v, n), avg(grad(u)))*dS\
          - dot(grad(v), n*u)*ds - dot(v*n, grad(u))*ds\
          + (alpha('+')/h('+'))*dot(jump(v, n), jump(u, n))*dS + (gamma/h)*v*u*ds\
          + u_exact*dot(grad(v), n)*ds - (gamma/h)*u_exact*v*ds
    # weak form of time gradient term
    a_tim = (u - u0)/dt*v*dx
    # weak form of advection-diffusion equation
    F_a_d = a_tim + a_a + a_d 
    # weak form of the whole system
    F = F_flo + F_vel + F_a_d

    # time-stepping loop
    while t < T:
        t +=dt
        # print "nx=%d order=%d t=%f" % (nx, order, t)
        # update time in time-dependent expressions
        f_1.t = t
        f_2.t = t
        f_3.t = t
        u_exact.t = t
        p_exact.t = t
        uu_exact.t = t
        # solve the system and store solution in w
        solve(F==0, w, solver_parameters={"newton_solver": {"linear_solver": "gmres", "preconditioner": "ilu"}})
        # update w0 with w
        w0.vector()[:] = w.vector()
    # calculate differences between numerical and analytical solutions
    (uu, p, u) = w.split()
    p_error = calerrornorm(p_exact, p, CG)
    uu_error = calerrornorm(uu_exact, uu, VCG)
    u_error = calerrornorm(u_exact, u, CG)
    # number of degrees of freedom
    dof = len(w.vector().array())
    # return errors and number of dof
    cost = time.time() - cost0
    return [[p_error, uu_error, u_error], dof, cost]

# file to store convergence data
output_handle = file('mms_table.txt', 'w')
# low order element: 0, high order element: 1
orders = [0, 1]
# mesh
nxs = [4, 8, 16, 32]
for case in range(len(orders)):
    output = 'DG%d-BDM%d-DG%d\n' %(orders[case], orders[case]+1, orders[case]+1)
    print output
    # h is to store element size
    h = []
    # E is to store numerical errors
    E = []
    # DOF is to store number of degrees of freedom
    DOF = []
    # cost is to store computational cost (in seconds per step)
    cost = []
    # dt is time step length
    dt = 1.0e-5
    # t is simulation start time
    t = 4.0
    # steps to run
    steps = 1.0e2
    # T is simulation stop time
    T = t + dt*steps
    for nx in nxs:
        print "nx =  ", nx
        h.append(1.0/nx)
        order = orders[case]
        temp = mms_solver(nx, order,t, dt, T)
        E.append(temp[0])
        DOF.append(temp[1])
        cost.append(temp[2])

    # r is to store convergence rate
    r = np.zeros((len(nxs), 3))
    # calculate convergence rate
    for i in range(3):
        for j in range(len(E)):
            if j>0:
                r[j][i] = ln(E[j][i]/E[j-1][i])/ln(h[j]/h[j-1])
            else:
                r[j][i] = 0.0
    # print out output and store in .txt file.
    output += '-'*97 + '\n'
    output += '%8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s\n'\
           % ('h', 'error(p)', 'rate(p)', 'error(u)', 'rate(u)', 'error(T)', 'rate(T)', 'DOF', 'cost')
    for j in range(len(E)):
        output += '%8s | %8.2E | %8.2f | %8.2E | %8.2f | %8.2E | %8.2f | %8d | %8.2E\n'\
           % ('1/'+str(nxs[j]), E[j][0], r[j][0], E[j][1], r[j][1], E[j][2], r[j][2], DOF[j], cost[j]/steps)
    output += '-'*96 + '\n'
    print output
    output_handle.write(output)
output_handle.close()
print "DONE"
