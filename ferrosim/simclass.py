import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from .lattice import Lattice

class Ferro2DSim:

    """Ferro2DSim class that will setup and perform an
    Ising-like simulation of P vectors in a ferroelectric double well
      Random field defects are supported. The model is based on the paper by Ricinschii et al.
      J. Phys.: Condens. Matter 10 (1998) 477–492, but has been considerably extended.
    
    All inputs are optional (they will default to values shown below)
    
    Inputs: - n: (int) Size of lattice. Default = 10
              alpha: (float) alpha term in Landau expansion. Default = -1.85
              beta: (float) beta term in Landau expansion. Default = 1.25
              gamma: (float) kinetic coefficient in LK equation (~wall mobility). Default = 2.0
              k: (float or list of floats) coupling constant(s) for nearest neighbors in this model. Default = 1.0. 
                    If providing a list, it should be of length (n*n). 
                    This is provided in case you want to model random-bond disorder
              
              time_vec: (optional) (numpy array of size (time_steps,) If passing an applied electric field,
              then also pass the time vector,  i.e. np.linspace(0,max_time, time_steps)
              
              appliedE: (optional) (numpy array of size (time_steps, 2) indicating applied electric field 
              in units of Ec (Ec is nominal coercive field) for both x and y components
                                    
              defects: (optional) list of tuples of length (n*n) with each tuple (RFx, RFy) being floats. 
              If none provided, then (RFx,RFy) is (0,0) at all sites.
              This stipulates the strength of the random field defects at each site and will be multipled by Ec, the nominal coercive field
              r: (float) Radius for nearest neighbor correlations. Default = 1.1 (nearest neighbor only)
              rTip: (int) Tip radius. This is not really used properly so ignore for timebeing. Default = 3
              
              
              
    """

    def __init__(self, n=10, alpha=-1.6, beta=1.0, gamma=1.0,
                 k=1, r=1.1,rTip = 3.0, dep_alpha = 0.2,
                 time_vec = None, defects = None,
                 appliedE = None, initial_p = None):

        self.alpha = alpha #TODO: Need to add temperature dependence
        self.beta = beta
        self.gamma = gamma  # kinetic coefficient
        self.rTip = rTip  # Radius of the tip
        self.r = r  # how many nearest neighbors to search for (radius)
        self.n = n
        self.Pmat = []

        #Now we have to see if matrices were passed for the coupling constants and depolarization alphas
        if len(np.array([k]))==1:
            self.k = np.full(shape=(self.n*self.n), fill_value = k)
        else:
            #Do some checks
            if len(k)!=n*n:
                raise ShapeError("Length of provided coupling list should be {}, instead received {}".format(n,len(k)))
            else:
                self.k = k

        if len(np.array([dep_alpha]))==1:
            self.dep_alpha = np.full(shape=(self.n*self.n), fill_value=dep_alpha)
        else:
            # Do some checks
            if len(dep_alpha)!=n*n:
                raise ShapeError("Length of provided coupling list should be {}, instead received {}".format(n,len(dep_alpha)))
            else:
                self.dep_alpha = dep_alpha

        self.Ec = (-2 / 3) * self.alpha * np.sqrt((-self.alpha / (3 * self.beta)))
        if time_vec is not None or appliedE is not None:
            if appliedE is None and time_vec is not None:
                raise ValueError("You have supplied a time vector but not a field vector. This is not allowed. You must supply both")
            if appliedE is None and time_vec is not None:
                raise ValueError("You have supplied a field vector but not a time vector. This is not allowed. You must supply both")
        elif time_vec is None and appliedE is None:
            #these will be the defaults for the field, i.e. in case nothing is passed
            self.t_max = 1.0
            self.time_steps = 1000
            self.time_vec = np.linspace(0, self.t_max, self.time_steps)
            self.appliedE = np.zeros((self.time_steps, 2))
            self.appliedE[:, 1] = 10*self.Ec * np.sin(self.time_vec[:] * 2 * np.pi * 2) #field is by default in y direction

        if time_vec is not None and appliedE is not None:
            self.t_max = np.max(time_vec)
            self.time_steps = len(time_vec)
            if appliedE.shape[1]!=2: raise ShapeError ("Shape of applied field should be (time_steps,2). Input correct shape.")
            self.appliedE = appliedE*self.Ec
            self.time_vec = time_vec

        #We are going to define random fields based on the input of defects
        #defects will be a list of tuples of size N with the Ex, Ey field components
        #the field will be given in units of Ec, where Ec is defined for the pristine state (-2/3)alpha*(-alpha/3*beta)^1/2

        if defects is None:
            self.Eloc = [(0,0) for ind in range(self.n*self.n)]
        else:
            if len(defects)!=self.n*self.n: raise ShapeError("Length of defects is not equal to total number of lattice sites. Pass correct shape ")
            self.Eloc = [(Ex*self.Ec,Ey*self.Ec) for (Ex,Ey) in defects] #We will worry about random bond defects later.

        pr = -1 * np.sqrt(-self.alpha / self.beta) #/ (self.n * self.n)  # Remnant polarization, y component

        if initial_p is None: self.initial_p = [0,pr] #assuming zero x component
        else: self.initial_p = initial_p

        self.atoms = self.setup_lattice()  # setup the lattice

    def setup_lattice(self):
        atoms = []
        pos_vec = np.zeros(shape=(2, self.n * self.n))

        for i in range(self.n):
            for j in range(self.n):
                atoms.append(
                    Lattice(self.initial_p, (i, j), len(self.time_vec)))  # Make lattice objects for each lattice site.

        for ind in range(len(atoms)):
            pos_vec[:, ind] = np.array(atoms[ind].getPosition())  # put the positions into pos_vec

        for ind in range(len(atoms)):  # calculate and update the neighborhood for each atom
            atom_pos = np.reshape(np.array(atoms[ind].getPosition()), (2, 1))
            # Find the neighbors
            neigh_idx, d = self.knn_search(atom_pos, pos_vec, self.r)
            poi = []
            dist = []
            for value in neigh_idx[0]:
                poi.append((pos_vec[0, value], pos_vec[1, value]))  # stick them into a list

            atoms[ind].updateNeighbors(poi)
            atoms[ind].updateNeighborIndex(neigh_idx)
            atoms[ind].updateDistance(d[neigh_idx])

        return atoms

    def runSim(self, calc_pr = False):
        dpdt, pnew = self.calcODE()

        P_total = np.sum(pnew, 0)
        dP_total = np.sum(dpdt, 0)

        # Now simulate the real measurement with probing volume

        measured_resp = np.zeros(shape=(self.n, self.n, len(self.time_vec)))  # placeholder
        if calc_pr:

            for i in np.arange(self.rTip + 1, self.n - self.rTip - 1):
                for j in np.arange(self.rTip + 1, self.n - self.rTip - 1):

                    probe_vol = self.makeCircle(self.n, self.n, i, j, self.rTip)

                    for t in np.arange(0, len(self.time_vec)):
                        pimage = pnew[:, t].reshape((self.n, self.n))
                        measured_resp[i, j, t] = np.sum(probe_vol * pimage)
            measured_resp = measured_resp[(self.rTip+1):-(self.rTip+1),(self.rTip+1):-(self.rTip+1),:]
            line_Pvals = np.copy(measured_resp).reshape(-1, len(self.time_vec))

            results = {'Polarization': P_total, 'dPolarization': dP_total, 'measuredResponse': measured_resp,
                       'line_polarization_values': line_Pvals
                       }
        else:
            results = {'Polarization': P_total, 'dPolarization': dP_total
                       }

        self.results = results

        return results

    def getPNeighbors(self, index, t):
        """This function will return the sum of the polarization values of
        the nearest neighbors as a list, given the index and the time step"""

        nhood_ind = self.atoms[index].getNeighborIndex()
        nhood = np.ravel(nhood_ind[0])
        p_nhood = []

        for index in nhood:
            p_val = self.atoms[index].getP(t)
            p_nhood.append(p_val)

        return np.sum(p_nhood,axis=0)

    def calDeriv(self, index, p_n, sum_p, Evec, total_p):

        #total_p should be a tuple with (px, py).

        p_nx = p_n[0] # x component
        p_ny = p_n[1] # y component

        sum_px= sum_p[0]
        sum_py= sum_p[1]

        Evec_x = Evec[0]
        Evec_y = Evec[1]

        Eloc_x = Evec_x - self.dep_alpha[index] * total_p[0] + self.Eloc[index][0]
        Eloc_y = Evec_y - self.dep_alpha[index] * total_p[1] + self.Eloc[index][1]

        xcomp_derivative = -self.gamma * (self.beta * p_nx ** 3 + self.alpha * p_nx + self.k[index] * (p_nx - sum_px/4) - Eloc_x)
        ycomp_derivative = -self.gamma * (self.beta * p_ny ** 3 + self.alpha * p_ny + self.k[index] * (p_ny - sum_py/4) - Eloc_y)

        return [xcomp_derivative, ycomp_derivative]

    def calcODE(self):

        # Calculate the ODE (Landau-Khalatnikov 4th order expansion), return dp/dt and P

        N = self.n * self.n
        dpdt = np.zeros(shape=(N, 2, len(self.time_vec))) #N lattice sites, 2 dim (x,y) for P, 1 time dim
        pnew = np.zeros(shape=(N, 2, len(self.time_vec)))
        dt = self.time_vec[1] - self.time_vec[0]

        # For t = 0
        pnew[:,:, 0] = self.initial_p

        # t=1
        for i in range(N):

            p_i = pnew[i, :, 0]
            sum_p = self.getPNeighbors(i, 0)

            total_px = np.sum(pnew[:,0,0])
            total_py = np.sum(pnew[:,1,0])
            total_p = (total_px, total_py)

            dpdt[i, :, 1] = self.calDeriv(i, p_i, sum_p, self.appliedE[1,:], total_p)
            pnew[i, :, 1] = p_i + dpdt[i,:, 1] * dt

            self.atoms[i].setP(1, p_i + dpdt[i, :,1] * dt)
        #t>1
        for t in np.arange(2, len(self.time_vec)):

            dt = self.time_vec[t] - self.time_vec[t-1]

            for i in np.arange(0, N):
                p_i = pnew[i,:, t - 1]
                sum_p = self.getPNeighbors(i, t - 1)
                total_px = np.sum(pnew[:, 0, t-1])
                total_py = np.sum(pnew[:, 1, t-1])
                total_p = (total_px,total_py)

                dpdt[i,:, t] = self.calDeriv(i, p_i, sum_p, self.appliedE[t,:], total_p)
                pnew[i,:, t] = p_i + dpdt[i,:, t] * dt
                self.atoms[i].setP(t, p_i + dpdt[i, :,t] * dt)

        return dpdt, pnew

    def plot_quiver(self, time_step = None):
        #Plots a time sequence of P maps as quiver plots
        #if time step is provided then it plots only that time step

        if time_step is None:
            time_steps_chosen = [int(x) for x in np.linspace(0, self.time_steps-1, 9)]

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (16,16))
            for ind, ax in enumerate(axes.flat):
                time_step = time_steps_chosen[ind]
                Pvals = np.zeros(shape=(2, self.n * self.n))

                for i in range(self.n * self.n):
                    Pvals[:,i] = self.atoms[i].getP(time_step)

                Pvals = np.reshape(Pvals, (2, self.n, self.n))
                ax.quiver(Pvals[0,:,:], Pvals[1,:,:])
                ax.set_title('Polarization map at t = ' + str(time_step))
                ax.set_xlim(-2,self.n+1)
                ax.set_ylim(-2,self.n+1)
                ax.axis('tight')
        else:
            fig, axes = plt.subplots(figsize=(6, 6))
            Pvals = np.zeros(shape=(2, self.n * self.n, 1))
            for i in range(self.n * self.n):
                Pvals[:,i] = self.atoms[i].getP(time_step)
            Pvals = np.reshape(Pvals, (2, self.n, self.n))
            axes.quiver(Pvals[0,:,:], Pvals[1,:,:])
            axes.set_title('Polarization map at t = ' + str(time_step))
            axes.set_xlim(-2, self.n + 1)
            axes.set_ylim(-2, self.n + 1)
            axes.axis('tight')


        fig.tight_layout()
        return fig

    def plot_summary(self):

        fig101 = plt.figure(101)
        plt.plot(self.time_vec[:], self.results['Polarization'][0,:], label = 'Px')
        plt.plot(self.time_vec[:], self.results['Polarization'][1, :], label='Py')
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Polarization')
        plt.legend(loc = 'best')

        fig102 = plt.figure(102)
        plt.plot(self.appliedE[:], self.results['Polarization'][0,1:] , label = 'Px')
        plt.plot(self.appliedE[:], self.results['Polarization'][1, 1:], label = 'Py')
        plt.xlabel('Field (a.u.)')
        plt.ylabel('Total Polarization')
        plt.legend(loc='best')

        fig103 = plt.figure(103)
        plt.plot(self.time_vec[1:], self.appliedE[:])
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Field (a.u.)')

        S = self.results['Polarization'] ** 2
        fig104 = plt.figure(104)
        plt.plot(self.appliedE[:], S[0,1:] ,label = 'Sx')
        plt.plot(self.appliedE[:], S[1, 1:], label='Sy')
        plt.xlabel('Field (a.u.)')
        plt.ylabel('Amplitude')
        plt.legend(loc='best')

        return [fig101, fig102, fig103, fig104]
    
    def getPmat(self):
        "Returns the polarization matrix of shape (2,t,N,N) after simulation has been executed."
        Pmat = np.zeros(shape=(2,self.time_steps, self.n, self.n))

        for ind in range(self.time_steps):
            Pvals = np.zeros(shape=(2, self.n * self.n))
            for i in range(self.n * self.n):
                Pvals[:,i] = self.atoms[i].getP(ind)
            Pmat[:, ind, :, :] = Pvals[:,:].reshape(2,self.n, self.n)
            
        self.Pmat = Pmat
        return Pmat

    def make_interactive_plot(self):
        if self.Pmat == []: self.getPmat()

        fig2 = plt.figure(constrained_layout=True, figsize=(6, 6))
        spec2 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig2, )
        ax1 = fig2.add_subplot(spec2[:-1, :])
        ax2 = fig2.add_subplot(spec2[-1, :])
        Pmat = self.Pmat
        time_vec = self.time_vec
        applied_field = self.appliedE

        def updateData(curr):
            if curr <= 2: return
            for ax in (ax1, ax2):
                ax.clear()
            ax1.quiver(Pmat[0, curr, :, :], Pmat[1, curr, :, :])
            ax1.set_title('Time Step: {}'.format(curr))
            ax1.axis('off')
            ax1.axis('equal')
            # Electric field in x direction
            ax2.plot(time_vec, applied_field[:, 0], 'k--', label='$E_x$')
            ax2.plot(time_vec[curr], applied_field[curr, 0], 'ro')

            # Electric field in y direction
            ax2.plot(time_vec, applied_field[:, 1], 'b--', label='$E_y$')
            ax2.plot(time_vec[curr], applied_field[curr, 1], 'ko')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Field (/$E_c$)')


        fig2.tight_layout()
        simulation = animation.FuncAnimation(fig2, updateData, interval=50, frames=range(0, self.time_steps, 5), repeat=False)

        plt.show()

        return

    @staticmethod
    def makeCircle(xsize, ysize, xpt, ypt, radius):
        # Returns a numpy array with a circle in a grid of size (xsize,ysize)
        # with circle radius 'radius' and centered on (xpt,ypt).

        a = np.zeros((xsize, ysize)).astype('uint8')
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x ** 2 + y ** 2 <= radius ** 2
        a[ypt - radius:ypt + radius, xpt - radius:xpt + radius][index] = 1

        return a

    @staticmethod
    def knn_search(x, D, R):
        """
        Find nearest neighbours of data among D. Return only those for which r<R
        """
        ndata = D.shape[1]
        sqd = np.sqrt(((D - x[:, :ndata]) ** 2).sum(axis=0))
        sqd = np.array(sqd)
        # return the indexes of nearest neighbours within radius R
        idx = (np.where((sqd > 0) & (sqd <= R)))
        return idx, sqd
