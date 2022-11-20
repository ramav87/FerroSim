import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from numba import jit
#TODO: Plotting might have to be taken out into a separete utils file. It's starting to take up too much room here.

from .lattice import Lattice

class Ferro2DSim:

    """Ferro2DSim class that will setup and perform an
    Ising-like simulation of P vectors in a ferroelectric double well
      Random field defects are supported. The model is based on the paper by Ricinschii et al.
      J. Phys.: Condens. Matter 10 (1998) 477–492, but has been considerably extended.
    
    All inputs are optional (they will default to values shown below)
    
    Inputs: - n: (int) Size of lattice. Default = 10

              gamma: (float) kinetic coefficient in LK equation (~wall mobility). Default = 2.0

              k: (float or list of floats) coupling constant(s) for nearest neighbors in this model. Default = 1.0. 
                    If providing a list, it should be of length (n*n). 
                    This is provided in case you want to model random-bond disorder
              
              time_vec: (optional) (numpy array of size (time_steps,) If passing an applied electric field,
              then also pass the time vector,  i.e. np.linspace(0,max_time, time_steps)
              
              appliedE: (optional) (numpy array of size (time_steps, 2) indicating applied electric field 
              in units of Ec (Ec is nominal coercive field) for both x and y components. Default is a sine wave
                                    
              defects: (optional) list of tuples of length (n*n) with each tuple (RFx, RFy) being floats. 
              If none provided, then (RFx,RFy) is (0,0) at all sites.
              This stipulates the strength of the random field defects at each site and will be multipled by Ec, the nominal coercive field

              r: (float) Radius for nearest neighbor correlations. Default = 1.1 (nearest neighbor only)

              rTip: (int) Tip radius. This is not really used properly so ignore for timebeing. Default = 3

              dep_alpha (float): Depolarization constant (will be multipled by total polarization and subtracted from energy at lattice site). Default = 0

              mode: (optional) (string) Mode of simulation. Can choose 'uniaxial', 'squreelectric', 'tetragonal' or 'rhomnohedral'. Consult documentation for details.

              landau_parms: (optional) (dictionary) . Dictionary containing Landau parameters. You will need to provide according to 'mode'. Consult documentation for details.

              Note: if you wish to change the coeffiicents in the functional, you should pass a dictionary
              with the necessary parameters upon initialization.
              
    """

    def __init__(self, n=10, gamma=1.0,
                 k=1, r=1.1,rTip = 3.0, dep_alpha = 0.0,
                 time_vec = None, defects = None,
                 appliedE = None, initial_p = None, init = 'pr', mode = 'tetragonal',
                 landau_parms = None, temp=None, stochastic = True):

        self.gamma = gamma  # kinetic coefficient
        self.rTip = rTip  # Radius of the tip
        self.r = r  # how many nearest neighbors to search for (radius)
        self.n = n
        self.mode = mode
        self.Pmat = []
        self._set_Landau_parms(landau_parms)
        self.temp = temp
        self.T0 = 400
        self.stochastic = stochastic #whether to use stochastic gradients. If True, 4x speedup
                                    #but may be less prone to convergence

        if self.temp is not None:
            if landau_parms is None:
                self.alpha1 = -1*self.alpha1*(self.temp - self.T0) #keep it negative
            else:
                self.alpha1 = self.alpha1 * (self.temp - self.T0)


        np.seterr('raise') #raise errors for debugging purpose

        #Now we have to see if matrices were passed for the coupling constants and depolarization alphas
        if len(np.array([k]))==1:
            self.k = np.full(shape=(self.n*self.n), fill_value = k)
        else:
            #Do some checks
            if len(k)!=n*n:
                raise AssertionError("Length of provided coupling list should be {}, instead received {}".format(n,len(k)))
            else:
                self.k = k

        if len(np.array([dep_alpha]))==1:
            self.dep_alpha = np.full(shape=(self.n*self.n), fill_value=dep_alpha)
        else:
            # Do some checks
            if len(dep_alpha)!=n*n:
                raise AssertionError("Length of provided coupling list should be {}, instead received {}".format(n,len(dep_alpha)))
            else:
                self.dep_alpha = dep_alpha

        self.Ec = (-2 / 3) * self.alpha1 * np.sqrt(np.abs(-self.alpha1 / (3 * self.alpha2)))
        if time_vec is not None or appliedE is not None:
            if appliedE is None and time_vec is not None:
                raise AssertionError("You have supplied a time vector but not a field vector. This is not allowed. You must supply both")
            if appliedE is None and time_vec is not None:
                raise AssertionError("You have supplied a field vector but not a time vector. This is not allowed. You must supply both")
        elif time_vec is None and appliedE is None:
            #these will be the defaults for the field, i.e. in case nothing is passed
            self.t_max = 1.0
            self.time_steps = 1000
            self.time_vec = np.linspace(0, self.t_max, self.time_steps)
            self.appliedE = np.zeros((self.time_steps, 2))
            self.appliedE[:, 1] = 40*self.Ec * np.sin(self.time_vec[:] * 2 * np.pi * 2) #field is by default in y direction

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

        pr = -1 * np.sqrt(np.abs(-self.alpha1 / self.alpha2)) #/ (self.n * self.n)  # Remnant polarization, y component
        if init =='pr':
            self.init = 'pr'
            if initial_p is None:
                self.initial_p = np.full((self.n, self.n, 2), 0)
                self.initial_p[:,:,0] = 0 #assume zero x component
                self.initial_p[:, :, 1] = pr
            else: self.initial_p = initial_p
        elif init =='random': self.init = 'random'

        self.atoms = self._setup_lattice()  # setup the lattice

    def _setup_lattice(self):
        atoms = []
        pos_vec = np.zeros(shape=(2, self.n * self.n))

        for i in range(self.n):
            for j in range(self.n):
                if self.init =='pr':
                    atoms.append(
                    Lattice(self.initial_p[i,j,:], (i, j), len(self.time_vec)))  # Make lattice objects for each lattice site.
                elif self.init =='random':
                    prand = tuple(0.5*np.random.randn(2))
                    atoms.append(
                    Lattice(prand, (i, j), len(self.time_vec)))  # Make lattice objects for each lattice site.

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

    def _set_Landau_parms(self, landau_parms):
        #Will set the Landau parameters

        if self.mode == 'uniaxial' or self.mode == 'squareelectric':
            #uniaxial and squareelectric model requires only alpha1, alpha2
            if landau_parms is not None:
                try:
                    self.alpha1 = landau_parms['alpha1']
                    self.alpha2 = landau_parms['alpha2']
                except KeyError:
                    print("Dictionary doesn't contain required keys. The keys for model {} \
                          are 'alpha1' and 'alpha2'.Reverting to default.".format(self.mode))
                    self.alpha1 = -1.85
                    self.alpha2 = 1.25
            else:
                #If we dont have a dictionary passed, use defaults
                self.alpha1 = -1.85
                self.alpha2 = 1.25
        elif self.mode == 'tetragonal':
            #tetragonal requires alpha1, alpha2 and alpha3
            if landau_parms is not None:
                try:
                    self.alpha1 = landau_parms['alpha1']
                    self.alpha2 = landau_parms['alpha2']
                    self.alpha3 = landau_parms['alpha3']
                except KeyError:
                    print("Dictionary doesn't contain required keys. The keys for model {} \
                          are 'alpha1', 'alpha2' and 'alpha3' .Reverting to default.".format(self.mode))
                    self.alpha1 = -1.6
                    self.alpha2 = 12.2
                    self.alpha3 = 40.0

            else:
                #If we dont have a dictionary passed, use defaults
                self.alpha1 = -1.6
                self.alpha2 = 12.2
                self.alpha3 = 40.0
        elif self.mode == 'rhombohedral':
            # rhombohedral requires alpha1, alpha2 and alpha3
            if landau_parms is not None:
                try:
                    self.alpha1 = landau_parms['alpha1']
                    self.alpha2 = landau_parms['alpha2']
                    self.alpha3 = landau_parms['alpha3']
                except KeyError:
                    print("Dictionary doesn't contain required keys. The keys for model {} \
                          are 'alpha1', 'alpha2' and 'alpha3' .Reverting to default.".format(self.mode))
                    self.alpha1 = -10.6
                    self.alpha2 = 10.2
                    self.alpha3 = -10.0
            else:
                # If we dont have a dictionary passed, use defaults
                self.alpha1 = -10.6
                self.alpha2 = 10.2
                self.alpha3 = -10.0
        else:
            raise ValueError ("You passed {} but allowable values are 'uniaxial', \
                              'squareelectric', 'tetragonal', 'rhombohedral' ".format(self.mode))

    def runSim(self, calc_pr = False, verbose = True):
        dpdt, pnew = self.calcODE(verbose = verbose, stochastic=self.stochastic)

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

    def runSimtoNextState(self, time_vec, start_p, verbose = False):
        dpdt, pnew = self.calcODE(verbose = verbose, 
        stochastic=self.stochastic, time_vec = time_vec, start_p = start_p)
        P_total = np.sum(pnew, 0)
        dP_total = np.sum(dpdt, 0)
        results = {'Polarization': P_total, 'dPolarization': dP_total}
        self.results = results
        return results

    def getPNeighbors(self, index, t):
        """This function will return the actual, and sum, of the polarization values of
        the nearest neighbors as a list, given the index and the time step"""

        nhood_ind = self.atoms[index].getNeighborIndex()
        nhood = np.ravel(nhood_ind[0])
        p_nhood = []

        for index in nhood:
            p_val = self.atoms[index].getP(t)
            p_nhood.append(p_val)

        return np.sum(p_nhood,axis=0), p_nhood

    @jit(nopython=False, forceobj=True)
    #TODO: Fix the numba stuff...

    def calDeriv(self, index, p_n, sum_p, Evec, total_p):

        #total_p should be a tuple with (px, py).
        
        p_nx = np.array(p_n[0], dtype = np.float64) # x component
        p_ny = np.array(p_n[1], dtype = np.float64) # y component

        #Sometimes we get convergence issues. So clip the x and y components to be within a range.
        p_nx = np.clip(p_nx, -10, 10)
        p_ny = np.clip(p_ny, -10, 10)

        sum_px= sum_p[0]
        sum_py= sum_p[1]

        Evec_x = Evec[0]
        Evec_y = Evec[1]

        Eloc_x = Evec_x - self.dep_alpha[index] * total_p[0]/(self.n*self.n) + self.Eloc[index][0]
        Eloc_y = Evec_y - self.dep_alpha[index] * total_p[1]/(self.n*self.n) + self.Eloc[index][1]

        if self.mode == 'squareelectric':
            xcomp_derivative = -self.gamma * (self.alpha2 * p_nx ** 3 +
                                              self.alpha1 * p_nx + self.k[index] * (p_nx - sum_px/4) - Eloc_x)

            ycomp_derivative = -self.gamma * (self.alpha2 * p_ny ** 3 +
                                               self.alpha1 * p_ny + self.k[index] * (p_ny - sum_py/4) - Eloc_y)
            derivative = [xcomp_derivative, ycomp_derivative]

        elif self.mode =='uniaxial':
            ycomp_derivative = -self.gamma * (self.alpha2 * p_ny ** 3 +
                                              self.alpha1 * p_ny + self.k[index] * (p_ny - sum_py / 4) - Eloc_y)
            derivative = [0, ycomp_derivative]
        elif self.mode =='rhombohedral' or self.mode=='tetragonal':

            new_x_derivative = -self.gamma*(
                    2*self.alpha1*p_nx + 4*self.alpha2*p_nx**3 + 2*self.alpha3*(p_ny**2) * p_nx +
                                    self.k[index] * (p_nx - sum_px / 4) - Eloc_x)

            new_y_derivative = -self.gamma * (
                    2*self.alpha1*p_ny + 4*self.alpha2*p_ny**3 + 2*self.alpha3*(p_nx**2) * p_ny +
                                    self.k[index] * (p_ny - sum_py / 4) - Eloc_y)
            derivative = [new_x_derivative, new_y_derivative]

        return derivative


    def calcODE(self, stochastic = True, verbose=True, time_vec = None, start_p = None):

        if time_vec is None: time_vec = self.time_vec
        if start_p is None: 
            init_p_new = [self.atoms[i].getP(0) for i in range(len(self.atoms))]
            init_p_new = np.array(init_p_new)
        else:
            init_p_new = start_p


        # Calculate the ODE (Landau-Khalatnikov 4th order expansion), return dp/dt and P
        N = self.n * self.n
        dpdt = np.zeros(shape=(N, 2, len(time_vec))) #N lattice sites, 2 dim (x,y) for P, 1 time dim
        pnew = np.zeros(shape=(N, 2, len(time_vec)))
        dt = time_vec[1] - time_vec[0]
        
        # For t = 0
        
        init_p_new = init_p_new.reshape(self.n*self.n,-1)
        pnew[:,:, 0] = init_p_new
        [self.atoms[i].setP(0, (px,py)) for i, (px,py) in enumerate(init_p_new)]

        # t=1
        
        for i in range(N):
            p_i = np.array(pnew[i, :, 0], dtype = np.float64)
            sum_p, _ = self.getPNeighbors(i, 0)
            sum_p = np.array(sum_p, dtype = np.float64)
            total_px = np.sum(pnew[:,0,0])
            total_py = np.sum(pnew[:,1,0])
            total_p = np.array((total_px, total_py), dtype = np.float64)

            dpdt[i, :, 1] = self.calDeriv(i, p_i, sum_p, self.appliedE[1,:], total_p)
            pnew[i, :, 1] = p_i + self.gamma*dpdt[i,:, 1] * dt

            self.atoms[i].setP(1, p_i + self.gamma*dpdt[i, :,1] * dt)
        #t>1
        if verbose==True: 
            print('---Performing simulation---')
            disable_tqdm = False
        else:
            disable_tqdm = True
        
        for t in tqdm(np.arange(2, len(time_vec)), disable = disable_tqdm):

            dt = time_vec[t] - time_vec[t-1]
            
            if stochastic:
                selected_idx = np.random.choice(np.arange(N), N//4, replace = False)
            else: 
                selected_idx = np.arange(0, N)

            for i in np.arange(0, N):
                p_i = pnew[i,:, t - 1]
                sum_p,_ = self.getPNeighbors(i, t - 1)
                total_px = np.sum(pnew[:, 0, t-1])
                total_py = np.sum(pnew[:, 1, t-1])
                total_p = (total_px,total_py)
                if i in selected_idx:
                    dpdt[i,:, t] = self.calDeriv(i, p_i, sum_p, self.appliedE[t,:], total_p)
                    pnew[i,:, t] = p_i + dpdt[i,:, t] * dt
                else:
                    dpdt[i,:, t] = dpdt[i,:, t-1]
                    pnew[i,:, t] = pnew[i,:, t-1]

                self.atoms[i].setP(t, p_i + self.gamma*dpdt[i, :,t] * dt)

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
        plt.plot(self.appliedE[:,0], self.results['Polarization'][0,:] , label = 'Px')
        plt.plot(self.appliedE[:,1], self.results['Polarization'][1, :], label = 'Py')
        plt.xlabel('Field (a.u.)')
        plt.ylabel('Total Polarization')
        plt.legend(loc='best')

        fig103 = plt.figure(103)
        plt.plot(self.time_vec[:], self.appliedE[:,0], label = 'Ex')
        plt.plot(self.time_vec[:], self.appliedE[:, 1], label='Ey')
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Field (a.u.)')
        plt.legend(loc = 'best')

        S = self.results['Polarization'] ** 2
        fig104 = plt.figure(104)
        plt.plot(self.appliedE[:,0], S[0,:] ,label = 'Sx')
        plt.plot(self.appliedE[:,1], S[1, :], label='Sy')
        plt.xlabel('Field (a.u.)')
        plt.ylabel('Amplitude')
        plt.legend(loc='best')

        return
    
    def getPmat(self, time_step=None):
        "Returns the polarization matrix of shape (2,t,N,N) after simulation has been executed, or at given time step"
        if time_step==None:

            Pmat = np.zeros(shape=(2,self.time_steps, self.n, self.n))
            for ind in range(self.time_steps):
                Pvals = np.zeros(shape=(2, self.n * self.n))
                for i in range(self.n * self.n):
                    Pvals[:,i] = self.atoms[i].getP(ind)
                Pmat[:, ind, :, :] = Pvals[:,:].reshape(2,self.n, self.n)

            self.Pmat = Pmat
        else:
            Pmat = np.zeros(shape=(2, self.n, self.n))
            Pvals = np.zeros(shape=(2, self.n * self.n))
            for i in range(self.n * self.n):
                Pvals[:, i] = self.atoms[i].getP(time_step)
            Pmat[:,:,:] = Pvals.reshape(2,self.n, self.n)

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
            _, angle = self.return_angles_and_magnitude(Pmat[:, curr, :, :])

            ax1.imshow(angle, cmap='bwr')
            #divider = make_axes_locatable(ax1)
            #cax = divider.append_axes('right', size='5%', pad=0.05)
            #cb1 = fig.colorbar(im1, cax=cax, orientation='vertical')
            #cb1.ax.set_ylabel('Angle (rad.)')
            ax1.quiver(Pmat[0, curr, :, :], Pmat[1, curr, :, :],
                           color='black', edgecolor='black', linewidth=0.5)

            #ax1.quiver(Pmat[0, curr, :, :], Pmat[1, curr, :, :])

            ax1.set_title('Time Step: {}'.format(curr))
            #ax1.axis('off')
            #ax1.axis('equal')
            # Electric field in x direction
            ax2.plot(time_vec, applied_field[:, 0], 'k--', label='$E_x$')
            ax2.plot(time_vec[curr], applied_field[curr, 0], 'ro')

            # Electric field in y direction
            ax2.plot(time_vec, applied_field[:, 1], 'b--', label='$E_y$')
            ax2.plot(time_vec[curr], applied_field[curr, 1], 'ko')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Field (/$E_c$)')


        fig2.tight_layout()
        sim_animation = animation.FuncAnimation(fig2, updateData, interval=50, frames=range(0, self.time_steps, 5), repeat=False)

        #plt.show()

        return sim_animation

    def plot_mag_ang(self, time_step=0, plot_arrows=True):
        '''Returns a plot of the P distribution as a magnitude/angle plot, and also provides the matrices
         Takes as input the time step (default = 0)'''
        if self.Pmat == []: 
            _ = self.getPmat();
        magnitude, angle = self.return_angles_and_magnitude(self.Pmat[:, time_step, :, :])

        # Plot it
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        im1 = axes[0].imshow(angle, cmap='bwr')
        axes[0].set_title('Angle')
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb1 = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb1.ax.set_ylabel('Angle (rad.)')
        if plot_arrows:
            axes[0].quiver(self.Pmat[0, time_step, :, :], self.Pmat[1, time_step, :, :],
                       color='black', edgecolor='black', linewidth=0.5)

        im2 = axes[1].imshow(magnitude, cmap='bwr')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb2 = fig.colorbar(im2, cax=cax, orientation='vertical')
        axes[1].set_title('Magnitude')
        cb2.ax.set_ylabel('Magnitude')
        if plot_arrows:
            axes[1].quiver(self.Pmat[0, time_step, :, :], self.Pmat[1, time_step, :, :],
                       color='black', edgecolor='black', linewidth=0.5)
        fig.tight_layout()

        return fig, magnitude, angle
    
    def plot_polar(self, time_step=0, plot_arrows=True):
        '''Returns a plot of the P distribution polar plot, and also provides the matrices
         Takes as input the time step (default = 0)'''
        time_step = int(time_step)
        if self.Pmat == []: 
            _ = self.getPmat();
        magnitude, angle = self.return_angles_and_magnitude(self.Pmat[:, time_step, :, :])
        polar_vec = magnitude*np.cos(angle)
        # Plot it
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
        im1 = axes.imshow(polar_vec, cmap='jet')
        axes.set_title('Polar Plot')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb1 = fig.colorbar(im1, cax=cax, orientation='vertical')
        cb1.ax.set_ylabel('Rcos($\phi$)')
        if plot_arrows:
            axes.quiver(self.Pmat[0, time_step, :, :], self.Pmat[1, time_step, :, :],
                       color='black', edgecolor='black', linewidth=0.5)

        fig.tight_layout()

        return fig, [magnitude, angle, polar_vec]

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

    @staticmethod
    def return_angles_and_magnitude(Pvals):
        # Pvals is of shape #2,n,n
        Pvals_lin = np.reshape(Pvals, (2, Pvals.shape[1] * Pvals.shape[2]))
        magnitude = np.zeros(Pvals.shape[1] * Pvals.shape[2])
        angle = np.zeros(Pvals.shape[1] * Pvals.shape[2])
        for ind in range(Pvals_lin.shape[-1]):
            magnitude[ind] = np.sqrt(Pvals_lin[0, ind] ** 2 + Pvals_lin[1, ind] ** 2)
            vector_1 = [1, 0]
            vector_2 = [Pvals_lin[0, ind], Pvals_lin[1, ind]]
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle[ind] = np.arccos(dot_product)

        return magnitude.reshape(Pvals.shape[1], Pvals.shape[2]), \
               angle.reshape(Pvals.shape[1], Pvals.shape[2])
    
    @staticmethod
    @jit(fastmath=True)
    def calc_curl(Pmat):
        #input the Pmat of size (2,N,N) and return the curl
        
        #the curl will be a scalar in the z direction    
        VxF =np.zeros((Pmat.shape[1],Pmat.shape[2]))

        for i in range(0,Pmat.shape[1]):
            for j in range(0,Pmat.shape[2]):
                #dP_y/dx - dP_x/dy
                VxF[i,j] = 0.5 * ((Pmat[1][(i+1)%Pmat.shape[2],j]-Pmat[1][i-1,j])-

                                (Pmat[0][i,(j+1)%Pmat.shape[1]]-Pmat[0][i,j-1]))

        return VxF

