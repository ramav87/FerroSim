import numpy as np
import matplotlib.pyplot as plt

from .lattice import Lattice

class Ferro2DSim:
    """Ferro2DSim class that will setup and perform an
    Ising-like simulation of P vectors in a ferroelectric double well
      Random field defects are supported. The model is based on the paper by Ricinschii et al.
      J. Phys.: Condens. Matter 10 (1998) 477–492.
      Don't take this model too seriously. It's grossly wrong, but it's still fun.

    The methods available are:

    setPosition(): sets the (x,y) location of the object
    getPosition(): returns the position of the object

    setP(): sets the polarization value
    getP(): gets the polarization value

    calcDeriv(): calculates the derivative
    caldODE(): solve the ODE

    Inputs: - n: (int) Size of lattice. Default = 10
              alpha: (float) alpha term in Landau expansion. Default = -1.85
              beta: (float) beta term in Landau expansion. Default = 1.25
              gamma: (float) kinetic coefficient in LK equation (~wall mobility). Default = 2.0
              k: (float) coupling constant for nearest neighbors in this model. Default = 1.0. Would be lower near PT/for relaxors, etc.
              r: (float) Radius for nearest neighbor correlations. Default = 1.1 (nearest neighbor only)
              t_max: (float) time max (will create a time vector from [0,time_max] with 1000 steps). Default = 1.0
              E_frac: (float), Electric field maximum as a ratio over coercive field. Default = 80
              defect_number: (int) number of defects (must be less than n^2). Will place defects at random sites.
              rfield_strength: (float) Strength of the random field, as a fraction of Ec. E_bi Will be randomly distributed around this value.
    """

    def __init__(self, n=10, alpha=-1.85, beta=1.25, gamma=2.0,
                 E_frac=-80, k=1, r=1.1, t_max=1.0, defect_number=5, rfield_strength=2.0, rTip = 3.0):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # kinetic coefficient
        self.E_frac = E_frac
        self.k = k
        self.n = n
        self.r = r
        self.t_max = t_max
        self.defect_number = defect_number
        self.rfield_strength = rfield_strength
        self.rTip = rTip  # Radius of the tip
        self.time_steps = 1000  # hard wired for now
        self.pval = 1.0  # hard wired for now. Polarization at 0th time step

        # Pass the polarization value along with (x,y) tuple for location of atom
        self.P = np.zeros(shape=(self.time_steps))
        self.P[0] = self.pval
        self.position = (0, 0)
        self.pr = -1 * np.sqrt(-self.alpha / self.beta) / (self.n * self.n)  # Remnant polarization
        self.E, self.appliedE, self.time_vec = self.setup_field()  # setup the electic field

        self.atoms = self.setup_lattice()  # setup the lattice

    def setup_field(self):
        time_vec = np.linspace(0, self.t_max, self.time_steps)

        # Field and degraded regions
        E = np.zeros(shape=(len(time_vec), self.n * self.n))
        Ebi = np.zeros(shape=(self.n * self.n))
        Ec = (-2 / 3) * self.alpha * np.sqrt((-self.alpha / (3 * self.beta)))
        E0 = self.E_frac * Ec
        rand_selection = np.random.choice(range(self.n * self.n), size=self.defect_number, replace=False)
        Ebi[rand_selection] = np.random.rand(len(rand_selection)) * Ec * self.rfield_strength

        for i in range(self.n * self.n):
            E[1:, i] = E0 * np.sin(time_vec[1:] * 2 * np.pi * 2) + Ebi[i]

        appliedE = E0 * np.sin(time_vec[1:] * 2 * np.pi * 2)

        return E, appliedE, time_vec

    def setup_lattice(self):
        atoms = []
        pos_vec = np.zeros(shape=(2, self.n * self.n))

        for i in range(self.n):
            for j in range(self.n):
                atoms.append(
                    Lattice(self.pr, (i, j), len(self.time_vec)))  # Make lattice objects for each lattice site.

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

    def runSim(self):
        dpdt, pnew = self.calcODE()

        P_total = np.sum(pnew, 0)
        dP_total = np.sum(dpdt, 0)

        # Now simulate the real measurement with probing volume
        measured_resp = np.zeros(shape=(self.n, self.n, len(self.time_vec)))  # placeholder

        for i in np.arange(self.rTip + 1, self.n - self.rTip - 1):
            for j in np.arange(self.rTip + 1, self.n - self.rTip - 1):

                probe_vol = self.makeCircle(self.n, self.n, i, j, self.rTip)

                for t in np.arange(0, len(self.time_vec)):
                    pimage = pnew[:, t].reshape((self.n, self.n))
                    measured_resp[i, j, t] = np.sum(probe_vol * pimage)
        measured_resp = measured_resp[(self.rTip+1):-(self.rTip+1),(self.rTip+1):-(self.rTip+1),:]
        line_Pvals = np.copy(measured_resp).reshape(-1, len(self.time_vec))
        #max_val = np.max(line_Pvals)

        results = {'Polarization': P_total, 'dPolarization': dP_total, 'measuredResponse': measured_resp,
                   'line_polarization_values': line_Pvals
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

        return np.sum(p_nhood)

    def calDeriv(self, p_n, sum_p, Evec):
        return -self.gamma * (self.beta * p_n ** 3 + self.alpha * p_n + self.k * (2 * p_n - sum_p) - Evec)

    def calcODE(self):

        # Calcualte the ODE (Landau-khalatnikov 4th order expansion), return dp/dt and P
        N = self.n * self.n

        dpdt = np.zeros(shape=(N, len(self.time_vec)))
        p = np.zeros(shape=(N, len(self.time_vec)))
        pnew = np.zeros(shape=(N, len(self.time_vec)))

        dt = self.time_vec[1] - self.time_vec[0]
        pr = -self.gamma * np.sqrt(-self.alpha / self.beta)  # Remnant polarization

        # For t = 0
        p[:, 0] = pr
        pnew[:, 0] = pr

        # t=1
        for i in range(N):
            p_i = pnew[i, 0]
            sum_p = self.getPNeighbors(i, 0)

            dpdt[i, 1] = self.calDeriv(p_i, sum_p, self.E[1, i])
            pnew[i, 1] = p_i + dpdt[i, 1] * dt
            self.atoms[i].setP(1, p_i + dpdt[i, 1] * dt)

        for t in np.arange(2, len(self.time_vec)):

            for i in np.arange(0, N):
                p_i = pnew[i, t - 1]
                sum_p = self.getPNeighbors(i, t - 1)

                dpdt[i, t] = self.calDeriv(p_i, sum_p, self.E[t, i])
                pnew[i, t] = p_i + dpdt[i, t] * dt
                self.atoms[i].setP(t, p_i + dpdt[i, t] * dt)

        return dpdt, pnew

    def plot_quiver(self, time_step = None):
        #Plots a time sequence of P maps as quiver plots
        #if time step is provided then it plots only that time step

        U = np.full(shape=(self.n, self.n), fill_value=0) #x component is just zero

        if time_step is None:
            time_steps_chosen = [int(x) for x in np.linspace(0, self.time_steps-1, 25)]

            fig, axes = plt.subplots(nrows=5, ncols=5, figsize = (16,16))
            for ind, ax in enumerate(axes.flat):
                time_step = time_steps_chosen[ind]
                Pvals = np.zeros(shape=(self.n * self.n, 1))

                for i in range(self.n * self.n):
                    Pvals[i, :] = self.atoms[i].getP(time_step)

                Pvals = np.reshape(Pvals, (self.n, self.n))
                ax.quiver(U, Pvals)
                ax.set_title('Polarization map at t = ' + str(time_step))
                ax.set_xlim(-2,self.n+1)
                ax.set_ylim(-2,self.n+1)
                ax.axis('tight')
        else:
            fig, axes = plt.subplots(figsize=(6, 6))
            Pvals = np.zeros(shape=(self.n * self.n, 1))
            for i in range(self.n * self.n):
                Pvals[i, :] = self.atoms[i].getP(time_step)
            Pvals = np.reshape(Pvals, (self.n, self.n))
            axes.quiver(U, Pvals)
            axes.set_title('Polarization map at t = ' + str(time_step))
            axes.set_xlim(-2, self.n + 1)
            axes.set_ylim(-2, self.n + 1)
            axes.axis('tight')


        fig.tight_layout()
        return fig

    def plot_summary(self):

        fig101 = plt.figure(101)
        plt.plot(self.time_vec[1:], self.results['Polarization'][1:])
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Total Polarization')

        fig102 = plt.figure(102)
        plt.plot(self.appliedE[:], self.results['Polarization'][1:])
        plt.xlabel('Field (a.u.)')
        plt.ylabel('Total Polarization')

        fig103 = plt.figure(103)
        plt.plot(self.time_vec[1:], self.appliedE[:])
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Field (a.u.)')

        S = self.results['Polarization'] ** 2
        fig104 = plt.figure(104)
        plt.plot(self.appliedE[:], S[1:])
        plt.xlabel('Field (a.u.)')
        plt.ylabel('Amplitude')

        fig105 = plt.figure(105)
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize = (12,12))
        time_steps_chosen = [int(x) for x in np.linspace(1, self.time_steps-1, 25)]

        max_pr = np.max(self.results['measuredResponse'])
        min_pr = np.min(self.results['measuredResponse'])

        for ind, ax in enumerate(axes.flat):
            ax.imshow(self.results['measuredResponse'][:,:,time_steps_chosen[ind]])#  vmin = min_pr, vmax = max_pr
            ax.axis('off')
            ax.set_title('PR at t = {}'.format(time_steps_chosen[ind]))
        fig.tight_layout()

        return


    def make_interactive_plot(self):
        #Here we need to make an interactive plot that we can scrub through. It should also allow you to export to videos.
        #WIll add soon.

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