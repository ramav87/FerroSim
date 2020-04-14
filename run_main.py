from ferrosim import Ferro2DSim

if __name__ == "__main__":
    sim = Ferro2DSim()
    results = sim.runSim()
    print('Completed running')
    sim.plot_quiver()
