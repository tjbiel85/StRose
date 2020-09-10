import strose
import unittest
import simpy
from scipy import stats

class TestGaussianRandomHelper(unittest.TestCase):
    """Battery of tests to make sure I don't break the g_rand() helper function"""
    def setUp(self, mu=0, sigma=1, iterations=100000, alpha=1e-6):
        self.mu = mu
        self.sigma = sigma
        self.iterations = iterations
        self.alpha = alpha
    
    def test_normal(self):
        """Test whether the function generates normally distributed samples"""
        samples = [strose.g_rand(mu=self.mu, sigma=self.sigma, output_integers=False) for x in range(self.iterations)]
        k2, p = stats.normaltest(samples)
        self.assertGreater(p, self.alpha)
    
    def test_upperBound(self, maximum=2):
        """Test whether an upper bound, as specified by keyword 'maximum', is respected"""
        samples = [strose.g_rand(self.mu, sigma=self.sigma, maximum=maximum, output_integers=False) for x in range(self.iterations)]
        self.assertTrue(all([x <= maximum for x in samples]))
    
    def test_lowerBound(self, minimum=-1):
        """Test whether a lower bound, as specified by keyword 'minimum', is respected"""
        samples = [strose.g_rand(self.mu, sigma=self.sigma, minimum=minimum, output_integers=False) for x in range(self.iterations)]
        self.assertTrue(all([x >= minimum for x in samples]))
    
    def test_sigmaUpperBound(self, sigma_max = '3s'):
        """Test whether an upper bound based on maximum standard deviations is respected"""
        true_maximum = self.mu + (3 * self.sigma)
        samples = [strose.g_rand(self.mu, sigma=self.sigma, maximum=sigma_max, output_integers=False) for x in range(self.iterations)]
        self.assertTrue(all([x <= true_maximum for x in samples]))
    
    def test_sigmaLowerBound(self, sigma_min='3s'):
        """Test whether a lower bound based on maximum standard deviations is respected"""
        true_minimum = self.mu - (3 * self.sigma)
        samples = [strose.g_rand(self.mu, sigma=self.sigma, minimum=sigma_min, output_integers=False) for x in range(self.iterations)]
        self.assertTrue(all([x >= true_minimum for x in samples]))


class TestNoWaitSimulation(unittest.TestCase):
    """Generate a basic simulation with no anticipated wait time. In this
    case, the number of pre-op slots, procedure rooms, recovery slots,
    and patients are all the same, and all the patients show up at the
    start of the simulation."""
    
    def setUp(self, iterations=10000, preop=30, procedure=90, pacu=60, patient_num=5):
        self.iterations = iterations
        self.preop_time = {'mu': preop, 'sigma': 0 }
        self.procedure_time = {'mu': procedure, 'sigma': 0 }
        self.pacu_time = {'mu': pacu, 'sigma': 0 }
        
        self.patients_start = patient_num
        self.patients_max = patient_num
        self.preop_slots = patient_num
        self.procedure_rooms = patient_num
        self.pacu_slots = patient_num
        
        self.generic_case = strose.Case(
            self.preop_time,
            self.procedure_time,
            self.pacu_time)
        
        self.anticipated_runtime = preop + procedure + pacu
        
        self.runtimes = []
        self.units = []
        for x in range(self.iterations):
            env = simpy.Environment()
            unit = strose.ProceduralUnit(env,
                                         self.preop_slots,
                                         self.procedure_rooms,
                                         self.pacu_slots)
            env.process(strose.run_simulation(unit, env,
                                          starting_patients=self.patients_start,
                                          total_patients=self.patients_max,
                                          standard_case=self.generic_case))
            env.run()
            self.units.append(unit)
            self.runtimes.append({'iteration': x, 'runtime': env.now})
    
    def test_uniformRuntimes(self):
        """Test whether total run times are uniform, as expected
        when there is no variability, bottlenecks, or waiting."""
        self.assertTrue(all([x['runtime'] == self.anticipated_runtime for x in self.runtimes]))
        
    def test_anticipatedPatientJournalEntries(self):
        """Test whether patients collect the expected series of journal entries.
        
        Should be 1:1 with DEFAULT_EVENT_LABELS entries
        """
        for unit in self.units:
            for p in unit.patients:
                self.assertTrue(all([x['entry'] == strose.DEFAULT_EVENT_LABELS[i] for i, x in enumerate(p.journal)]))
        
if __name__ == '__main__':
    unittest.main()