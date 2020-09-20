import unittest
import simpy
import pandas as pd
from scipy import stats
import strose

def run_simulation(unit, env, logger=None):
    """Assumes patients are in a collection at unit.patients
    """
    for p in unit.patients:
        env.process(unit.provide_care(p))
        yield env.timeout(0)


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
    
    def setUp(self, iterations=1000, patient_num=5, time_requirements=[30, 90, 60]):
        self._iterations = iterations
        self._patient_num = 5
        self._units = []
        self._runtimes = []
        
        self._resource_definitions = {
            'preop_slot' : { 'capacity': self._patient_num },
            'procedure_room' : { 'capacity': self._patient_num },
            'recovery_slot' : { 'capacity': self._patient_num }
        }

        self._activity_definitions = {
            'preop' : { 'time_requirements': { 'mu': time_requirements[0] }, 'required_resources': ['preop_slot'] },
            'procedure' : { 'time_requirements': { 'mu': time_requirements[1] }, 'required_resources': ['procedure_room'] },
            'recovery' : { 'time_requirements': { 'mu': time_requirements[2] }, 'required_resources': ['recovery_slot'] }
        }
        
        self._generic_patient_definition = { 'needs_list': ['preop', 'procedure', 'recovery' ] }
        self._anticipated_runtime = sum(time_requirements)
        
        
        for i in range(self._iterations):
            env = simpy.Environment()
            u = strose.SimulatedUnit(env, resources=strose.gen_resource_universe(self._resource_definitions, env))

            for p in range(self._patient_num):
                u.patients.append(strose.Patient(**self._generic_patient_definition,
                                          activity_universe=strose.gen_activity_universe(self._activity_definitions, u.resources)))

            env.process(run_simulation(u, env))
            env.run()

            self._units.append(u)
            self._runtimes.append({'iteration': i, 'runtime': env.now})
        
    
    def test_expectedRuntimes(self):
        """Ensure these multiple patients all finish at the expected time,
        given that they should never be waiting for Resource requests.
        """
        for r in self._runtimes:
            self.assertEqual(r['runtime'], self._anticipated_runtime)
    
    
    def test_uniformRuntimes(self):
        """Test whether total run times are uniform, as expected
        when there is no variability, bottlenecks, or waiting."""
        self.assertTrue(all([r['runtime'] == self._runtimes[0]['runtime'] for r in self._runtimes]))


class TestBottleneckedSimulation(unittest.TestCase):
    """Generate a basic simulation that is bottlenecked by limited Resource.
    """
    
    def setUp(self, iterations=1000):
        self._time_requirements = [30, 90, 60]
        self._patient_num = 5
        self._resource_capacity = 1
        self._anticipated_finishes = [180, 270, 360, 450, 540]

        self._iterations = 1000
        self._units = []
        self._runtimes = []
        self._finishes = []

        self._resource_definitions = {
            'preop_slot' : { 'capacity': self._resource_capacity },
            'procedure_room' : { 'capacity': self._resource_capacity },
            'recovery_slot' : { 'capacity': self._resource_capacity }
        }

        self._activity_definitions = {
            'preop' : { 'time_requirements': { 'mu': self._time_requirements[0] }, 'required_resources': ['preop_slot'] },
            'procedure' : { 'time_requirements': { 'mu': self._time_requirements[1] }, 'required_resources': ['procedure_room'] },
            'recovery' : { 'time_requirements': { 'mu': self._time_requirements[2] }, 'required_resources': ['recovery_slot'] }
        }

        self._generic_patient_definition = { 'needs_list': ['preop', 'procedure', 'recovery' ] }


        for i in range(self._iterations):
            env = simpy.Environment()
            u = strose.SimulatedUnit(env, resources=strose.gen_resource_universe(self._resource_definitions, env))

            for p in range(self._patient_num):
                u.patients.append(strose.Patient(**self._generic_patient_definition,
                                          activity_universe=strose.gen_activity_universe(self._activity_definitions, u.resources)))

            env.process(run_simulation(u, env))
            env.run()

            self._units.append(u)
            self._runtimes.append({'iteration': i, 'runtime': env.now})
            df = pd.DataFrame(strose.extract_event_data(u.patients))

            self._finishes.append(list(df[ df['entry'] == 'recovery:end' ]['timestamp']))
        
    
    def test_expectedFinishes(self):
        """Ensure these patients all finish at the expected time,
        given the wait for Resource requests.
        """
        self.assertTrue(all([f == self._anticipated_finishes for f in self._finishes]))
        
        
if __name__ == '__main__':
    unittest.main()