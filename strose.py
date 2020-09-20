import numpy as np
import simpy

def g_rand(mu, sigma=0, minimum=None, maximum=None, output_integers=True):
    """Helper function for generating pseudo-random numbers using a
    truncated gaussian distribution.
    
    mu: required, number for center of distribution
    sigma: required, standard deviation for distribution
    minimum: optional, may be a number to specify a minimum, or a string
        containing the letter 's' and a number to indicate the bound is
        mu - that number of standard deviations. May also be None to
        indicate no lower bound.
    maximum: optional, may be a number to specify a maximum, or a string
        containing the letter 's' and a number to indicate the bound is
        mu + that number of standard deviations.  May also be None to
        indicate no upper bound.
    """
    
    n = np.random.normal(mu, sigma)
    #print(n)
    
    if maximum is not None:
        if isinstance(maximum, str) and 's' in maximum:
            sigmas_max = float(maximum.replace('s', ''))
            maximum = mu + (sigma * sigmas_max)
        
        if n > maximum:
            n = maximum
    
    if minimum is not None:
        if isinstance(minimum, str) and 's' in minimum:
            sigmas_min = float(minimum.replace('s', ''))
            minimum = mu - (sigma * sigmas_min)
        
        if n < minimum:
            n = minimum
    
    if output_integers:
        n = int(n)
    
    return n


def gen_resource_universe(resource_definitions, env):
    """Accepts a dictionary of Resource definitions, a la
    
    RESOURCE_DEFINITIONS = {
        'preop_slot' : { 'capacity': PREOP_SLOTS },
        'procedure_room' : { 'capacity': PROCEDURE_ROOMS },
        'recovery_slot' : { 'capacity': RECOVERY_SLOTS }
    }
    
    """
    return { k: simpy.Resource(env, v['capacity']) for k, v in resource_definitions.items() }


def gen_activity_universe(activity_definitions, resources_all):
    """Accepts a dictionary of CareActivity definitions, a la
    
    ACTIVITY_DEFINITIONS = {
    'preop' : {
        'time_requirements': {
            'mu': 30, #'sigma': 10, #'minimum': 30, #'maximum': '3s'
            },
        'required_resources': ['preop_slot']
        }
    }
    
    """
    
    return { k: CareActivity(v['time_requirements'],
             required_resources=[resources_all[r] for r in v['required_resources']],
             label=k) for k, v in activity_definitions.items() }


def extract_event_data(patient_list, append_data={}):
    """Helper function to put patient event data into a list of dicts
    
    patient_list: required, iterable of Patient instances
    append_data: optional dictionary of key: value pairs to tack on, such as an iteration number
    
    """

    event_data = []
    for patient_number, patient in enumerate(patient_list):
        #print(patient.journal)
        for e in patient.journal:
            datum = {'patient': patient_number,
                    'entry': e['entry'],
                    'timestamp': e['timestamp']}
            datum.update(append_data)
            #e.update({'patient': patient_number})
            #print(e)
            #e['patient'] = patient_number
            event_data.append(datum)
            #for key, value in e.items():
            #    d = {'patient': patient_number}
            #    d.update(e)
            #    event_data.append(d)
    
    return event_data


class Patient(object):
    def __init__(self, needs=[], label=None, logger=None,
                 needs_list=None, activity_universe=None):
        """Can pass with a list of needs (Need instances) or, alternatively,
        a needs_list with labels of required CareActivity instances.
        """
        self.label = label
        self.needs = needs # list of Need instances
        self.logger = logger
        self.journal = []
        
        if len(self.needs) == 0 and all([isinstance(i, str) for i in needs_list]):
            # no needs were supplied
            # but we did get a list of strings in needs_list
            # which ought to be a bunch of labels for CareActivity instances
            
            # overwrite the needs with these new, matched ones
            self.needs = [self._need_from_activity_label(n, activity_universe) for n in needs_list]
    
    def __str__(self): # human-readable string
        return "<Patient {}>".format(self.label)
    
    def __repr__(self): # unambiguous representation
        return "<Patient instance: label={}, status={}, unmet_need_count={}>".format(self.label, self.status, self.unmet_need_count)
    
    def _need_from_activity_label(self, activity_label, activity_universe):
        """Internal helper function for 
        
        label : required
            The target CareActivity from which to generate a Need object
        activity_universe : required
            A dictionary of pre-defined label: CareActivity pairs
        """
        return Need(activity_universe[activity_label])
    
    def _hr_route(self):
        """Kick out a human-readable CareActivity flow for this patient
        """
        return ' --> '.join([n.care_activity_label for n in self.needs])
    
    def _unmet_need_count(self):
        return sum([1 if not n.met == True else 0 for n in self.needs])
    
    def _needs_met(self):
        return self._unmet_need_count == 0
    
    def _string_unmet_needs(self, separator=', '):
        return separator.join([n.label for n in self.needs if not n.met])
    
    def get_next_unmet_need(self):
        for n in self.needs:
            if n.is_unmet:
                return n
        
        return None
    
    
    # properties for easy access
    hr_route = property(_hr_route)
    unmet_need_count = property(_unmet_need_count)
    needs_met = property(_needs_met)
    next_unmet_need = property(get_next_unmet_need)
    string_unmet_needs = property(_string_unmet_needs)
    
    
    # if timestamp is passed here, it obviates the need to pass env at
    # instantiation and store it above
    def status_update(self, status, timestamp=None, log=True,
                      loglevel=20, journal=False, chunk_separator=';;'):
        """
        loglevel: default 20, this is a Python logger object loglevel. Note
            that 20 corresponds to 'INFO' level logging.
        """
        self.status = status
        if log == True and self.logger is not None:
            message = chunk_separator.join([str(timestamp), str(self), self.status])
            self.logger.log(loglevel, message)
        
        if journal == True:
            self.journal_entry(self.status, timestamp)
    
    # this may seem like a duplicate record, but it keeps these entries
    # stored on the Patient instance for more convenient inspection
    def journal_entry(self, entry, timestamp=None):
        self.journal.append({'timestamp': timestamp,
                            'entry': entry})


class CareActivity(object):
    """Object archtype for things patients require to progress
    their care. For example, a Procedure, a PreOpCheckIn,
    a PacuRecovery.
    """
    def __init__(self, time_requirements, required_resources=[],
                 label='Untitled', keep_resources_until_next_activity=True):

        self.time_requirements = time_requirements
        self.label = label
        self._keep_resources_until_next_activity = keep_resources_until_next_activity
        
        if all([isinstance(r, simpy.Resource) for r in required_resources]):
            self.required_resources = required_resources
        else:
            raise Exception("required_resources must be iterable containing only simpy Resource objects or subclasses thereof")


class Need(object):
    def __init__(self, care_activity, met=False):
        """Belongs to a Patient and connects to an CareActivity (or CareActivity
        subclass). 
        """
        
        if isinstance(care_activity, CareActivity):
            self.care_activity = care_activity
        else:
            raise Exception("obj_activity must be a CareActivity object or subclass thereof")
        
        self.met = met
        self._met_timestamp = None
    
    def _status(self):
        if self.met:
            return 'met at {}'.format(self._met_timestamp)
        else:
            return 'unmet'
    
    def _care_activity_label(self):
        """Shortcut to associated CareActivity label"""
        return self.care_activity.label
    
    def _is_unmet(self):
        if self.met:
            return False
        else:
            return True
    
    def _get_label(self):
        return self.care_activity.label
    
    def _time_reqs(self):
        return self.care_activity.time_requirements
    
    def _meet(self, timestamp=None):
        self.met = True
        self._met_timestamp = timestamp
    
    care_activity_label = property(_care_activity_label)
    is_unmet = property(_is_unmet)
    label = property(_get_label)
    status = property(_status)
    time_requirements = property(_time_reqs)
    
    
    def __str__(self): # human-readable string
        return "<Need: {}, {}>".format(self.care_activity_label, self.status)
    
    def __repr__(self): #unambiguous string representation
        return "<Need object instance, CareActivity.label='{}', status='{}'>".format(self.care_activity_label, self.status)


class SimulatedUnit(object):
    def __init__(self, env, resources={}, care_activities={}):
        """For example, an outpatient surgery center, a GI suite, an inpatient
        acute care ward, an intensive care unit, an emergency department, etc.
        """
        self.env = env
        self.patients = []
        self.resources = resources
        self.care_activities = {}
        
    
    def _patient_count(self):
        return len(self.patients)
    
    # collate properties
    patient_count = property(_patient_count)
    
    
    def provide_care(self, patient):
        
        while not patient.next_unmet_need == None:
            # as long as we have unmet needs,
            # get the next unmet need
            # and try to meet it
            
            patient.status_update('has {} unmet need(s): {}'.format(patient.unmet_need_count,
                                                                    patient.string_unmet_needs), self.env.now)

            # identify next unmet need in this patient's care sequence
            n = patient.next_unmet_need

            patient.status_update('unmet need found: {}'.format(n.label), self.env.now)
            patient.status_update('{}:queue'.format(n.label), journal=True, timestamp=self.env.now)
            patient.status_update('unmet need {} requires resources: {}'.format(n.label, n.care_activity.required_resources), self.env.now)

            reqs = [r.request() for r in n.care_activity.required_resources]
            reqs_allof = simpy.AllOf(self.env, reqs)

            # wait til all required resources are available
            yield reqs_allof
            patient.status_update('unmet need {} required resources available: {}'.format(n, reqs_allof.ok), self.env.now)

            # meet that need
            patient.status_update('{}:start'.format(n.label), journal=True, timestamp=self.env.now)
            yield self.env.timeout(g_rand(**n.time_requirements))
            timestamp_met = self.env.now
            patient.status_update('timeout elapsed, activity {} complete'.format(n.care_activity), self.env.now)

            n._meet(timestamp_met)
            patient.status_update('need met {}'.format(n), self.env.now)
            
            patient.status_update('{}:end'.format(n.label), journal=True, timestamp=self.env.now)

            # release occupied resources
            [resource.release(reqs[i]) for i, resource in enumerate(n.care_activity.required_resources)]
            patient.status_update('resources released {}'.format(n.care_activity.required_resources), self.env.now)
        
        patient.status_update('Complete')
