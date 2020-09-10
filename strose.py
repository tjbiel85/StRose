import simpy
import pandas as pd
import numpy as np

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

def extract_event_data(patient_list):
    """Helper function to put patient event data into a list of dicts
    
    patient_list: required, iterable of Patient instances"""

    event_data = []
    for patient_number, patient in enumerate(patient_list):
        #print(patient.journal)
        for e in patient.journal:
            datum = {'patient': patient_number,
                    'entry': e['entry'],
                    'timestamp': e['timestamp']}
            #e.update({'patient': patient_number})
            #print(e)
            #e['patient'] = patient_number
            event_data.append(datum)
            #for key, value in e.items():
            #    d = {'patient': patient_number}
            #    d.update(e)
            #    event_data.append(d)
    
    return event_data


# describe these discretely for uniform descriptions in logging/journaling
DEFAULT_EVENT_LABELS = [
    'arrived', # 0
    'ready for pre-op', # 1
    'assigned pre-op slot', # 2
    'completed pre-op', # 3
    'ready for procedure room', # 4
    'assigned procedure room', # 5
    'completed procedure', # 6
    'ready for PACU', # 7
    'assigned PACU slot', # 8
    'completed recovery' # 9
]

# object definitions
class ProceduralUnit(object):
    def __init__(self, env, num_preop_slots=1, num_procedure_rooms=1, num_pacu_slots=1,
                event_labels=DEFAULT_EVENT_LABELS):
        """For example, an outpatient surgery center, an hospital procedural care unit,
        a GI suite, an inpatient surgery unit, etc.
        """
        self.env = env
        self.preop_slot = simpy.Resource(env, num_preop_slots)
        self.procedure_room = simpy.Resource(env, num_procedure_rooms)
        self.pacu_slot = simpy.Resource(env, num_pacu_slots)
        self.patients = []
        self.event_labels = event_labels
    
    def _patient_count(self):
        return len(self.patients)
    
    patient_count = property(_patient_count)
    
    # need methods for processes involved in patient care here
    # hypothetically these could go within subclassed Resource objects or anywhere else
    # for now, keep them all here
    
    def preop_patient(self, patient):
        yield self.env.timeout(g_rand(**patient.case.preop_time_params))
    
    def procedurize_patient(self, patient):
        yield self.env.timeout(g_rand(**patient.case.procedure_time_params))
    
    def recover_patient(self, patient):
        yield self.env.timeout(g_rand(**patient.case.recovery_time_params))
    
    def process_encounter(self, patient):
        """Process a patient through the workflow for a daypatient
        procedural encounter. As in:
        
        pre-op --> procedure --> recovery
        
        TODO: can add an env.timeout(refresh resource) type call to
            simulate room cleaning before they are put back in the resource
            pool
        
        TODO: revisit event labels for patient status/journals. Hard-coded
            right now and not Pythonic or extensible.
        
        # arguments
        patient: required, a Patient instance
        preop_time_reqs: required, a dict containing arguments for g_rand; at the
            very least, must provide a 'mu' key with an average time for the
            process to take. In the absence of other arguments the average time will
            simply be the time required.
        preop_time_reqs: required, as per preop_time_reqs
        recovery_time_reqs: required, as per preop_time_reqs
        
        """
        patient.status_update(self.event_labels[0], self.env.now)
        arrival_time = self.env.now
        # three steps: preop, OR, PACU
        # could have some logging or timestamp type things on Patient objects
        # log things like wait time for pre-op slot, for an OR, for a PACU slot
        
        preop_slot = self.preop_slot.request()
        patient.status_update(self.event_labels[1], self.env.now)
        
        # wait until Resource is available
        yield preop_slot
        
        patient.status_update(self.event_labels[2], self.env.now)
        
        yield self.env.process(self.preop_patient(patient))
        patient.status_update(self.event_labels[3], self.env.now)
        
        procedure_room = self.procedure_room.request()
        patient.status_update(self.event_labels[4], self.env.now)
        # instantiated using "with" --> releases resource at completion

        # wait until Resource is available
        yield procedure_room
        patient.status_update(self.event_labels[5], self.env.now)
        
        # return pre-op slot to resource pool
        self.preop_slot.release(preop_slot)

        # get procedurized
        yield self.env.process(self.procedurize_patient(patient))
        patient.status_update(self.event_labels[6], self.env.now)
        
        pacu_slot = self.pacu_slot.request()
        
        patient.status_update(self.event_labels[7], self.env.now)    
        # wait until Resource is available (how to see queue size??)
        yield pacu_slot
        patient.status_update(self.event_labels[8], self.env.now)
        
        # return procedure_room to resource pool
        self.procedure_room.release(procedure_room)

        # get recovered
        yield self.env.process(self.recover_patient(patient))
        patient.status_update(self.event_labels[9], self.env.now)
        
        # return PACU slot to resource pool
        self.pacu_slot.release(pacu_slot)


class Patient(object):
    def __init__(self, case, label=None, logger=None):
        self.label = label
        self.case = case
        self.logger = logger
        self.journal = []
    
    def __str__(self):
        return "<Patient {}: {}>".format(self.label, self.status)
    
    # if timestamp is passed here, it obviates the need to pass env at
    # instantiation and store it above
    def status_update(self, status, timestamp=None, log=True,
                      loglevel=20, journal=True):
        """
        loglevel: default 20, this is a Python logger object loglevel. Note
            that 20 corresponds to 'INFO' level logging.
        """
        self.status = status
        if log == True and self.logger is not None:
            message = '[' + str(timestamp) + ']' + str(self)
            self.logger.log(loglevel, message)
        
        if journal == True:
            self.journal_entry(status, timestamp)
    
    # this may seem like a duplicate record, but it keeps these entries
    # stored on the Patient instance for more convenient inspection
    def journal_entry(self, entry, timestamp=None):
        self.journal.append({'timestamp': timestamp,
                            'entry': entry})


class Case(object):
    def __init__(self, preop_time_params, procedure_time_params,
                 recovery_time_params):
        """Object representation of a procedural case.
        """
        self.preop_time_params = preop_time_params
        self.procedure_time_params = procedure_time_params
        self.recovery_time_params = recovery_time_params
    
    def __str__(self):
        return "<Case object instance>"


def run_simulation(unit, env, logger=None, starting_patients=1,
                   total_patients=1, arrival_interval=0, **kwargs):
    for x in range(starting_patients):
        p = Patient(label='{}'.format(unit.patient_count+1),
                    logger=logger,
                    case=kwargs['standard_case'])
        unit.patients.append(p)
        env.process(unit.process_encounter(p))
    
    while len(unit.patients) < total_patients:
        if arrival_interval == 0:
            yield env.timeout(g_rand(arrival_interval))
        elif isinstance(arrival_interval, dict):
            yield env.timeout(g_rand(**arrival_interval))
        
        #TODO: error handling for neither 0 nor a dict
        
        p = Patient(label='{}'.format(unit.patient_count+1),
                    logger=logger,
                    case=kwargs['standard_case'])
        unit.patients.append(p)
        env.process(unit.process_encounter(p))