"""
Mini decision engine for subjective refraction tests.
Implements three functions required by the evaluation harness:
 - initialize_test(starting_conditions: dict) -> dict
 - next_step(test_state: dict) -> dict
 - final_conclusion(history: list) -> dict

Design notes:
 - Uses conservative, clinically plausible step sizes (0.25D sphere/cylinder).
 - Follows a simplified JCC-like approach for cylinder refinement.
 - Works sequentially on eyes (RE then LE) but can be extended.
 - Detects inconsistent / non-cooperative / pathology-like behaviour and requests referral.

This is a self-contained single-file implementation. No external dependencies.
"""

from copy import deepcopy
import math

# Clinical limits
SPHERE_MIN = -20.0
SPHERE_MAX = 20.0
CYL_MIN = -6.0   # cylinder negative convention (power is negative or 0)
CYL_MAX = 0.0
AXIS_MIN = 0
AXIS_MAX = 180

# Step sizes
SPHERE_STEP = 0.25
CYL_STEP = 0.25
AXIS_STEP = 10

# Maximum steps to avoid infinite loops
DEFAULT_MAX_STEPS = 40


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _normalize_axis(ax):
    # ensure axis is in [0,180)
    if ax is None:
        return 0
    ax = int(round(ax)) % 180
    if ax == 0:
        return 180
    return ax


def _safe_add_sphere(sph, delta):
    new = round(sph + delta, 2)
    return _clamp(new, SPHERE_MIN, SPHERE_MAX)


def _safe_add_cyl(cyl, delta):
    new = round(cyl + delta, 2)
    return _clamp(new, CYL_MIN, CYL_MAX)


class DecisionEngine:
    def __init__(self, starting_conditions: dict):
        # internal state
        self.state = {
            'initialized': False,
            'step': 0,
            'max_steps': starting_conditions.get('max_steps', DEFAULT_MAX_STEPS),
            'target_va': starting_conditions.get('target_va', 0.8),  # default target acuity ~6/7.5
            'eyes_queue': ['RE', 'LE'],
            'current_eye': 'RE',
            'powers': {
                'RE': {'sphere': 0.0, 'cyl': 0.0, 'axis': 180},
                'LE': {'sphere': 0.0, 'cyl': 0.0, 'axis': 180},
            },
            'history': [],
            'last_actions': [],
            'jcc_phase': { 'RE': 'SPHERE_SEARCH', 'LE': 'SPHERE_SEARCH' },
            'stability_count': { 'RE': 0, 'LE': 0 },
            'inconsistent_count': 0,
            'pathology_flag': False,
        }

        # read starting powers if provided
        for eye in ['RE', 'LE']:
            sp = starting_conditions.get(eye, {})
            if isinstance(sp, dict):
                s = sp.get('sphere')
                c = sp.get('cyl')
                a = sp.get('axis')
                if s is not None:
                    self.state['powers'][eye]['sphere'] = _clamp(round(float(s),2), SPHERE_MIN, SPHERE_MAX)
                if c is not None:
                    # expect negative cyl or 0
                    self.state['powers'][eye]['cyl'] = _clamp(round(float(c),2), CYL_MIN, CYL_MAX)
                if a is not None:
                    self.state['powers'][eye]['axis'] = _normalize_axis(int(a))

        self.state['initialized'] = True

    def initial_action(self):
        # Start with right eye monocular (occlude left) and show high-contrast distance chart
        self.state['current_eye'] = 'RE'
        return {
            'action_type': 'SWITCH_OCCLUDER',
            'action_parameters': {'occlude_eye': 'LE'},
            'reasoning': 'Begin with right eye monocular subjective refraction.'
        }

    def interpret_response(self, patient_response):
        # patient_response can be simple strings or dicts
        # Normalize to one of: 'better', 'worse', 'no_change', 'can't_see', 'letter', 'inconsistent', 'exact_acuity'
        if patient_response is None:
            return 'no_response'
        if isinstance(patient_response, dict):
            # if contains acuity
            if 'acuity' in patient_response:
                return {'type': 'acuity', 'value': float(patient_response['acuity'])}
            if 'text' in patient_response:
                patient_response = patient_response['text']
            else:
                # fallback to to-string
                patient_response = str(patient_response)
        if isinstance(patient_response, (int, float)):
            return {'type': 'acuity', 'value': float(patient_response)}
        txt = str(patient_response).strip().lower()
        if txt in ['better', 'yes', 'clear', 'i can see better', 'improved']:
            return 'better'
        if txt in ['worse', 'no', 'less clear', 'blurry']:
            return 'worse'
        if txt in ['no change', 'same', 'equal', 'no difference']:
            return 'no_change'
        if txt in ['cant see', "can't see", 'cannot see', 'nothing', 'no letters', 'i cannot see']:
            return 'cant_see'
        if txt in ['random', 'guessing', 'i am guessing', 'inconsistent']:
            return 'inconsistent'
        # catch single-letter responses as positive (letter seen)
        if len(txt) == 1 and txt.isalpha():
            return 'letter'
        # fallback
        return txt

    def _decide_sphere_step(self, eye, interpreted):
        p = self.state['powers'][eye]
        # If interpreted is acuity value, use it to decide
        if isinstance(interpreted, dict) and interpreted.get('type') == 'acuity':
            acu = interpreted['value']
            # if acuity worse than target, try to add plus for hyperopia or reduce minus for myopia
            # heuristics: if current sphere negative -> myopia present -> make more minus to sharpen distant acuity
            if acu < 0.2 or acu <= 0:  # extremely poor
                # ask to switch chart or check pathology
                self.state['inconsistent_count'] += 1
                return {'action_type': 'ASK_QUESTION', 'action_parameters': {'question': 'Can you tell me if letters look blurry or faint?'}, 'reasoning': 'Very low acuity — check cooperation or pathology.'}
            if acu >= self.state['target_va']:
                return None
        # Basic step: if JCC phase, keep sphere adjustments small
        # Try decreasing sphere by 0.25 if patient says "better" when more minus was tried previously
        # We'll make policy: try -0.25 (more minus) first for negative/0 sphere, else +0.25 for positive sphere (hyperopia)
        if p['sphere'] <= 0:
            delta = -SPHERE_STEP
            reasoning = 'Applying more minus to improve distance clarity.'
        else:
            delta = SPHERE_STEP
            reasoning = 'Applying plus to reduce accommodative demand (hyperopic correction).'
        # update in state
        new_sph = _safe_add_sphere(p['sphere'], delta)
        # if no change possible, stop sphere stage
        if new_sph == p['sphere']:
            return None
        return {
            'action_type': 'CHANGE_POWER',
            'action_parameters': {'eye': eye, 'sphere_change': round(delta,2)},
            'reasoning': reasoning
        }

    def _decide_cylinder_axis(self, eye, interpreted):
        p = self.state['powers'][eye]
        phase = self.state['jcc_phase'][eye]
        # Simple finite-state JCC-like routine
        if phase == 'SPHERE_SEARCH':
            # switch to cylinder refinement when sphere appears stable
            return None
        if phase == 'CYL_DETERMINE':
            # apply cross cylinder: toggle cyl by -0.25 and +0.25
            # choose direction based on previous JCC responses
            history = self.state['history']
            # if no cylinder, first trial set to -0.50 to check astigmatism
            if p['cyl'] == 0:
                new_cyl = _safe_add_cyl(p['cyl'], -0.5)
                return {'action_type': 'CHANGE_POWER', 'action_parameters': {'eye': eye, 'cyl_change': round(-0.5,2)}, 'reasoning': 'Initial cylinder trial to check astigmatism.'}
            # else refine magnitude by 0.25
            # if last response said 'better' when cyl decreased, continue that way
            last = self.state['last_actions'][-1] if self.state['last_actions'] else None
            # default: try smaller magnitude (towards 0)
            try_delta = SPHERE_STEP if p['cyl'] < 0 else -SPHERE_STEP
            new_cyl = _safe_add_cyl(p['cyl'], try_delta)
            if new_cyl == p['cyl']:
                return None
            return {'action_type': 'CHANGE_POWER', 'action_parameters': {'eye': eye, 'cyl_change': round(try_delta,2)}, 'reasoning': 'Refining cylinder magnitude.'}
        if phase == 'AXIS_REFINED':
            # rotate axis by AXIS_STEP and observe
            new_axis = ((_normalize_axis(p['axis'] + AXIS_STEP)) if p['axis'] else AXIS_STEP)
            return {'action_type': 'CHANGE_POWER', 'action_parameters': {'eye': eye, 'axis_change': AXIS_STEP}, 'reasoning': 'Refining astigmatic axis.'}
        return None

    def step(self, current_machine_state: dict, patient_response, persona_id: str):
        self.state['step'] += 1
        interpreted = self.interpret_response(patient_response)

        # store history entry
        hist_entry = {
            'step': self.state['step'],
            'eye': self.state['current_eye'],
            'machine_state': deepcopy(current_machine_state),
            'patient_response_raw': patient_response,
            'patient_response_interpreted': interpreted,
            'powers_before': deepcopy(self.state['powers'][self.state['current_eye']])
        }
        self.state['history'].append(hist_entry)

        eye = self.state['current_eye']

        # detect pathology-like patterns: repeated 'cant_see' or 'inconsistent' responses
        if interpreted == 'cant_see' or interpreted == 'inconsistent' or interpreted == 'no_response':
            self.state['inconsistent_count'] += 1
        else:
            # reset slight inconsistent flag if user gives reasonable answer
            self.state['inconsistent_count'] = max(0, self.state['inconsistent_count'] - 0)

        if self.state['inconsistent_count'] >= 5:
            self.state['pathology_flag'] = True
            return {
                'action_type': 'END_TEST',
                'action_parameters': {},
                'reasoning': 'Multiple inconsistent or non-seeing responses — suspect pathology and stop.'
            }

        # Limit steps
        if self.state['step'] > self.state['max_steps']:
            # give up
            return {
                'action_type': 'END_TEST',
                'action_parameters': {},
                'reasoning': 'Maximum steps exceeded; finalizing current best correction.'
            }

        # Determine which phase we're in for current eye
        phase = self.state['jcc_phase'][eye]

        # If patient gave acuity value, and it meets target -> mark stability
        if isinstance(interpreted, dict) and interpreted.get('type') == 'acuity':
            acu = interpreted['value']
            if acu >= self.state['target_va']:
                self.state['stability_count'][eye] += 1
            else:
                self.state['stability_count'][eye] = 0

        # Sphere search phase
        if phase == 'SPHERE_SEARCH':
            # If last two steps didn't change sphere, move to cylinder determination
            last_powers = self.state['powers'][eye]
            # heuristic: if stability_count high or no recent sphere-change actions then move on
            recent_actions = [a for a in self.state['last_actions'][-4:] if a.get('action_type') == 'CHANGE_POWER' and a['action_parameters'].get('eye') == eye and 'sphere_change' in a['action_parameters']]
            if self.state['stability_count'][eye] >= 2 or len(recent_actions) >= 2 and all(abs(a['action_parameters']['sphere_change']) < 0.001 for a in recent_actions):
                self.state['jcc_phase'][eye] = 'CYL_DETERMINE'
                return {
                    'action_type': 'SWITCH_CHART',
                    'action_parameters': {'chart': 'CROSS_CYLINDER_CHART'},
                    'reasoning': 'Sphere appears stable — proceed to cylinder (JCC-style) refinement.'
                }

            # Otherwise decide a sphere step
            decision = self._decide_sphere_step(eye, interpreted)
            if decision is None:
                # no sphere decision -> move to cylinder
                self.state['jcc_phase'][eye] = 'CYL_DETERMINE'
                return {
                    'action_type': 'SWITCH_CHART',
                    'action_parameters': {'chart': 'CROSS_CYLINDER_CHART'},
                    'reasoning': 'Sphere adjustment not needed — proceed to cylinder refinement.'
                }

            # apply change to internal state (simulate)
            params = decision['action_parameters']
            if 'sphere_change' in params:
                delta = params['sphere_change']
                old = self.state['powers'][eye]['sphere']
                self.state['powers'][eye]['sphere'] = _safe_add_sphere(old, delta)

            # record last action
            self.state['last_actions'].append(decision)
            return decision

        # Cylinder/axis phases
        if phase in ['CYL_DETERMINE', 'AXIS_REFINED']:
            decision = self._decide_cylinder_axis(eye, interpreted)
            if decision is None:
                # finish this eye
                # mark completed and switch occluder to other eye (or end if both done)
                finished_eye = eye
                # rotate queue
                if self.state['eyes_queue'] and self.state['eyes_queue'][0] == eye:
                    self.state['eyes_queue'].pop(0)
                next_eye = self.state['eyes_queue'][0] if self.state['eyes_queue'] else None
                if next_eye:
                    self.state['current_eye'] = next_eye
                    # reset some counters for next eye
                    return {
                        'action_type': 'SWITCH_OCCLUDER',
                        'action_parameters': {'occlude_eye': 'RE' if next_eye == 'LE' else 'LE'},
                        'reasoning': f'Finished {finished_eye}. Proceeding to {next_eye}.'
                    }
                else:
                    # both eyes done
                    return {
                        'action_type': 'END_TEST',
                        'action_parameters': {},
                        'reasoning': 'Both eyes refined — end test.'
                    }

            # apply change to internal state: sphere/cyl/axis changes
            params = decision['action_parameters']
            if 'cyl_change' in params:
                delta = params['cyl_change']
                old = self.state['powers'][eye]['cyl']
                self.state['powers'][eye]['cyl'] = _safe_add_cyl(old, delta)
                # after applying cylinder, move to axis refinement
                self.state['jcc_phase'][eye] = 'AXIS_REFINED'
            if 'axis_change' in params:
                delta = params['axis_change']
                old = self.state['powers'][eye]['axis']
                new_axis = _normalize_axis(old + delta)
                self.state['powers'][eye]['axis'] = new_axis
                # if axis cycles all the way without improvement, conclude stability
                # simple heuristic: after a certain number of axis adjustments, stop
                # (we track via stability_count)
                self.state['stability_count'][eye] += 1

            self.state['last_actions'].append(decision)
            return decision

        # fallback: when nothing matches, ask a clarifying question
        return {
            'action_type': 'ASK_QUESTION',
            'action_parameters': {'question': 'Could you please tell me whether the letters are clearer with the current lenses or the previous ones?'},
            'reasoning': 'Clarification required to decide next step.'
        }


# Public functions required by harness

def initialize_test(starting_conditions: dict) -> dict:
    """
    Initialize internal state and return an initial action / context.

    starting_conditions may contain initial autorefractor values for 'RE' and 'LE' as dicts with keys 'sphere','cyl','axis'.
    """
    engine = DecisionEngine(starting_conditions)
    # Save engine into returned context so harness can pass it back in next_step as opaque test_state
    test_state = {
        'engine': engine,
        'context': {
            'current_machine_state': starting_conditions.get('machine_state', {}),
            'persona_id': starting_conditions.get('persona_id')
        }
    }
    first = engine.initial_action()
    # store to history
    engine.state['history'].append({'event':'initialize','action': first})
    return test_state | {'initial_action': first}


def next_step(test_state: dict) -> dict:
    """
    Accepts a dict with keys:
      - current_machine_state
      - patient_response
      - history
      - persona_id
    and returns a single action dict.

    The harness must pass back the object returned by initialize_test (which includes an 'engine').
    """
    # Expect engine instance in test_state
    engine = test_state.get('engine')
    if engine is None:
        # try to recover: create new engine with defaults
        engine = DecisionEngine({})
        test_state = {'engine': engine}

    current_machine_state = test_state.get('current_machine_state') or {}
    patient_response = test_state.get('patient_response') if 'patient_response' in test_state else None
    persona_id = test_state.get('persona_id', '')

    # The harness spec originally expects an input structure; to be robust, accept that
    # if a nested dict is provided, extract fields
    if isinstance(current_machine_state, dict) and 'current_machine_state' in current_machine_state:
        # flatten
        cm = current_machine_state['current_machine_state']
        patient_response = current_machine_state.get('patient_response', patient_response)
        persona_id = current_machine_state.get('persona_id', persona_id)
        current_machine_state = cm

    # call engine.step and return an action
    action = engine.step(current_machine_state, patient_response, persona_id)

    # update the provided test_state history so the harness can inspect it
    test_state['last_action'] = action
    test_state['engine'] = engine
    return action


def final_conclusion(history: list) -> dict:
    """
    Inspect engine history (expected to contain engine.state) and return either REFRACTION_ACHIEVED or REFRACTION_NOT_POSSIBLE.
    history is expected to be a list; if it's a dict containing an engine, we will read it.
    """
    # try to locate engine/state in history param
    engine = None
    if isinstance(history, dict) and 'engine' in history:
        engine = history['engine']
    elif isinstance(history, dict) and 'state' in history:
        engine = history
    else:
        # if list, try to find the last dict with 'engine'
        if isinstance(history, list):
            for item in reversed(history):
                if isinstance(item, dict) and 'engine' in item:
                    engine = item['engine']
                    break

    if engine is None:
        # fallback: cannot determine — report not possible
        return {
            'status': 'REFRACTION_NOT_POSSIBLE',
            'referral': 'OPHTHALMOLOGIST_REQUIRED',
            'reasoning': 'No engine state found in provided history.'
        }

    state = engine.state
    if state.get('pathology_flag'):
        return {
            'status': 'REFRACTION_NOT_POSSIBLE',
            'referral': 'OPHTHALMOLOGIST_REQUIRED'
        }

    # build final powers
    final_power = {}
    for eye in ['RE', 'LE']:
        p = state['powers'].get(eye, {'sphere': 0.0, 'cyl': 0.0, 'axis': 180})
        # safety clamp
        sph = _clamp(round(float(p.get('sphere',0.0)),2), SPHERE_MIN, SPHERE_MAX)
        cyl = _clamp(round(float(p.get('cyl',0.0)),2), CYL_MIN, CYL_MAX)
        ax = _normalize_axis(int(p.get('axis',180) if p.get('axis') is not None else 180))
        final_power[eye] = {'sphere': sph, 'cyl': cyl, 'axis': ax}

    return {
        'status': 'REFRACTION_ACHIEVED',
        'final_power': final_power
    }


# If this file executed directly, demonstrate a tiny self-check (won't run in harness)
if __name__ == '__main__':
    sc = {'RE': {'sphere': -1.25, 'cyl': -0.5, 'axis': 90}, 'LE': {'sphere': -1.0, 'cyl': 0.0, 'axis': 180}, 'max_steps': 10}
    ctx = initialize_test(sc)
    print('Initial action:', ctx['initial_action'])
    engine = ctx['engine']
    # simulate a few steps
    action = next_step({'engine': engine, 'current_machine_state': {}, 'patient_response': {'acuity': 0.6}})
    print('Step1:', action)
    action = next_step({'engine': engine, 'current_machine_state': {}, 'patient_response': 'better'})
    print('Step2:', action)
    print('Final:', final_conclusion({'engine': engine}))

