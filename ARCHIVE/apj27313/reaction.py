"""
Module for reaction based on Ion class
"""

import re
from numpy import sign

import numpy as np
from copy import copy

from starfit.autils.isotope import Ion, ion, register_other_bits, GAMMA, Isotope, Isomer, lepton, Lepton, Photon
from starfit.autils.utils import CachedAttribute, cachedmethod

class DecIon(Ion):
    """
    Class for decays.

    Will store A, Z, E in ion class portion
    Will set flag F_REACTION
    Will store extra info in "B" field, for now just the 12 recognized decay modes

    Provides list of ions, a dictionary with numbers -
        number of ions/particles that come out.

    The general reaction class should have these be negative for
    things that go in and positive for things that come out whereas the
    A, Z, E fields will store IN - OUT net values.

    LIMITATION:
        Currently this class only handles a pre-defined set of decays.

    PLAN:
        Extension for sf - complicated decays - maybe a separate
        derived class?

    TODO:
        For general class provide a generalized *index*.
        probably multiply the FZAE indices
        (plus BMUL/2 to take care of negative values in Z)
        with according numbers of BMUL?
        Add in flag for IN/OUT?
        It does not have to be nice, just unique...
        General reactions need have dE = 0 but still connect and 2 (Z,A,E) states.
    """

    # define KepIon bits
    F_OTHER_REACTION = 1
    F_REACTION = Ion.F_OTHER * F_OTHER_REACTION

    # list of decay names accepted
    reaction_list = {
        ''     :  0,
        's'    :  0,
        'g'    :  1,
        'g1'   :  1,
        '1g'   :  1,
        'b-'   :  2,
        'bm'   :  2,
        'b+'   :  3,
        'bp'   :  3,
        'ec'   :  4,
        'n'    :  5,
        'p'    :  6,
        'a'    :  7,
        '2a'   :  8,
        'a2'   :  8,
        '2p'   :  9,
        'p2'   :  9,
        'n2'   : 10,
        '2n'   : 10,
        'np'   : 11,
        'pn'   : 11,
        'bn'   : 12,
        'b2n'  : 13,
        'g2'   : 14,
        '2g'   : 14,
        'g3'   : 15,
        '3g'   : 15,
        'g4'   : 16,
        '4g'   : 16,
        'b2-'  : 17,
        '2b-'  : 17,
        'bb-'  : 17,
        'b2+'  : 18,
        '2b+'  : 18,
        'bb+'  : 18,
        }

    # OK, we want names to be unique on output
    reaction_names = {
          0 : 's'  ,
          1 : 'g'  ,
          2 : 'b-' ,
          3 : 'b+' ,
          4 : 'ec' ,
          5 : 'n'  ,
          6 : 'p'  ,
          7 : 'a'  ,
          8 : '2a' ,
          9 : '2p' ,
         10 : '2n' ,
         11 : 'pn' ,
         12 : 'bn' ,
         13 : 'b2n',
         14 : 'g2' ,
         15 : 'g3' ,
         16 : 'g4' ,
         17 : 'bb-',
         18 : 'bb+',
        }

    # list of particles that come out...
    particle_list = {
         0 : {},
         1 : {ion('g')   : +1},
         2 : {ion('e-')  :  1},
         3 : {ion('e+')  :  1},
         4 : {ion('e-')  : -1},
         5 : {ion('nt1') :  1},
         6 : {ion('h1')  :  1},
         7 : {ion('he4') :  1},
         8 : {ion('he4') :  2},
         9 : {ion('h1')  :  2},
        10 : {ion('nt1') :  2},
        11 : {ion('nt1') :  1, ion('h1')  : 1},
        12 : {ion('e-')  :  1, ion('nt1') : 1},
        13 : {ion('e-')  :  1, ion('nt1') : 2},
        14 : {ion('g')   : +2},
        15 : {ion('g')   : +3},
        16 : {ion('g')   : +4},
        17 : {ion('e-')  :  2},
        18 : {ion('e+')  :  2},
        }

    @property
    def is_photon(self):
        return self.E != 0

    _custom_add = True

    def _add(self, x, sign1 = +1, sign2 = +1):
        """
        Add reaction to Ion of two reactions

        if Ion is an isomer, return isomer, otherwise return isotope
        """
        if isinstance(x, self.__class__):
            new = self.__class__('')
            new.B = -1
            new._particles = {}
            for p,m in self._particles.items():
                new._particles[p] = sign1 * m
            for p,n in x._particles.items():
                n *= sign2
                m = new._particles.get(p, 0)
                if m == 0:
                    new._particles[p] = n
                else:
                    m += n
                    if m == 0:
                        del new._particles[p]
                    else:
                        new._particles[p] = m
            for b,p in new.particle_list.items():
                if p == new._particles:
                    new.B = b
            new.Z, new.A, new.E = new.particles2zae(new._particles)
            return new

        if not isinstance(x, Ion):
            x = ion(x)
        A = sign1 * self.A + sign2 * x.A
        Z = sign1 * self.Z + sign2 * x.Z
        if x.is_isomer:
            E = max(sign2 * x.E + sign1 * self.E, 0)
        else:
            E = None
        return ion(Z = Z, A = A, E = E)

    def __add__(self, x):
        return self._add(x)
    __radd__ = __add__

    def __call__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        return self._add(x, +1, -1)

    def __rsub__(self, x):
        return self._add(x, -1, +1)

    def __mul__(self, x):
        assert np.mod(x, 1) == 0, " should only mutiply integers"

        new = self.__class__('')
        new.B = -1
        new._particles = {}
        for p,m in self._particles.items():
            new._particles[p] = m * x
        for b,p in new.particle_list.items():
            if p == new._particles:
                new.B = b
        new.Z, new.A, new.E = new.particles2zae(new._particles)
        return new

    __rmul__ = __mul__

    def __init__(self, s, parent=None):
        """
        Set up decay reaction.

        Currently only a string is allowed for initialization.
        The main purpose is the interface to the decay.dat file
        and the Decay class setup in ionmap.py.

        TO - use parent to set up fission and cluster decays
        """
        if isinstance(s, str):
            self.B, self.F, self.Z, self.A, self.E = self.ion2bfzae(s)
        elif isinstance(s, type(self)):
            self.B, self.F, self.Z, self.A, self.E = s.tuple()
        else:
            raise AttributeError('Wrong type')
        assert 0 <= self.B < len(self.reaction_names), "Unknown decay/reaction."
        self._particles = self.particle_list[self.B]
        self._update_idx()

    @classmethod
    def from_reactants(cls, input, output, hint = None):
        """
        e.g., call with 'c14','n14m1'
        """
        raise NotImplementedError()



    # # for deepcopy:
    # def __getstate__(self):
    #     return self.B
    # def __setstate__(self, x):
    #     self.__init__(self.reaction_names[x])

    @classmethod
    def ion2bfzae(cls, s=''):
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            s=s[1:-1].strip()
        b =  cls.reaction_list.get(s.lower(), -1)
        assert b >= 0, "decay not found"
        z, a, e = cls.particles2zae(cls.particle_list.get(b,{}))
        return b, cls.F_REACTION, z, a, e

    @staticmethod
    def particles2zae(particles):
        z, a, e = 0, 0, 0
        for i,k in particles.items():
            a -= i.A * k
            z -= i.Z * k
            e -= i.E * k
        return z, a, e

    def _name(self, upcase = None):
        """
        Return pre-defined name from list.
        """
        if self.B >= 0:
            return self.reaction_names.get(self.B, self.VOID_STRING)
        else:
            i = []
            o = []
            for p,n in self._particles.items():
                if n < 0:
                    i += [p._name(upcase)] * (-n)
                else:
                    o += [p._name(upcase)] * (+n)
            if len(i) > 0 and len(o) == 0:
                o += [GAMMA._name(upcase)]
            if len(i) > 0:
                s = ' '.join(i) + ', '
            else:
                s = ''
            s += ' '.join(o)
            return s


    def particles(self):
        """
        Returns all paricles in the reaction/decay.
        """
        return copy(self._particles)

    def hadrons(self):
        """
        Returns just the hardrons in the reaction/decay.

        This is useful for networks like decay where photons and
        leptons usually are not accounted for.
        """
        h = dict()
        for i,j in self._particles.items():
            if i.is_hadron:
                h[i] = j
        return h

    def nuclei(self):
        """
        Returns just the nuclei in the reaction/decay.

        This is useful for networks like decay where photons and
        leptons usually are not accounted for.
        """
        h = dict()
        for i,j in self._particles.items():
            if i.is_nucleus:
                h[i] = j
        return h

    def leptons(self):
        """
        Returns just the leptons in the reaction/decay.
        """
        h = dict()
        for i,j in self._particles.items():
            if i.is_lepton:
                h[i] = j
        return h

    def photons(self):
        """
        Returns just the photons (if any) in the reaction/decay.
        """
        h = dict()
        for i,j in self._particles.items():
            if i.is_photon:
                h[i] = j
        return h

    def isstable(self):
        """
        Return if 'reaction'/'decay' is 'stable'.
        """
        return self.B == 0

register_other_bits(DecIon.F_OTHER_REACTION, DecIon)

# Q: not sure this will become sub- or super-class of DecIon
# A: should become a superclass
class ReactIon(DecIon):
    """
    Class for pre-defined set of reactions.

    I suppose we don't need separate 'In' and 'Out' list - just
    positive and negative values in the dictionary of particles.
    We could support 'In' and 'Out', though.
    We should add pre-defined strings for the common set of p,n,a,g,b+/b-.

    Will store A, Z, E in ion class portion
    Will set flag F_REACTION
    Will store extra info in "B" field, for now just the recognized reactions

    Provides list of ions, a dictionary with numbers -
        number of ions/particles that come out or come in.

    The general reaction class should have these be negative for
    things that go in and positive for things that come out whereas the
    A, Z, E fields will store IN - OUT net values.

    LIMITATION:
        Currently this class only handles a pre-defined set of reactions

    PLAN:
        Extension for sf - complicated reactions - maybe a separate
        derived class?

    TODO:
        For general class provide a generalized *index*.
        probbaly multiply the FZAE indices
        (plus BMUL/2 to take care of negative values in Z)
        with according numbers of BMUL?
        Add in flag for IN/OUT?
        It does not have to be nice, just unique...
    TODO:
        'g,...' and '...,g' should be used only when level
            change is intended
    """

    # list of reactions names accepted
    reaction_list = {
        ''     :  0,
        'b-'   :  1,
        'ec'   :  2,
        'b+'   :  3,
        'pc'   :  4,
        'g,n'  :  5,
        'gn'   :  5,
        'n,g'  :  6,
        'ng'   :  6,
        'g,p'  :  7,
        'gp'   :  7,
        'p,g'  :  8,
        'pg'   :  8,
        'g,a'  :  9,
        'ga'   :  9,
        'a,g'  : 10,
        'ag'   : 10,
        'n,p'  : 11,
        'np'   : 11,
        'p,n'  : 12,
        'pn'   : 12,
        'n,a'  : 13,
        'na'   : 13,
        'a,n'  : 14,
        'an'   : 14,
        'p,a'  : 15,
        'pa'   : 15,
        'a,p'  : 16,
        'ap'   : 16,
        'g'    : 17,
        'g*'   : 18,
        'ad'   : 19,
        'ac'   : 20,
        }

    # OK, we want names to be unique on output
    reaction_names = {
          0 : ''   ,
          1 : 'b-' ,
          3 : 'ec' ,
          2 : 'b+' ,
          4 : 'pc' ,
          5 : 'g,n',
          6 : 'n,g',
          7 : 'g,p',
          8 : 'p,g',
          9 : 'g,a',
         10 : 'a,g',
         11 : 'n,p',
         12 : 'p,n',
         13 : 'n,a',
         14 : 'a,n',
         15 : 'p,a',
         16 : 'a,p',
         17 : 'g'  ,
         18 : 'g*' ,
         19 : 'ad',
         20 : 'ac',
        }

    reaction_names_latex = {
          0 : r''   ,
          1 : r'\beta^-' ,
          3 : r'\mathsf{EC}' ,
          2 : r'\beta^+' ,
          4 : r'\mathsf{PC}' ,
          5 : r'\gamma,\mathsf{n}',
          6 : r'\mathsf{n},\gamma',
          7 : r'\gamma,\mathsf{p}',
          8 : r'\mathsf{p},\gamma',
          9 : r'\gamma,\alpha',
         10 : r'\alpha,\gamma',
         11 : r'\mathsf{n},\mathsf{p}',
         12 : r'\mathsf{p},\mathsf{n}',
         13 : r'\mathsf{n},\alpha',
         14 : r'\alpha,\mathsf{n}',
         15 : r'\mathsf{p},\alpha',
         16 : r'\alpha,\mathsf{p}',
         17 : r'\gamma'  ,
         18 : r'\gamma^*' ,
         19 : r'\mathsf{AD}',
         20 : r'\mathsf{AC}',
        }

    # list of particles that come out...
    particle_list = {
         0 : {},
         1 : {                 ion('e-')  : 1},
         2 : {                 ion('e+')  : 1},
         3 : {ion('e-')  : -1                },
         4 : {ion('e+')  : -1,               },
         5 : {                 ion('nt1') : 1},
         6 : {ion('nt1') : -1                },
         7 : {                 ion('h1')  : 1},
         8 : {ion('h1')  : -1                },
         9 : {                 ion('he4') : 1},
        10 : {ion('he4') : -1                },
        11 : {ion('nt1') : -1, ion('h1')  : 1},
        12 : {ion('h1')  : -1, ion('nt1') : 1},
        13 : {ion('nt1') : -1, ion('he4') : 1},
        14 : {ion('he4') : -1, ion('nt1') : 1},
        15 : {ion('h1')  : -1, ion('he4') : 1},
        16 : {ion('he4') : -1, ion('h1')  : 1},
        17 : {ion('g')   : -1                },
        18 : {                 ion('g')   : 1},
        19 : {                 ion('he4') : 1},
        20 : {ion('he4') : -1                },
        }

    def _name(self, upcase = None):
        """
        Return pre-defined name from list.
        """
        return "({})".format(self.reaction_names.get(self.B, self.VOID_STRING))

    @cachedmethod
    def mpl(self):
        s = self.reaction_names_latex.get(self.B, self.VOID_STRING)
        s = s.replace(',', '$,$')
        s = f'$({s})$'
        return s

    def __mul__(self, x):
        raise NotImplementedError("Can't multiply tabulated reactions yet.")

    def __neg__(self):
        return self.rev()

    def rev(self):
        return self.__class__(self.reaction_names[self.B - 1 + 2 * (self.B % 2)])

    # later we could return a general reaction here

# TODO - add split function

# define more general class

from collections import Counter, OrderedDict

def dissect(s):
    """
    cases
     - \w+\d match ion
     - (\d?) \w+ match (special) ion
     - \d \w+ match ion
    """

    k = re.findall(r'\d+|[-+*a-zA-Z]', s)
    if ''.join(k) != s:
        return None
    r = recmatch(k)
    return r

def recmatch(k):
    if len(k) == 0:
        return None
    # try single match
    s = ''.join(k)
    x = ion(s)
    # print('rec:', s)
    if isinstance(x, (Lepton, Photon, Isotope)):
        return [s]
    if isinstance(x, Isomer) and not re.fullmatch('\d+', k[0]):
        return [s]

    # try separate out leading multplier
    if (len(k) > 0) and re.fullmatch('\d+', k[0]):
        if len(k) == 1:
            return None
        x = recmatch(k[1:])
        if x is not None and len(x) == 1:
            return x * int(k[0])
    # for i in range(1, len(k)):
    for i in range(len(k)-1,0,-1):
        y = recmatch(k[-i:])
        if y is None:
            continue
        x = recmatch(k[:-i])
        if x is None:
            continue
        print(x+y)
        return x + y
    return None

nuc_split = re.compile(r'\s+|(?<=[^tmel])\+|(?<=\+)\+|(?<=nu[emtx])\+')

def disassemble(nuc):

    #split by space and +
    nuc =  nuc_split.split(nuc)
    nuc = [s.strip() for s in nuc if len(s.strip()) > 0]

    # divide in run-ups such as 'aap'
    new = list()

    for x in nuc:
        y = dissect(x)
        if y is not None:
            new.extend(y)
            continue
        new.append(x)

    nuc = new
    new = list()
    i = 1
    n = 0
    for x in nuc:
        x = x.strip()
        if x in ('+', ''):
            continue

        if re.fullmatch('\d+', x):
            i = int(x)
            n = 1
        else:
            new.extend([x] * i)
            i = 1
            n = 0

    assert n == 0, f'invalid {x}'

    nuc = list()
    lep = list()
    gam = list()

    # sorting things out
    for n in new:
        i = ion(n)
        if i.is_lepton:
            lep.append(i)
        elif i.is_photon:
            gam.append(i)
        else:
            nuc.append(i)

    return nuc, lep, gam


class Reaction(object):
    """General reaction class, including all that goes in and out.

    Currently only tracks nucleons, intensinally, as this is what we
    need to do nuclear reactions.

    TODO - add output format that uses () notation

    TODO - allow two-way auto-complete
    """
    def __init__(self, nuc_in, nuc_out=None,/, flags=None, reversible=None,
                 check_Z=True, check_l=True, check_A=True,
                 ):
        """
        nuc_in, nuc_out - ion, list of ions, [list of] tuple[s] of (#, ion)
        """

        repl = (
            ('(ac)', '(a,g)'), ('(AC)', '(a,g)'),
            # ('(ad)', '(;a)'),
            ('(AD)', '(;a)'),
            ('(nu)', '(nu,nu)'), ('(l)', '(l,l)'),
            ('(nux)', '(nux,nux)'), ('(nbx)', '(nbx,nbx)'),
            ('(nue)', '(nue,nue)'), ('(nbe)', '(nbe,nbe)'),
            ('(num)', '(num,num)'), ('(nbm)', '(nbm,nbm)'),
            ('(nut)', '(nut,nut)'), ('(nbt)', '(nbt,nbt)'),
            )

        # here we use e- for in channel, b- for our channel

        change = {
            '(0nuee)':{'e-':-2}, # common name
            '(0nu2e)':{'e-':-2}, '(0nu2e-)':{'e-':-2},
            '(0nue-e-)':{'e-':-2},

            '(ECEC)':{'e-':-2,'nue':+2}, '(ecec)':{'e-':-2,'nue':+2},
            '(2e-)':{'e-':-2,'nue':+2}, '(ee)':{'e-':-2,'nue':+2},
            '(e-e-)':{'e-':-2,'nue':+2},

            '(ECb+)':{'e-':-1,'e+':+1,'nue':+2},
            '(ecb+)':{'e-':-1,'e+':+1,'nue':+2},

            '(EC)':{'e-':-1,'nue':+1}, '(ec)':{'e-':-1,'nue':+1},
            '(e-)':{'e-':-1,'nue':+1}, '(e)':{'e-':-1,'nue':+1},

            '(0nu2b+)':{'e+':+2},'(0nub+b+)':{'e+':+2},
            '(0nu2B+)':{'e+':+2},'(0nuB+B+)':{'e+':+2},

            '(2b+)':{'e+':+2,'nue':+2}, '(2B+)':{'e+':+2,'nue':+2},
            '(b+b+)':{'e+':+2,'nue':+2}, '(B+B+)':{'e+':+2,'nue':+2},
            'bb+':{'e+':+2,'nue':+2}, 'BB+':{'e+':+2,'nue':+2},
            'b+b+':{'e+':+2,'nue':+2}, 'B+B+':{'e+':+2,'nue':+2},
            'b2+':{'e+':+2,'nue':+2}, 'B2+':{'e+':+2,'nue':+2},

            '(PD)':{'e+':+1,'nue':+1}, '(pd)':{'e+':+1,'nue':+1},
            '(b+)':{'e+':+1,'nue':+1}, '(B+)':{'e+':+1,'nue':+1},
            'b+':{'e+':+1,'nue':+1}, 'B+':{'e+':+1,'nue':+1},

            '(PC)':{'e+':-1,'nbe':+1}, '(pc)':{'e+':-1,'nbe':+1},
            '(e+)':{'e+':-1,'nbe':+1},

            '(0nub-b-)':{'e-':+2}, '(0nu2b-)':{'e-':+2},
            '(0nubb)':{'e-':+2}, # common name

            '(2b-)':{'e-':+2,'nbe':+2}, '(2B-)':{'e-':-2,'nbe':+2},
            '(b-b-)':{'e-':+2,'nbe':+2}, '(B-B-)':{'e-':-2,'nbe':+2},
            '(bb)':{'e-':+2,'nbe':+2}, '(BB)':{'e-':-2,'nbe':+2},
            'bb-':{'e-':+2,'nbe':+2}, 'BB-':{'e-':-2,'nbe':+2},
            'b-b-':{'e-':+2,'nbe':+2}, 'B-B-':{'e-':-2,'nbe':+2},
            'b2-':{'e-':+2,'nbe':+2}, 'B2-':{'e-':-2,'nbe':+2},

            '(ED)':{'e-':+1,'nbe':+1}, '(ed)':{'e-':+1,'nbe':+1},
            '(BD)':{'e-':+1,'nbe':+1}, '(bd)':{'e-':+1,'nbe':+1},
            '(b-)':{'e-':+1,'nbe':+1}, '(B-)':{'e-':+1,'nbe':+1},
            '(b)':{'e-':+1,'nbe':+1}, '(B)':{'e-':+1,'nbe':+1},
            'b-':{'e-':+1,'nbe':+1}, 'B-':{'e-':+1,'nbe':+1},
            }

        leptons = {'e-':'e-', 'e+':'e+', }

        autocomplete = 0

        lep_in = Counter()
        lep_out = Counter()
        gam_in = Counter()
        gam_out = Counter()

        lepton_violation = np.zeros(4, dtype=np.int64)

        if isinstance(nuc_in, str) and nuc_out is None:
            if (nuc_in.count('(') != nuc_in.count(')') or
                nuc_in.count('(') not in (0, 1)):
                raise Exception(f'bracket error "{nuc_in}')
            if nuc_in.count(',') + nuc_in.count(';') > 1:
                raise Exception(f'comma error "{nuc_in}')

            if nuc_in.strip().endswith(')'):
                autocomplete = 1
            if nuc_in.strip().startswith('('):
                autocomplete = -1

            for x,y in repl:
                if nuc_in.count(x) == 1:
                    nuc_in = nuc_in.replace(x, y)

            x = re.findall('\(([-+0-9a-zA-Z]+)\)', nuc_in)
            if len(x) == 1:
                x = x[0]
                y = dissect(x)
                if y is not None:
                    c = Counter(y)
                    assert list(c.keys())[0] == y[0]
                    if len(c) == 2:
                        s = ','.join(f'{b:d} {a:s}' for a,b in c.items())
                        nuc_in = nuc_in.replace(f'({x})', f' {s} ')

            for c,dec in change.items():
                if nuc_in.count(c) == 1:
                    for l,n in dec.items():
                        l = Lepton(l)
                        if n < 0:
                            lep_in.update({l:-n})
                        else:
                            lep_out.update({l:n})
                        lepton_violation[l.flavor-1] += n * l.number
                    nuc_in = nuc_in.replace(c, ',')
                    if reversible is None:
                        reversible = False
                    break

            nuc_in = nuc_in.replace('(', ' ')
            nuc_in = nuc_in.replace(')', ' ')

            split = False
            for s in (',', '<-->', '<->', '-->', '->', '<==>', '<=>', '<>'):
                if nuc_in.count(s) == 1:
                    nuc_in, nuc_out = nuc_in.split(s)
                    if reversible is None:
                        reversible = True
                    split = True
                    break
            if not split:
                for s in (';', '==>', '=>', ):
                    if nuc_in.count(s) == 1:
                        nuc_in, nuc_out = nuc_in.split(s)
                        if reversible is None:
                            reversible = False
                        split = True
                        break
            if not split:
                # need separate to avoid confusion with case above
                for s in ('>', ):
                    if nuc_in.count(s) == 1:
                        nuc_in, nuc_out = nuc_in.split(s)
                        if reversible is None:
                            reversible = True
                        split = True
                        break
            if not split:
                for s in ('<==', '<=',):
                    if nuc_in.count(s) == 1:
                        nuc_out, nuc_in = nuc_in.split(s)
                        if reversible is None:
                            reversible = False
                        split = True
                        break
            if not split:
                for s in ('<--', '<-', '<'):
                    if nuc_in.count(s) == 1:
                        nuc_out, nuc_in = nuc_in.split(s)
                        if reversible is None:
                            reversible = True
                        split = True
                        break
            if not split:
                for s in ('...', '..', '.', ):
                    if nuc_in.count(s) == 1:
                        nuc_in_, nuc_out_ = nuc_in.split(s)
                        if len(nuc_in_) > 0 and len(nuc_out_) > 0:
                            nuc_in = nuc_in_
                            nuc_out = nuc_out_
                            if reversible is None:
                                reversible = True
                            split = True
                            autocomplete = True
                            break

            if split:
                if len(nuc_in.strip()) == 0:
                    autocomplete = -1
                if len(nuc_out.strip()) == 0:
                    autocomplete = 1
            else:
                nuc_out = ''
                y = dissect(nuc_in)
                if autocomplete == 0:
                    if y is not None:
                        if len(y) >= 3:
                            c = Counter(y)
                            assert list(c.keys())[0] == y[0]
                            if len(c) == 3:
                                raw = [f'{b:d} {a:s}' for a,b in c.items()]
                                nuc_in = ' '.join(raw[:2])
                                nuc_out = raw[2]
                autocomplete = 1

            nuc_in = nuc_in.strip()
            nuc_out = nuc_out.strip()

            complete = ('...', '..', '.', )
            for c in complete:
                if nuc_in.count(c) == 1:
                    nuc_in = nuc_in.replace(c, ' ').strip()
                    autocomplete = -1
                    break
            else:
                for c in complete:
                    if nuc_out.count(c) == 1:
                        nuc_out = nuc_out.replace(c, ' ').strip()
                        autocomplete = 1
                        break

            for c,l in leptons.items():
                while nuc_in.count(c) > 0:
                    lep_in.update({ion(l):1})
                    nuc_in = nuc_in.replace(c, ' ', 1)
                while nuc_out.count(c) > 0:
                    lep_out.update({ion(l):1})
                    nuc_out = nuc_out.replace(c, ' ')

        # print(nuc_in, nuc_out)

        if isinstance(nuc_in, str):
            nuc_in, lep, gam = disassemble(nuc_in)
            lep_in.update(lep)
            gam_in.update(gam)

        if isinstance(nuc_out, str):
            nuc_out, lep, gam = disassemble(nuc_out)
            lep_out.update(lep)
            gam_out.update(gam)

        if isinstance(nuc_in, (Ion, str)):
            nuc_in = (1, nuc_in)
        if isinstance(nuc_in, tuple):
            nuc_in = [nuc_in]
        if isinstance(nuc_in, (list, np.ndarray)):
            x_in = Counter()
            for x in nuc_in:
                if isinstance(x, tuple):
                    i,n = x
                else:
                    i,n = 1,x
                if isinstance(n, str):
                    n = ion(n)
                assert isinstance(i, int)
                assert isinstance(n, Ion)
                x_in.update({n:i})
            nuc_in = x_in
        if isinstance(nuc_in, dict):
            nuc_in = Counter(nuc_in)
        assert isinstance(nuc_in, Counter)
        for n,i in nuc_in.items():
            if isinstance(n, str):
                del nuc_in[n]
                nuc_in.update({ion(n):i})

        if isinstance(nuc_out, (Ion, str)):
            nuc_out = (1, nuc_out)
        if isinstance(nuc_out, tuple):
            nuc_out = [nuc_out]
        if isinstance(nuc_out, (list, np.ndarray)):
            x_out = Counter()
            for x in nuc_out:
                if isinstance(x, tuple):
                    i,n = x
                else:
                    i,n = 1,x
                if isinstance(n, str):
                    n = ion(n)
                assert isinstance(i, int)
                assert isinstance(n, Ion)
                x_out.update({n:i})
            nuc_out = x_out
        if isinstance(nuc_out, dict):
            nuc_out = Counter(nuc_out)
        assert isinstance(nuc_out, Counter)
        for n,i in nuc_out.items():
            if isinstance(n, str):
                del nuc_out[n]
                nuc_out.update({ion(n):i})

        N = Z = E = 0
        for n,i in nuc_in.items():
            assert isinstance(n, (Isotope, Isomer))
            N += n.N * i
            Z += n.Z * i
            E += n.E * i

        for n,i in nuc_out.items():
            assert isinstance(n, (Isotope, Isomer))
            N -= n.N * i
            Z -= n.Z * i
            E -= n.E * i

        dZ = 0
        dl = np.zeros(4, dtype=np.int64)
        dln0 = np.zeros(2, dtype=np.int64)
        dln1 = np.zeros(2, dtype=np.int64)
        for i,n in lep_in.items():
            dZ += n * i.Z
            f = i.flavor
            if f > 0:
                dl[f-1] += i.lepton_number * n
            else:
                if i.is_neutrino:
                    dln0[0] += 1
                else:
                    dln0[1] += 1
        for i,n in lep_out.items():
            dZ -= n * i.Z
            f = i.flavor
            if f > 0:
                dl[f-1] -= i.lepton_number * n
            else:
                if i.is_neutrino:
                    dln1[0] += 1
                else:
                    dln1[1] += 1
        dln = dln0 - dln1

        dl += lepton_violation

        if reversible is None:
            for l in lep_out:
                if l.Z == 0:
                    reversible = False
                    break
            else:
                reversible = True

        if autocomplete == 1:
            if Z == -N and Z+dZ != 0:
                lep_out.update({lepton(Z=1,number=sign(N), flavor=1):abs(N)})
                dZ += N
                dl[0] -= N

        # replace unspecified neutrinos by those of best-matching kind
        for i in (-1, 1):
            if i == -autocomplete:
                for j in (0, 1):
                    while dln1[j] > 0 and np.any(np.abs(dl)>0):
                        for l,n in lep_out.items():
                            if l.flavor == 0 and n > 0 and int(l.is_neutrino) + j == 1:
                                lep_out.update({l:-1})
                                dln1[j] -= 1
                                dln[j] += 1
                                if dln1[j] == 0:
                                    del lep_out[l]
                                for i,n in enumerate(dl):
                                    if n != 0:
                                        if j == 0:
                                            Zi = 0
                                        else:
                                            Zi = -sign(n)
                                            dZ -= Zi
                                        lep_out.update({lepton(Z=Zi,number=sign(n), flavor=i+1):1})
                                        dl[i] -= sign(n)
                                        break
                                break
            elif i == autocomplete:
                for j in (0, 1):
                    while dln0[j] > 0 and np.any(np.abs(dl)>0):
                        for l,n in lep_in.items():
                            if l.flavor == 0 and n > 0 and int(l.is_neutrino) + j == 1:
                                lep_in.update({l:-1})
                                dln0[j] -= 1
                                dln[j] -= 1
                                if dln0[j] == 0:
                                    del lep_in[l]
                                for i,n in enumerate(dl):
                                    if n != 0:
                                        if j == 0:
                                            Zi = 0
                                        else:
                                            Zi = -sign(n)
                                            dZ -= Zi
                                        lep_in.update({lepton(Z=Zi,number=-sign(n), flavor=i+1):1})
                                        dl[i] -= sign(n)
                                        break
                                break


        if autocomplete == 1 or autocomplete == -1 and not Z == N == 0:
            Z += dZ
            N -= dZ
            dZ = 0
            if N >= 0 and Z >= 0:
                try:
                    i = ion(N=N,Z=Z)
                except:
                    i = ion('-')
                if isinstance(i, (Isotope, Isomer)):
                    nuc_out.update({i:1})
                    Z = N = 0
            if N <= 0 and Z <= 0 and not N == Z == 0:
                try:
                    i = ion(N=-N,Z=-Z)
                except:
                    i = ion('-')
                if isinstance(i, (Isotope, Isomer)):
                    nuc_in.update({i:1})
                    Z = N = 0

            if N > 0:
                nuc_out.update({ion('n'):N})
                N = 0
            elif N < 0:
                nuc_in.update({ion('n'):-N})
                N = 0

            if Z > 0:
                nuc_out.update({ion('p'):Z})
                Z = 0
            elif Z < 0:
                nuc_in.update({ion('p'):-Z})
                Z = 0

            if Z != 0 or N != 0:
                raise Exception(f'AutoCompletion failed {N=} {Z=}')


        if autocomplete == 1:
            lep_out.update({lepton(Z=0,number=sign(n),flavor=i+1):abs(n) for i,n in enumerate(dl) if n != 0})
            dl[:] = 0
            for i in (0, 1):
                if dln[i] > 0:
                    lep_out.update({lepton(Z=0,flavor=0,number=i):dln[i]})
                elif dln[i] < 0:
                    lep_in.update({lepton(Z=0,flavor=0,number=i):-dln[i]})
                dln[i] = 0
        elif autocomplete == -1:
            lep_in.update({lepton(Z=0,number=sign(n),flavor=i+1):abs(n) for i,n in enumerate(dl) if n != 0})
            dl[:] = 0
            for i in (0, 1):
                if dln[i] > 0:
                    lep_out.update({lepton(Z=0,flavor=0,number=i):dln[i]})
                elif dln[i] < 0:
                    lep_in.update({lepton(Z=0,flavor=0,number=i):-dln[i]})
                dln[i] = 0

        # else:

        if check_Z:
            assert Z + dZ == 0, f'Charge not conserved {Z+dZ=}, {N=}, {E=}'
        if check_A:
            assert N + Z == 0, f'Mass not conserved {Z=}, {N=}, {E=}'
        if check_l:
            assert np.all(dl) == 0, f'Lepton number not conserved {dl}'
            assert dln[0] == 0, f'neutrino number not conserved {dln[0]}'
            assert dln[1] == 0, f'lepdon number not conserved {dln[1]}'
            assert np.all(dl == 0), f'lepton number not conserved {dl}'


        self.nuc_in = OrderedDict(sorted(nuc_in.items(), reverse = True))
        self.nuc_out = OrderedDict(sorted(nuc_out.items(), reverse = True))

        self.lep_in  = OrderedDict(sorted(lep_in.items()))
        self.lep_out = OrderedDict(sorted(lep_out.items()))

        self.gam_in  = OrderedDict(sorted(gam_in.items()))
        self.gam_out = OrderedDict(sorted(gam_out.items()))

        self.flags = flags

        self.reversible = reversible

    @CachedAttribute
    def inlist(self):
        ions = []
        for n,i in self.nuc_in.items():
            ions.extend([n]*i)
        return ions

    @CachedAttribute
    def outlist(self):
        ions = []
        for n,i in self.nuc_out.items():
            ions.extend([n]*i)
        return ions

    # @cachedmethod
    # def __str__(self):
    #     s = (' + '.join([str(n) for n in self.inlist]) +
    #          ' --> ' +
    #          ' + '.join([str(n) for n in self.outlist]))
    #     return(s)

    @cachedmethod
    def __str__(self):
        if self.reversible:
            arrow = ' --> '
        else:
            arrow = ' ==> '
        s = (' + '.join(
            [str(n) if i == 1 else f'{i} {n}' for n,i in self.nuc_in.items()] +
            [str(n) if i == 1 else f'{i} {n}' for n,i in self.lep_in.items()] +
            [str(n) if i == 1 else f'{i} {n}' for n,i in self.gam_in.items()]
            ) +
            arrow +
            ' + '.join(
            [str(n) if i == 1 else f'{i} {n}' for n,i in self.nuc_out.items()] +
            [str(n) if i == 1 else f'{i} {n}' for n,i in self.lep_out.items()] +
            [str(n) if i == 1 else f'{i} {n}' for n,i in self.gam_out.items()]
            ))
        return(s)

    @cachedmethod
    def mpl(self):
        s = (' $+$ '.join([n.mpl() for n in self.inlist]) +
             ' $\longmapsto$ ' +
             ' $+$ '.join([n.mpl() for n in self.outlist]))
        return(s)

    @cachedmethod
    def __repr__(self):
        return '[' + str(self) + ']'

    def __lt__(self, x):
        assert isinstance(x, self.__class__)
        for n,i, nx, ix in zip(self.nuc_in.keys(),
                               self.nuc_in.values(),
                               x.nuc_in.keys(),
                               x.nuc_in.values(),
                               ):
            if n < nx:
                return True
            elif n > nx:
                return False
            if i < ix:
                return True
            elif i > ix:
                return False
        if len(self.nuc_in) < len(x.nuc_in):
            return True
        elif len(self.nuc_in) > len(x.nuc_in):
            return False

        for n,i, nx, ix in zip(self.nuc_out.keys(),
                               self.nuc_out.values(),
                               x.nuc_out.keys(),
                               x.nuc_out.values(),
                               ):
            if n < nx:
                return True
            elif n > nx:
                return False
            if i < ix:
                return True
            elif i > ix:
                return False
        if len(self.nuc_out) < len(x.nuc_out):
            return True
        elif len(self.nuc_out) > len(x.nuc_out):
            return False

        if self.flags == None and x.flags != None:
            return True
        elif self.flags != None and x.flags == None:
            return False
        elif self.flags != None and x.flags != None:
            if self.flags != x.flags:
                return self.flags < x.flags

        # equal
        assert self.__eq__(x)
        return False

    def __eq__(self, x):
        assert isinstance(x, self.__class__)
        if len(self.nuc_in) != len(x.nuc_in):
            return False
        if len(self.nuc_out) != len(x.nuc_out):
            return False
        if not np.alltrue(np.array(list(self.nuc_in.values())) ==
                          np.array(list(x.nuc_in.values()))):
            return False
        if not np.alltrue(np.array(list(self.nuc_out.values())) ==
                          np.array(list(x.nuc_out.values()))):
            return False
        if not np.alltrue(np.array(list(self.nuc_in.keys())) ==
                          np.array(list(x.nuc_in.keys()))):
            return False
        if not np.alltrue(np.array(list(self.nuc_out.keys())) ==
                          np.array(list(x.nuc_out.keys()))):
            return False
        if not self.flags == x.flags:
            return False
        return True

    @CachedAttribute
    def reverse(self):
        """
        Return reverse reaction.
        """
        reverse = self.__class__('')
        reverse.nuc_out = self.nuc_in
        reverse.nuc_in  = self.nuc_out
        reverse.lep_out = self.lep_in
        reverse.lep_in  = self.lep_out
        reverse.gam_out = self.gam_in
        reverse.gam_in  = self.gam_out
        reverse.reversible = self.reversible
        reverse.flags   = self.flags
        return reverse

    def __hash__(self):
        return hash(str(self))
