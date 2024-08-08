import time  # always good to have
from pathlib import Path

import numpy as np
from starfit.autils.abuset import AbuData, AbuSet
from starfit.autils.human.time import time2human
from .ionmap import decay
from starfit.autils.isotope import ion as I
from starfit.autils.stardb import StarDB
from tqdm import tqdm

_translate = {
    "h": "h1",
    "p": "h1",
    "a": "he4",
    "n": "nt1",
}


class YisoNod(object):
    """
    class for Limongi & Chieffi 2012 *.yiso_nod files
    """

    def __init__(self, filename):
        abu = dict()
        with open(filename, "rt") as f:
            next(f)
            for l in f:
                x = l.split(',')
                i = x[0].lower()
                i = I(_translate.get(i, i)).isotope()
                a = float(x[5])
                try:
                    abu[i] += a
                except:
                    abu[i] = a
        abu = AbuSet(abu, sort=True)
        self.velocity = float(Path(filename).stem[1:4])
        self.mass = float(Path(filename).stem[6:8])
        self.metallicity = -float(Path(filename).stem[-1])
        # ejecta = abu.norm()
        # self.remnant = self.mass - ejecta
        self.abu = abu.normalized()


class Data(object):

    def __init__(self, sdir, silent=False):
        self.sdir = Path(sdir)
        self.silent = silent
        pattern = "*.dat"
        files = list()
        for filename in (self.sdir).glob(pattern):
            files.append(filename)
        if not self.silent:
            print(f" [{self.__class__.__name__}] Found {len(files)} models.")
        self.files = files
        self.db = self.make_stardb(mode='el')
        # self.make_stardb(files[0], mode='el')

    def load_dumps(self):
        starttime = time.time()
        if not hasattr(self, "dumps"):
            if not self.silent:
                print(f" [{self.__class__.__name__}] loading dumps.")
            dumps = list()
            for filename in tqdm(self.files):
                dumps.append(YisoNod(filename))
            self.dumps = dumps
            if not self.silent:
                print(f" [{self.__class__.__name__}] loaded {len(dumps)} nucleo dumps.")
        if not self.silent:
            print(
                f" [{self.__class__.__name__}] finished in {time2human(time.time() - starttime)}"
            )

    def make_stardb(
        self,
        filename=None,
        mode=(
            "alliso",
            "radiso",
            "deciso",
            "el",
        ),
    ):
        # def make_stardb(self, filename=None, mode='el', ):
        starttime = time.time()
        self.load_dumps()

        if isinstance(mode, (list, tuple)):
            assert filename is None
            dbs = list()
            for m in mode:
                db = self.make_stardb(mode=m)
                dbs.append(db)
            return dbs

        comments = (
            "Presupernova Evolution and Explosive Nucleosynthesis of Rotating Massive Stars in the Metallicity Range -3<=[Fe/H]<=0",
            "Limongi & Chieffi, ApJS, 237, 13 (2018).",
        )

        if not (hasattr(self, "abu") and hasattr(self, "fielddata")):
            dtype = np.dtype([
                ('velocity', np.float64),
                ('mass', np.float64),
                ('metallicity', np.float64),])

            fielddata = list()
            abu = list()
            for d in tqdm(self.dumps):
                velocity = d.velocity
                mass = d.mass
                metallicity = d.metallicity
                fielddata.append((velocity, mass, metallicity))
                abu.append(d.abu)
            
            print(len(abu))
            abu = AbuData.from_abusets(abu)
            fielddata = np.array(fielddata, dtype=dtype)

            ii = np.argsort(fielddata)
            fielddata = fielddata[ii]
            abu.data = abu.data[ii]

            self.abu = abu
            self.fielddata = fielddata

        parms = dict()

        basename = "rot_LC18"
        if mode == "el":
            data = decay(self.abu, molfrac_out=True, elements=True, decay=False)
            parms["abundance_type"] = StarDB.AbundanceType.element
            parms["abundance_class"] = StarDB.AbundanceClass.dec
        elif mode == "alliso":
            data = self.abu.as_molfrac()
            parms["abundance_type"] = StarDB.AbundanceType.isotope
            parms["abundance_class"] = StarDB.AbundanceClass.raw
        elif mode == "radiso":
            data = decay(self.abu, molfrac_out=True, decay=False, stable=False)
            parms["abundance_type"] = StarDB.AbundanceType.isotope
            parms["abundance_class"] = StarDB.AbundanceClass.dec
        elif mode == "deciso":
            data = decay(self.abu, molfrac_out=True, decay=False, stable=True)
            parms["abundance_type"] = StarDB.AbundanceType.isotope
            parms["abundance_class"] = StarDB.AbundanceClass.dec
        else:
            raise AttributeError(f"Unknown {mode=}.")

        parms["name"] = f"{basename}.{mode}.y"
        parms["comments"] = comments

        parms["data"] = data
        parms["fielddata"] = self.fielddata

        parms["fieldnames"] = ['velocity', 'mass', 'metallicity']
        parms["fieldunits"] = ['km s^-1', 'solar masses', 'dex']
        parms["fieldtypes"] = [StarDB.Type.float64] * 3
        parms["fieldformats"] = ["6.3F", "6.3F", "1.1F"]
        parms["fieldflags"] = [StarDB.Flags.parameter] * 3
        parms["abundance_unit"] = StarDB.AbundanceUnit.mol_fraction
        parms["abundance_total"] = StarDB.AbundanceTotal.ejecta
        parms["abundance_norm"] = None
        parms["abundance_data"] = StarDB.AbundanceData.all_ejecta
        parms["abundance_sum"] = StarDB.AbundanceSum.number_fraction

        db = StarDB(**parms)
        if filename is None:
            filename = parms["name"] + ".stardb.xz"
        db.write(filename)
        
        if not self.silent:
            print(
                f" [{self.__class__.__name__}] finished in {time2human(time.time() - starttime)}"
            )
        return db