import numpy as np
import pandas as pd

# Implement TTreeFormula-like functionality on a dataframe
class DataFrameFormula(object):
    def __init__(self, name, formula, df):
        self.name = name

        # fix binary operators to python
        self.formula = formula.replace("&&", "&").replace("||", "|")

        self.df = df

    def DefinedVariable(self, name):
        return name in self.df.columns

    def EvalInstance(self):
        # Return 0 on empty string (this is what ROOT does)
        if len(self.formula) == 0:
            return pd.Series(0., self.df.index, name=self.name)

        # inject dataframe columns into namespace
        namespace = dict([(c, self.df[c]) for c in self.df.columns])

        # also inject helper functions
        namespace["abs"] = np.abs
        namespace["sin"] = np.sin
        namespace["cos"] = np.cos
        namespace["tan"] = np.tan
        namespace["ln"] = np.log
        namespace["ln10"] = np.log10
        namespace["sqrt"] = np.sqrt
        namespace["exp"] = np.exp
        namespace["pow"] = np.power

        # and constants
        namespace["pi"] = np.pi
        namespace["e"] = np.e
        namespace["infinity"] = np.inf

        ret = eval(self.formula, namespace)

        # Make singleton values into series
        if not isinstance(ret, pd.Series):
            ret = pd.Series(ret, self.df.index)

        ret.name = self.name

        return ret.astype(np.float32)

    def PrintValue(self):
        return self.formula

SPLINE_COLUMNS = ["ms3", "ms2", "ms1", "cv", "ps1", "ps2", "ps3"]
SPLINE_SHIFTS = [-3, -2, -1, 0, 1, 2, 3]

def syst_index_name(index, mode):
    if mode == "spline": # Multisigma
        return SPLINE_COLUMNS[index]
    else: # Multisim
        return "univ_%i" % index
    # TODO: morph

# Dataframe object with specific layout for systematic weights. Adds a couple helper functions
class SystematicsDF(pd.DataFrame):
    def __init__(self, df):
        super().__init__(df)
        if self.empty: return

        # Fill in cv if we need to
        for s in self.systematics():
            syst = self.systematic(s)
            if syst.mode() == "spline" and "cv" not in syst.columns:
                self[s, "cv"] = 1.

    @staticmethod
    def build(df, modes):
        # Reformat to the structure we want -- a hierarchical index on the columns, flat index on the rows
        if isinstance(df.index, pd.MultiIndex) and not df.empty:
            assert(df.index.nlevels == 2)
            df = df.unstack()
            df.columns = pd.MultiIndex.from_tuples([(col[0], syst_index_name(col[1], modes[col[0]])) for col in df.columns])

        return SystematicsDF(df.dropna(axis=1, how='all'))

    @staticmethod
    def concat(dfs, **kwargs):
        if len(dfs) == 0:
            return SystematicsDF(pd.DataFrame())
        return SystematicsDF(pd.concat(dfs, **kwargs))

    def systematics(self):
        if self.empty:
            return []

        return self.columns.get_level_values(0).unique()

    def systematic(self, s):
        return SystematicSeries(self[s])

    def nvariation(self, s):
        return len([c for c in self.columns if c[0] == s])

class SystematicSeries(pd.DataFrame):
    def __init__(self, df):
        super().__init__(df)

    def nvariation(self):
        return len(self.columns)

    def universe(self, i):
        return self["univ_%i" % i]

    def shift(self, s):
        if s not in SPLINE_SHIFTS: 
            raise ValueError("SHIFT %f IS NOT AVAILABLE" % s)
        return self[SPLINE_COLUMNS[SPLINE_SHIFTS.index(s)]]

    def mode(self):
        return "multisim" if self.columns[0].startswith("univ") else "spline"

