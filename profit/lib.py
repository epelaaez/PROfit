import profit

# re-export simple classes that we don't need to futz with
from _profit import SystStruct, PROpeller, PROsyst, PROspec, PROlog, PROsurf, LogLin, PROchi, EvalStrategy, LBFGSBParam, PROfitter, PROfitterConfig, PROmodel, NullModel, PROnumudis, PROnueapp, PRO3p1, PROdata
# re-export helper functions
from _profit import PROcess_CAFAna, FindGlobalBin, FindGlobalTrueBin, FillRecoSpectra, FillCVSpectrum, savePROsurf, CreatePROspecCV, CreatePROdata
# make Globals available
from _profit import Globals

# Override PROconfig to return our own BranchVariable objects
class PROconfig(profit._profit.PROconfig):
    def __init__(self, *args, **kwargs):
        profit._profit.PROconfig.__init__(self, *args, **kwargs)
        self._m_branch_variables = [[BranchVariable(b) for b in bs] for bs in super().m_branch_variables]

    @property
    def m_branch_variables(self):
        return self._m_branch_variables

# Override _profit BranchVariable to provide pythonic TTreeFormula functionality through DataFrameFormula
class BranchVariable(profit._profit.BranchVariable):
    def __init__(self, *args, **kwargs):
        profit._profit.BranchVariable.__init__(self, *args, **kwargs)

        # These we will all replace with DataFrameFormula objects
        self._branch_formula = None
        self._branch_true_L_formula = None
        self._branch_true_value_formula = None
        self._branch_true_pdg_formula = None
        self._branch_weight_formulas = []
    
    @property
    def branch_formula(self):
        return self._branch_formula
        
    @branch_formula.setter
    def branch_formula(self, v):
        if not isinstance(v, profit.pylib.DataFrameFormula): 
            raise ValueError("BranchVarible.branch_formula must be set to profit.DataFrameFormula")
        self._branch_formula = v

    @property
    def branch_true_L_formula(self):
        return self._branch_true_L_formula

    @branch_true_L_formula.setter
    def branch_true_L_formula(self, v):
        if not isinstance(v, profit.pylib.DataFrameFormula): 
            raise ValueError("BranchVarible.branch_true_L_formula must be set to profit.DataFrameFormula")
        self._branch_true_L_formula = v

    @property
    def branch_true_value_formula(self):
        return self._branch_true_value_formula
        
    @branch_true_value_formula.setter
    def branch_true_value_formula(self, v):
        if not isinstance(v, profit.pylib.DataFrameFormula): 
            raise ValueError("BranchVarible.branch_true_value_formula must be set to profit.DataFrameFormula")
        self._branch_true_value_formula = v

    @property
    def branch_true_pdg_formula(self):
        return self._branch_true_pdg_formula
        
    @branch_true_pdg_formula.setter
    def branch_true_pdg_formula(self, v):
        if not isinstance(v, profit.pylib.DataFrameFormula): 
            raise ValueError("BranchVarible.branch_true_pdg_formula must be set to profit.DataFrameFormula")
        self._branch_true_pdg_formula = v

    @property
    def branch_weight_formulas(self):
        return self._branch_weight_formulas

    @branch_weight_formulas.setter
    def branch_weight_formulas(self, v):
        if not isinstance(v, list):
            raise ValueError("BranchVariable.branch_weight_formulas must be set to a list of profit.DataFrameFormula")
        self._branch_weight_formulas = v

    def add_weight_formula(self, v):
        if not isinstance(v, profit.pylib.DataFrameFormula):
            raise ValueError("Weight formula must be a profit.DataFrameFormula")
        self._branch_weight_formulas.append(v)

    # Override BranchVariable::GetValue to use local DataFrameFormula
    def GetValue(self):
        if self.branch_formula is None:
            return 0

        return self.branch_formula.EvalInstance()

    # Override BranchVariable::GetTrueValue to use local DataFrameFormula
    def GetTrueValue(self):
        if self.branch_true_value_formula is None:
            return 0

        return self.branch_true_value_formula.EvalInstance()

    # Override BranchVariable::GetTrueL to use local DataFrameFormula
    def GetTrueL(self):
        if self.branch_true_L_formula is None:
            return 0

        return self.branch_true_L_formula.EvalInstance()
    
    # Override BranchVariable::GetTruePDG to use local DataFrameFormula
    def GetTruePDG(self):
        if self.branch_true_pdg_formula is None:
            return 0

        return self.branch_true_pdg_formula.EvalInstance()

    # Override BranchVariable::GetWeight to use local DataFrameFormula
    # Returns the weight at 0-based index i. Default to 1 if out of range.
    def GetWeight(self, i):
        if i >= 0 and i < len(self._branch_weight_formulas) and self._branch_weight_formulas[i] is not None:
            return self._branch_weight_formulas[i].EvalInstance()
        return 1

    # Override BranchVariable::GetTotalWeight to use local DataFrameFormula
    # Returns the product of all defined weights. Default to 1 if no weights.
    def GetTotalWeight(self):
        product = 1
        for f in self._branch_weight_formulas:
            if f is not None:
                product = product * f.EvalInstance()
        return product

    # Get number of defined weights
    def NumWeights(self):
        return len(self._branch_weight_formulas)

