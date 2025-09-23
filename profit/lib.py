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
        self._branch_monte_carlo_weight_formula = None
    
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
    def branch_monte_carlo_weight_formula(self):
        return self._branch_monte_carlo_weight_formula
        
    @branch_monte_carlo_weight_formula.setter
    def branch_monte_carlo_weight_formula(self, v):
        if not isinstance(v, profit.pylib.DataFrameFormula): 
            raise ValueError("BranchVarible.branch_monte_carlo_weight_formula must be set to profit.DataFrameFormula")
        self._branch_monte_carlo_weight_formula = v

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

    # Override BranchVariable::GetMonteCarloWeight to use local DataFrameFormula
    # Default to 1 instead of 0 if empty
    def GetMonteCarloWeight(self):
        if self.branch_monte_carlo_weight_formula is None:
            return 1

        return self.branch_monte_carlo_weight_formula.EvalInstance()

