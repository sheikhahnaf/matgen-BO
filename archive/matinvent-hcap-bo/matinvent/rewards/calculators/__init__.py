# Try to import calculators, but don't fail if dependencies are missing
try:
    from rewards.calculators.alignn.calc import ALIGNN
except (ImportError, RuntimeError) as e:
    import warnings
    warnings.warn(f"Could not import ALIGNN: {e}")
    ALIGNN = None

try:
    from rewards.calculators.pymatgen.calc import PyMatGen
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import PyMatGen: {e}")
    PyMatGen = None

try:
    from rewards.calculators.dft.calc import DFTCalc
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import DFTCalc: {e}")
    DFTCalc = None

try:
    from rewards.calculators.syn_score.calc import SynScore
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SynScore: {e}")
    SynScore = None

try:
    from rewards.calculators.fairchem.calc import FairChem
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import FairChem: {e}")
    FairChem = None

try:
    from rewards.calculators.orb.calc import ORBCalculator
    from rewards.calculators.orb.featurizer import ORBFeaturizer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ORB: {e}")
    ORBCalculator = None
    ORBFeaturizer = None
