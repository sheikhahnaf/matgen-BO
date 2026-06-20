# MatInvent Test Suite

Comprehensive test suite for the GP-based calculator routing system.

## Test Structure

```
tests/
├── unit/                        # Unit tests for individual components
│   ├── test_orb_calculator.py   # ORB calculator tests
│   ├── test_orb_featurizer.py   # ORB featurizer tests
│   ├── test_gp_surrogate.py     # GP model tests
│   ├── test_acquisition.py      # Acquisition function tests
│   ├── test_router.py           # Calculator router tests
│   └── test_gp_trainer.py       # GP training manager tests
│
├── integration/                 # Integration tests
│   ├── test_reward_integration.py   # Reward system with GP routing
│   └── test_ltm_integration.py      # LTM with GP data storage
│
├── end_to_end/                  # End-to-end tests
│   └── test_rl_with_gp.py       # Full RL pipeline with GP routing
│
├── fixtures/                    # Test fixtures
│   ├── structures.py            # Crystal structure fixtures
│   └── mock_calculators.py      # Mock calculator fixtures
│
├── conftest.py                  # Shared pytest configuration
└── README.md                    # This file
```

## Running Tests

### Prerequisites

Ensure you have the test environment set up:

```bash
conda activate matinvent
pip install pytest pytest-cov pytest-timeout
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Categories

**Unit tests only:**
```bash
pytest tests/unit/ -v
```

**Integration tests only:**
```bash
pytest tests/integration/ -v
```

**End-to-end tests only:**
```bash
pytest tests/end_to_end/ -v
```

### Run Tests by Marker

**Skip slow tests:**
```bash
pytest -m "not slow"
```

**Run only integration tests:**
```bash
pytest -m integration
```

**Run only unit tests:**
```bash
pytest -m unit
```

### Run Specific Test Files

```bash
pytest tests/unit/test_gp_surrogate.py -v
```

### Run Specific Test Functions

```bash
pytest tests/unit/test_gp_surrogate.py::test_gp_fit -v
```

## Test Coverage

Generate coverage report:

```bash
pytest --cov=rewards --cov=memory --cov=pipeline --cov-report=html
```

View coverage in browser:
```bash
open htmlcov/index.html
```

## Test Categories

### Unit Tests

Test individual components in isolation:

- **ORB Featurizer**: Embedding extraction, PCA reduction, batch processing
- **ORB Calculator**: Bulk modulus, shear modulus, formation energy calculations
- **GP Surrogate**: Training, prediction, uncertainty quantification
- **Acquisition Function**: EI calculation, calculator selection, cost optimization
- **Router**: Cold start, GP-based routing, cost tracking
- **GP Trainer**: Data collection, retraining, metrics tracking

### Integration Tests

Test component interactions:

- **Reward Integration**: GP routing in reward system, metadata extraction
- **LTM Integration**: Storing GP data (property values, features, calculator metadata)
- **GP Training Integration**: Online learning during RL

### End-to-End Tests

Test complete workflows:

- **RL with GP**: Full RL loop with GP routing, retraining, cost optimization
- **Metrics Tracking**: GP metrics saved correctly
- **Cost Reduction**: Verify GP routing reduces computational cost
- **Data Integrity**: LTM maintains correct data throughout RL

## Writing New Tests

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test functions: `test_<functionality>`
- Use descriptive names that explain what is being tested

### Using Fixtures

```python
def test_example(fcc_structure, mock_orb_calculator):
    """Test using shared fixtures."""
    result = mock_orb_calculator.calc([fcc_structure])
    assert result is not None
```

### Markers

Add markers to tests:

```python
@pytest.mark.slow
def test_expensive_operation():
    """This test takes a long time."""
    pass

@pytest.mark.integration
def test_component_interaction():
    """Test multiple components together."""
    pass
```

### Parameterization

Test multiple inputs:

```python
@pytest.mark.parametrize("n_samples,expected", [(10, 10), (50, 50), (100, 100)])
def test_batch_sizes(n_samples, expected):
    assert n_samples == expected
```

## Debugging Tests

### Verbose output:
```bash
pytest tests/unit/test_gp_surrogate.py -v -s
```

### Run with debugger:
```bash
pytest tests/unit/test_gp_surrogate.py --pdb
```

### Show print statements:
```bash
pytest tests/unit/test_gp_surrogate.py -s
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          conda env create -f env.yml
          conda activate matinvent
      - name: Run tests
        run: pytest tests/ -m "not slow"
```

## Test Data

Test fixtures provide:

- **Crystal structures**: FCC, BCC, HCP, rock salt, perovskite
- **Mock calculators**: Fast deterministic calculators for testing
- **Mock data**: Rewards, property values, features

## Known Limitations

- ORB calculator tests use mocks to avoid heavy computations
- Some end-to-end tests require significant memory
- GPU tests require CUDA-enabled GPU

## Troubleshooting

**Import errors:**
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/matinvent"
```

**Fixture not found:**
- Check that fixtures are imported in `conftest.py`
- Verify fixture is defined with `@pytest.fixture` decorator

**Tests hang:**
- Set timeout: `pytest --timeout=300`
- Check for infinite loops or missing mocks

## Contact

For test-related issues, please open an issue on GitHub.
