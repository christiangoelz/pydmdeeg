# Contributing to pydmdeeg

Thank you for your interest in contributing to pydmdeeg! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pydmdeeg.git
   cd pydmdeeg
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pydmdeeg --cov-report=html

# Run specific test file
pytest tests/test_dmd.py

# Run specific test
pytest tests/test_dmd.py::TestDMDInit::test_init_basic
```

### Code Quality

```bash
# Run linter
ruff check pydmdeeg/

# Auto-fix linting issues
ruff check pydmdeeg/ --fix

# Format code
ruff format pydmdeeg/

# Type checking
mypy pydmdeeg/
```

## Making Changes

### Branching Strategy

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, atomic commits.

3. Push your branch and create a pull request.

### Commit Messages

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or fixes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example:
```
feat: add support for multi-channel visualization

- Add plot_channels method for multi-channel DMD visualization
- Include colormap options for channel differentiation
- Update documentation with examples
```

### Pull Request Guidelines

1. **Title**: Clear, concise description of changes
2. **Description**: Include:
   - What changes were made
   - Why these changes were needed
   - Any breaking changes
   - Related issues (use `Fixes #123` to auto-close)

3. **Checklist**:
   - [ ] Tests pass locally
   - [ ] New features have tests
   - [ ] Documentation is updated
   - [ ] Code follows style guidelines

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation Style

- Use NumPy-style docstrings
- Include examples in docstrings where helpful
- Document all public functions and classes

Example:
```python
def compute_modes(
    data: np.ndarray,
    n_modes: int = 10,
) -> np.ndarray:
    """Compute DMD modes from data.

    Parameters
    ----------
    data : ndarray
        Input data matrix of shape (n_channels, n_times).
    n_modes : int, default=10
        Number of modes to compute.

    Returns
    -------
    modes : ndarray
        DMD modes of shape (n_channels, n_modes).

    Examples
    --------
    >>> data = np.random.randn(64, 1000)
    >>> modes = compute_modes(data, n_modes=5)
    >>> modes.shape
    (64, 5)
    """
```

## Testing Guidelines

- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Use fixtures for common setup
- Test edge cases and error conditions

## Reporting Issues

When reporting issues, please include:

1. **Environment**: Python version, OS, package version
2. **Description**: Clear description of the issue
3. **Reproduction**: Minimal code to reproduce
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Error messages**: Full traceback if applicable

## Questions?

Feel free to open an issue for questions or discussions about potential changes.

Thank you for contributing!
