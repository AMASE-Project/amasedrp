# Contributing to `amasedrp`

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/amase/amasedrp.git
cd amasedrp

# Create conda environment
conda env create -f environment.yml
conda activate amasedrp

# Install in editable mode
pip install -e .
```

## Workflow

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Write code**: Follow existing code style
3. **Add tests**: Ensure new features are covered by tests
4. **Run tests**: `pytest`
5. **Commit**: Use clear and descriptive commit messages
6. **Push and open a Pull Request**

## Code Style

- Use **4 spaces** for indentation
- Follow **PEP 8** style guide
- Add type annotations where applicable
- Write docstrings (Google or NumPy style)

## Commit Message Format

```
<type>: <description>

[optional body]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test-related changes
- `refactor`: Code refactoring
- `chore`: Maintenance and miscellaneous

**Example**:
```feat: Add optimal extraction method```

## Pull Request Checklist

Before submitting a PR, please ensure:

- [ ] Code runs and `pytest` passes
- [ ] New features are covered by tests
- [ ] Documentation is updated (if needed)
- [ ] No type errors or syntax warnings
- [ ] Branch naming follows convention: `feature/xxx`, `fix/xxx`, `docs/xxx`

## Reporting Bugs

Please include the following information:
1. **Environment**: Python version, operating system
2. **Reproduction steps**: Minimal code or commands to reproduce the issue
3. **Expected result**: What you expected to happen
4. **Actual result**: What actually happened, including error messages

## Reference

- Project architecture: `architecture.md`
- User guide: `README.md`
- Tutorials: `docs/tutorials/`

## Code of Conduct

- Be friendly and respectful
- Welcome constructive discussions
- Accept contributors from all backgrounds and skill levels

---

If you have any questions, feel free to open an issue for discussion.
