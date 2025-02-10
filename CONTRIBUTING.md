# 🐦‍⬛🔍 Contributing to Klarity

Thank you for your interest in contributing to Klarity! We're excited to have you join our mission of bringing clarity to models behaviour. Whether you're contributing code, ideas, documentation, or just spreading the word about models interpretability & explainability, every contribution helps make Klarity better! 🎯

## Ways to Contribute

* **Test Models**: Help us expand our model support by testing and validating new models.
* **Improve Analysis**: Enhance our semantic clustering algorithms or uncertainty metrics.
* **Support the Community**: Answer questions, review PRs, or help others in discussions.
* **Fix Bugs**: Identify and resolve issues in the existing codebase.
* **Share Ideas**: Propose new features or improvements for better uncertainty analysis.

One of the best ways to support us is by sharing your experience with Klarity! Write about how you're using semantic uncertainty analysis in your projects, share interesting insights you've discovered, or help others understand the importance of model uncertainty. Even a simple star on our repository helps spread the word! ⭐

## Submitting Issues

### Reporting Bugs 🐛

1. **Check Existing Issues**: Search through existing issues to avoid duplicates.
2. **Environment Details**: Include:
   * Python version
   * Model type and size being analyzed
   * Operating system
   * Relevant package versions (pytorch, transformers, etc.)
   * A minimal code example that reproduces the issue

3. **Additional Context**: 
   * Expected vs actual behavior
   * Error messages or stack traces
   * Example prompts that trigger the issue
   * Sample uncertainty metrics output

### Feature Requests 💡

1. **Describe the Need**: What problem would this feature solve?
2. **Use Cases**: Share specific examples of how you'd use this feature.
3. **Proposed Solution**: If you have ideas about implementation, we'd love to hear them!

## Making Contributions ❤️

1. Fork the repository
2. We use uv as our python project manager. To learn how to install uv visit [here](https://docs.astral.sh/uv/getting-started/). After downloading uv create a .venv using `uv venv`
3. Install dependencies by running `uv pip install -r pyproject.toml`. Alternatively if you don't want to use uv you
can run `pip install -e ".[dev]"`
4. Create a new branch for your feature: `git checkout -b feature/amazing-feature`
5. Make your changes
6. Run tests: via `uv pytest` or `python -m pytest tests/`
7. Lint and clean your code using 
```
uv tool install ruff (one time only)
uvx ruff check 
uvx ruff clean
```
8. Submit a PR with a clear description of your changes


## Spread the Word! 📢

Your support extends beyond code:

* Write blog posts about using Klarity
* Share your uncertainty analysis results
* Star our repository to show your support

Thank you for helping make Klarity better! We look forward to seeing your contributions. 🔍✨