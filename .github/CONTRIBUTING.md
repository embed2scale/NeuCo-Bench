# Contributing 

Thank you for considering contributing to this project! Your efforts help to strengthen the open-source community. We welcome all forms of contributions, including but not limited to the following:

- Introduction of new downstream tasks and data
- Introduction of new evaluation methods
- Run data challenges
- Documentation updates, bug fixing, and general code improvement

## Workflow

### Before You Contribute

For significant changes or bug reports, please open an issue first to discuss your proposed changes.

### Development Workflow


1. **Fork the repository** — Click the **Fork** button on GitHub to create your own copy.

2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/neuco-bench.git
   cd neuco-bench
   ```

3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/neuco-bench/neuco-bench.git
   ```

4. **Create a feature branch** with a descriptive name:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   Example: `feature/building-count-downstream-task`

5. **Make your changes** following the [PEP 8](https://peps.python.org/pep-0008) coding standard.

6. **Write clear commit messages**:
   ```bash
   git commit -m "Short description of changes" -m "More detailed explanation if needed"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** — Go to your fork on GitHub and click **Compare & pull request**. Provide a clear description of your changes and link any related issues.


## Code of Conduct
As contributor we expect you to follow the [Code of Conduct as specified by the Linux Foundation](https://docs.linuxfoundation.org/lfx/mentorship/mentor-guide/code-of-conduct).

## License
By contributing, you agree that your work will be licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).

## Need Help?
If you have any questions, feel free to open an issue or contact the project maintainers.


# Code structure 

Understanding the codebase will help you navigate and contribute effectively. The main code for NeuCo-Bench reside in `benchmark`:

```
benchmark/
├── data/
│   ├── embeddings.py       # Load and process embedding files
│   └── labels.py           # Load and process label files
├── evaluation/
│   ├── evaluation.py       # Main evaluation pipeline
│   ├── linear_probing.py   # Linear probe training and evaluation
│   ├── metrics.py          # Metrics
│   ├── results.py          # Results aggregation and leaderboard
│   └── visualizations.py   # Training and results visualizations
```

## Generate Embeddings

`generate_embeddings/` contain examples of embedding creation, dataloaders, and more.

## Adding Downstream Tasks

NeuCo-Bench is designed to be data agnostic. There are two main methods for adding new downstream tasks.

1. **Create your own dataset** either locally or on data repositories, e.g. Huggingface. Use the same structure as the [SSL4EO-S12-downstream](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) dataset, i.e. one folder `data` containing your data to embed (possibly in sub folders) and one folder `labels` containing one file per task containing a map between the data (`id`) and target (`label`). If you use a new folder structure or data types, this may require implementing new data loading functionality in this code base.
2. **Extend the [SSL4EO-S12-downstream](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) dataset** by creating a github issue on this repo, or contacting the admins of the [embed2scale](https://huggingface.co/embed2scale) Huggingface organization.

## Adding Evaluation Methods

Please see the **Contributing** section above.
