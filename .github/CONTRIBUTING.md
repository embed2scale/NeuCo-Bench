# Contributing 

Thank you for considering contributing to this project! Your efforts help to strengthen the open-source community. We welcome all forms of contributions, including but not limited to the following:

- Introduction of new downstream tasks and data
- Introduction of new evaluation methods
- Run data challenges
- Documentation updates, bug fixing, and general code improvement

## Workflow

0. For significant modifications or any bugs spotting, please consider opening an issue for discussion beforehand.
1. Fork and pull the latest repository (Click the **Fork** button on GitHub).
2. Clone your fork:
   ```sh
   git clone https://github.com/your-username/neuco-bench.git
   ```
3. Navigate into the project directory
   ```sh
   cd neuco-bench
   ```
4. Add the upstream repository
   ```sh
   git remote add upstream https://github.com/neuco-bench.git
   ```
5. Create a local branch
   ```sh
   git checkout -b feature-branch
   ```
   and use a descriptive branch name related to your changes, e.g. `feature/building-count-downstream-task`.
6. Make your changes following the [PEP 8](https://peps.python.org/pep-0008) coding standard.
7. Write clear and concise commit messages:
   ```sh
   git commit -m 'Short description of changes' -m'and more details'
   ```
8. Push your branch to your fork
   ```sh
   git push origin feature-branch  
   ```
9. Go to your fork on GitHub and click **Compare & pull request**. Provide a detailed explanation of your changes and link to relevant issues.


# Code of Conduct
As contributor we expect you to follow the [Code of Conduct as specified by the Linux Foundation}(https://docs.linuxfoundation.org/lfx/mentorship/mentor-guide/code-of-conduct).

## License
By contributing, you agree that your work will be licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).

## Need Help?
If you have any questions, feel free to open an issue or contact the project maintainers.


# Code structure 

The main code for NeuCo-Bench reside in `benchmark`.

## data

In `data`, basic modules for loading and processing downstream data and labels are located.

1. `embeddings` loads and processes files containing embeddings.
2. `labels.py` loads and processes label files.

## evaluation

`evaluation` contains code for evaluating embeddings on downstream tasks.

1. `evaluation`, main evaluation function
2. `linear_probing` contains code for repeatedly training Linear Probes for evaluating the embeddings on downstream tasks.
3. `metrics` contains metrics for various types of downstream tasks.
4. `results` aggregates results over multiplle downstream tasks and creates a leaderboard.
5. `visualizations` contains code to visualize intermediate results during training as well as overall results from evaluation.

## examples

`examples` contain examples of embedding creation, evaluation demos, and more.


# Adding new features

## Adding new downstream data

NeuCo-Bench is designed to be data agnostic. There are two main methods for adding new downstream tasks.

1. **Create your own dataset** either locally or on data repositories, e.g. Huggingface. Use the same structure as the [SSL4EO-S12-downstream](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) dataset, i.e. one folder `data` containing your data to embed (possibly in sub folders) and one folder `labels` containing one file per task containing a map between the data (`id`) and target (`label`). If you use new fodler structure or data types, this may require implementing new data loading functionality in this code base.
2. **Extend the [SSL4EO-S12-downstream](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) dataset** by creating a github issue on this repo, or contacting the admins of the [embed2scale](https://huggingface.co/embed2scale) Huggingface organization.

## Adding new evaluation methods

Please see the **Contributing** section above.
