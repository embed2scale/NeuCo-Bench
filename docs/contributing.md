# Contributing

We warmly welcome contributions to NeuCo-Bench!

You can help improve the project in many ways, for example:

- Adding **new downstream tasks or datasets**
- Benchmarking **new embedding or compression methods**
- Extending **evaluation metrics or ranking methods**
- Fixing **bugs** or enhancing **documentation and examples**
- Running your own **data challenges**

## How to Get Started

If you are planning a larger change, we recommend opening a GitHub issue first to discuss your idea.

For the technical contribution workflow (forking, branching, pull requests, coding standards), please see our GitHub contributing guide [GitHub Contributing Guide](https://github.com/embed2scale/NeuCo-Bench/blob/main/.github/CONTRIBUTING.md).

## Data and Task Contributions

NeuCo-Bench is designed to be data agnostic. There are two ways for adding new downstream tasks.

1. **Create your own dataset** either locally or on data repositories, e.g. Huggingface. Use the same structure as the [SSL4EO-S12-downstream](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) dataset, i.e. one folder `data` containing your data to embed and one folder `labels` containing one file per task containing a map between the data (`id`) and target (`label`).
2. **Extend the [SSL4EO-S12-downstream](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) dataset** by creating a Github issue on this repo, or contacting the admins of the [Embed2Scale](https://huggingface.co/embed2scale) Huggingface organization.

If you are unsure how to integrate your data or task, feel free to discuss it with us.

## License

By contributing to NeuCo-Bench, you agree that your contributions will be licensed under the **Apache 2.0 License**.
