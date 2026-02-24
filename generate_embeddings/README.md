## Example Workflows for NeuCo-Bench Embedding Generation

This folder contains examples for generating embeddings in the expected **NeuCo-Bench** format.

The `data/` folder provides a dataloader for the SSL4EO-S12 downstream dataset (the underlying NeuCo-Bench benchmark dataset), as well as utilities to generate and validate CSV files in the expected format.

Additionally, we provide:
- A sample notebook implementing a simple averaging baseline.
- A script to generate embeddings using a TorchGeo pretrained model backbone.

You can also generate embeddings in the expected format using the TerraTorch embedding generation workflow:  
https://github.com/terrastackai/terratorch/tree/main/examples/embeddings  
by specifying `"neuco_csv"` as the output format.