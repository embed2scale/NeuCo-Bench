import logging
from pathlib import Path
from typing import Callable, Tuple

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchgeo.models import ViTSmall16_Weights
from tqdm import tqdm

from data.submission_utils import create_submission_from_dict, test_submission
from data.dataset import (
    E2SChallengeDataset,
    Normalize,
    TemporalMean,
    InputResizer,
    collate_fn,
)

# Configurations
MODALITIES = ["s2l1c"] # We are using the Dino ViT model pretrained on SSL4EO L1C data.
INPUT_SIZE = 224 # Resize input images to the expected 224x224 pixels.
EMBEDDING_SIZE = 1024 # Set output embedding size, unused dimensions will be padded with zeros.
METHOD = "avg"  # How to reduce the per-patch features into a single vector; Options: ["avg", "cls", "max"].
DATA_PATH   = Path("./data") # Path to the SSL4EO-S12-downstream image directory.
OUTPUT_PATH = Path("./results.csv") # Path to the output CSV file.


# Load model backbone from TorchGeo
def load_torchgeo_model(
    model_name: str, weights_obj: ViTSmall16_Weights, device: torch.device
) -> Tuple[nn.Module, Callable]:
    """ 
    Loads a Timm model with specified TorchGeo weights.
    """
    in_chans = weights_obj.meta.get("in_chans")
    model = timm.create_model(model_name, in_chans=in_chans)
    state_dict = weights_obj.get_state_dict(progress=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    logging.info(
        "Loaded TorchGeo model '%s' with in_chans=%d",
        model_name,
        in_chans,
    )
    extractor = getattr(model, "forward_features", model)
    return extractor


# Main function to extract embeddings
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    weights = ViTSmall16_Weights.SENTINEL2_ALL_DINO
    extractor = load_torchgeo_model('vit_small_patch16_224', weights, device)

    transform = transforms.Compose([
        Normalize(), # scale to [0,1]
        TemporalMean() # average over 4 seasonal timesteps
        ])
    
    dataset = E2SChallengeDataset(
        DATA_PATH,
        modalities=MODALITIES,
        seasons=4,
        dataset_name='bands',
        transform=transform,
        concat=True,
        output_file_name=True,
    )

    logger.info(f'Dataset length: {len(dataset)} samples')
    sample = dataset[0]['data']
    logger.info(f'Sample data shape: {sample.shape}')

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    embeddings = {}
    resizer = InputResizer(INPUT_SIZE).to(device)

    # Define how to reduce patch embedding to a single vector
    reducers = {
        'avg': lambda f: f[:, 1:].mean(dim=1),
        'cls': lambda f: f[:, 0],
        'max': lambda f: f[:, 1:, :].max(dim=1)[0],
    } 

    # Extract embeddings
    for batch in tqdm(loader, desc='Extracting embeddings'):
        data = batch['data'].squeeze(0).to(device)
        data = resizer(data)

        with torch.no_grad():
            features = extractor(data)
            try:
                emb_compressed = reducers[METHOD](features)
            except KeyError:
                raise ValueError(f'Unknown METHOD {METHOD!r}')
        emb_flat = emb_compressed.flatten()
        
        # Pad with zeroes to fixed EMBEDDING_SIZE
        if emb_flat.shape[0] < EMBEDDING_SIZE: 
            emb_flat = F.pad(
                emb_flat,
                (0, EMBEDDING_SIZE - emb_flat.numel()),
            )

        embeddings[batch["file_name"][0]] = emb_flat.cpu().tolist()
    
    # Create and save embedding csv file
    submission_df = create_submission_from_dict(embeddings)
    logger.info(f"Number of embeddings: {len(submission_df)}")
    submission_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved embeddings to {OUTPUT_PATH}")

    # Validate output format
    ids = set(embeddings.keys())
    assert test_submission(
        OUTPUT_PATH,
        ids,
        EMBEDDING_SIZE,
    )


if __name__ == "__main__":
    main()