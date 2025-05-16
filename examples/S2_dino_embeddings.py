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

# Configuration
MODALITIES = ["s2l1c"]
INPUT_SIZE = 224
EMBEDDING_SIZE = 1024
METHOD = "avg"  # Options: ["avg", "cls", "max"]
DATA_PATH =  Path("/path/to/data")
OUTPUT_PATH = Path("/path/to/output_dir/results.csv")

def load_torchgeo_model(
    model_name: str, weights_obj: ViTSmall16_Weights
) -> Tuple[nn.Module, Callable]:
    """ 
    Loads a Timm model with specified TorchGeo weights.
    """
    in_chans = weights_obj.meta.get("in_chans")
    model = timm.create_model(model_name, in_chans=in_chans)
    state_dict = weights_obj.get_state_dict(progress=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logging.info(
        "Loaded TorchGeo model '%s' with in_chans=%d",
        model_name,
        in_chans,
    )
    extractor = getattr(model, "forward_features", model)
    return extractor

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights = ViTSmall16_Weights.SENTINEL2_ALL_DINO
    extractor = load_torchgeo_model('vit_small_patch16_224', weights)

    transform = transforms.Compose([Normalize(), TemporalMean()])
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
    reducers = {
        'avg': lambda f: f[:, 1:].mean(dim=1),
        'cls': lambda f: f[:, 0],
        'max': lambda f: f[:, 1:, :].max(dim=1)[0],
    }
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
        if emb_flat.shape[0] < EMBEDDING_SIZE:
            emb_flat = F.pad(
                emb_flat,
                (0, EMBEDDING_SIZE - emb_flat.numel()),
            )

        embeddings[batch["file_name"][0]] = emb_flat.cpu().tolist()
        break

    submission_df = create_submission_from_dict(embeddings)
    logger.info(f"Number of embeddings: {len(submission_df)}")
    submission_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved embeddings to {OUTPUT_PATH}")

    ids = set(embeddings.keys())
    assert test_submission(
        OUTPUT_PATH,
        ids,
        EMBEDDING_SIZE,
    )


if __name__ == "__main__":
    main()