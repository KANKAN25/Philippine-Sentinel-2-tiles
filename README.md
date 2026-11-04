# Group 4: EuroSAT to Philippine Sentinel-2 land-cover transfer
CMSC 178 IP, UP Cebu. Track A (domain shift): how well do EuroSAT-trained CNNs classify land cover in the Philippines, and which classes fail first?

## Status 
- Source data (EuroSAT MS): indexed, checksummed, split by spatial block. Done.
- Baseline (small CNN, RGB): in progress.
- Philippine target tiles: in progress.

## Where things are
- `data_card/data_card.md`: dataset facts, split rule, leakage check, problems found
- `data_card/splits.csv`: which tile is in train, val or test
- `data_card/tile_index.csv`: one row per tile (paths, CRS, coordinates)
- `data_card/sample_tiles.png`: one example tile per class
- `data_checksums.txt`: MD5 and SHA-256 of EuroSAT_MS.zip
- `01_data_card.ipynb`: builds the index, split and RGB PNGs
- `src/bands.py`: band name to file index lookup

## Before you use the data
1. EuroSAT MS is not in this repo. Download it from https://zenodo.org/records/7711810 and check the MD5 against `data_checksums.txt`.
2. The RGB PNGs (`data/train|val|test/<class>/`) are shared as a zip on Drive: [will insert the link after it finishes uploading].
3. Band order in the files is not the usual one (B8A is stored last, inferred from statistics). Select bands with `src/bands.py`.
4. The split is final. Do not change it, and do not use `test` until experiments are frozen.
5. EuroSAT has no atmospheric correction, so do not mix it with Level-2A scenes without recording the difference.
