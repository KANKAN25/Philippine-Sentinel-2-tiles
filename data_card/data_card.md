# Data card v0: EuroSAT (multispectral), source domain

- **Source:** https://zenodo.org/records/7711810 (EuroSAT_MS.zip); Helber et al., IEEE JSTARS 2019
- **Licence:** MIT; Copernicus Sentinel data terms also apply
- **Access date:** 2026-11-04
- **Checksum (EuroSAT_MS.zip):** MD5 091174add3c8e680a49244acf185b9f0 (matches Zenodo, verified 2026-11-04); SHA-256 eb0cf47380b0ff0dba68726bcb9184f8b261efcbb818d3d2b00b557ab0835b33
- **Images:** 27,000 GeoTIFF tiles, 64 x 64 pixels, 13 bands, dtype uint16
- **Classes (10) and counts:** AnnualCrop 3000, Forest 3000, HerbaceousVegetation 3000,
  Residential 3000, SeaLake 3000, Highway 2500, PermanentCrop 2500, Industrial 2500,
  River 2500, Pasture 2000
- **Value range (200-tile random sample, seed 0, all 13 bands pooled):** min 1, max 28,001, mean about 1,429
- **Ground resolution:** 10 m
- **Origin:** Sentinel-2A scenes of European Urban Atlas cities in 34 countries, low cloud cover, acquired across the year; tiles manually checked by the authors. No exact acquisition dates documented.
- **Processing level:** no atmospheric correction, so not surface reflectance. Level-1C is implied but not named by the authors.
- **+1000 offset:** not documented; unlikely for 2015-2017 imagery.
- **Band order:** confirmed with the paper: B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12. RGB = B04, B03, B02 = indices 3, 2, 1.
- **RGB conversion:** fixed stretch, clip(value, 0, 3000) / 3000 \* 255, same for all tiles

## Split rule

GroupShuffleSplit by 0.5-degree lon/lat grid block (from each tile's top-left GeoTIFF
coordinate), seed 1, 948 blocks. Resulting proportions: train 70.6%, val 11.3%, test 18.1%.
The original EuroSAT benchmark used a random class-wise 80/20 split, so
scores here are not directly comparable to its reported accuracy.

| class                | train | val  | test |
| -------------------- | ----- | ---- | ---- |
| AnnualCrop           | 1805  | 658  | 537  |
| Forest               | 2226  | 246  | 528  |
| HerbaceousVegetation | 2074  | 280  | 646  |
| Highway              | 1764  | 244  | 492  |
| Industrial           | 1844  | 213  | 443  |
| Pasture              | 1397  | 398  | 205  |
| PermanentCrop        | 1973  | 179  | 348  |
| Residential          | 2330  | 233  | 437  |
| River                | 1811  | 255  | 434  |
| SeaLake              | 1832  | 354  | 814  |
| **total**            | 19056 | 3060 | 4884 |

## Leakage check

Assertion passed: no block appears in more than one split.

## Problems found

- Mild class imbalance (2,000 to 3,000 per class); class mix differs across splits
  because splitting is by block, not by class. Pasture has only 205 test tiles.
- Tiles have no scene IDs; blocks are a proxy. Tiles on opposite sides of a block
  edge may still be neighbours.
- Fixed 0-3000 stretch clips bright pixels (values up to 28,001 exist).
- No atmospheric correction: some tiles have a colour cast. A Philippine target set built from Level-2A (surface reflectance) would not match this processing level.
- All 13 bands are stored at 64 x 64, so the 20 m and 60 m bands were resampled to the 10 m grid (inferred from file shapes).
