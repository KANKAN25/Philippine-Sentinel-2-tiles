# EuroSAT_MS file order (0-based)
BAND_IDX = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
            "B08": 7, "B09": 8, "B10": 9, "B11": 10, "B12": 11, "B8A": 12}
RGB = (BAND_IDX["B04"], BAND_IDX["B03"], BAND_IDX["B02"])   # (3, 2, 1)
RGB_NIR = RGB + (BAND_IDX["B08"],)                          # for E4