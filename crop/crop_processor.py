import rasterio
from rasterio.transform import rowcol
from pathlib import Path

class CropProcessor:
    def __init__(self, ggcp10_dir: str = "data/GGCP10/"):
        self.ggcp10_dir = Path(ggcp10_dir)
        
    def get_crop_yield(self, lat: float, lon: float, crop: str, year: int) -> float:
        """Get crop yield from GGCP10 TIFF"""
        tiff_path = self.ggcp10_dir / f"{crop}_{year}.tiff"
        
        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
            
        try:
            with rasterio.open(tiff_path) as src:
                row, col = src.index(lon, lat)
                return src.read(1)[row, col]
        except (rasterio.RasterioError, IndexError) as e:
            print(f"Crop Data Error: {str(e)}")
            return None

# Example usage:
# yield_val = CropProcessor().get_crop_yield(28.6139, 77.2090, "wheat", 2020)
