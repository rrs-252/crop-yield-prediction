import rasterio
from rasterio.transform import rowcol

class SoilProcessor:
    def __init__(self, hwsd_path: str = "data/HWSD/hwsd.bil"):
        self.hwsd_path = hwsd_path
        self.soil_params = {
            1: "T_OC",    # Organic Carbon
            2: "PH_H2O",  # pH in water
            3: "T_CLAY",  # Clay content
            4: "T_SAND",  # Sand content
            5: "T_CEC"    # Cation Exchange Capacity
        }
        
    def get_soil_properties(self, lat: float, lon: float) -> dict:
        """Extract soil properties for coordinates"""
        try:
            with rasterio.open(self.hwsd_path) as src:
                row, col = rowcol(src.transform, lon, lat)
                return {
                    self.soil_params[band]: src.read(band)[row, col]
                    for band in self.soil_params.keys()
                }
        except (FileNotFoundError, IndexError) as e:
            print(f"Soil Data Error: {str(e)}")
            return None

# Example usage:
# soil = SoilProcessor().get_soil_properties(28.6139, 77.2090)  # Delhi coordinates
