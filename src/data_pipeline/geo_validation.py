import re
from typing import Tuple

def validate_district_name(name: str) -> bool:
    """Validate district name format"""
    return bool(re.match(r'^[A-Za-z\s\-\.\(\)]{3,50}$', name))

def validate_coordinates(lat: float, lon: float) -> Tuple[bool, str]:
    """Validate geographic coordinates"""
    errors = []
    if not -90 <= lat <= 90:
        errors.append("Latitude out of range (-90 to 90)")
    if not -180 <= lon <= 180:
        errors.append("Longitude out of range (-180 to 180)")
    return (len(errors) == 0, ", ".join(errors))
