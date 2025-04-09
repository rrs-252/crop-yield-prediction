import re

def validate_district_name(name: str) -> bool:
    """Check district name format using dataset patterns"""
    return bool(re.match(r'^[A-Za-z\s\-\.\(\)]{3,50}$', name))

def validate_coordinates(lat: float, lon: float) -> None:
    """Raise detailed errors for invalid coordinates"""
    errors = []
    if not -90 <= lat <= 90:
        errors.append(f"Latitude {lat} out of range (-90 to 90)")
    if not -180 <= lon <= 180:
        errors.append(f"Longitude {lon} out of range (-180 to 180)")
    
    if errors:
        raise ValueError("Invalid coordinates: " + ", ".join(errors))
