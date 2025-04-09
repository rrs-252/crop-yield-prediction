from src.utils.geospatial import GeoLocator
from src.utils.exceptions import GeoMatchError
from src.validation.geo_validation import validate_coordinates

def predict_yield(args):
    geo = GeoLocator('data/UnApportionedIdentifiers.csv')
    
    try:
        if args.district:
            lat, lon = geo.coords_from_district(args.district)
        else:
            lat, lon = args.coords
            validate_coordinates(lat, lon)
            
        district = geo.district_from_coords(lat, lon)
        print(f"Predicting {args.crop} yield for {district}")
        # Existing prediction logic here
        
    except GeoMatchError as e:
        print(f"Error: {str(e)}")
