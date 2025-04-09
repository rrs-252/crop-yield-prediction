import click
from src.core.predict import predict_yield
from src.utils.geospatial import GeoLocator

@click.command()
@click.option('--district', help='District name')
@click.option('--lat', type=float, help='Latitude')
@click.option('--lon', type=float, help='Longitude')
@click.option('--crop', required=True, 
              type=click.Choice(['Rice', 'Wheat', 'Maize', 'Pearl Millet',
                               'Finger Millet', 'Barley']),
              help='Crop type')
def main(district, lat, lon, crop):
    geo = GeoLocator('data/geo/UnApportionedIdentifiers.csv')
    
    try:
        if district:
            lat, lon = geo.coords_from_district(district)
            click.echo(f"Resolved {district} to coordinates: {lat:.4f}, {lon:.4f}")
        elif lat and lon:
            district = geo.district_from_coords(lat, lon)
            click.echo(f"Resolved coordinates to district: {district}")
        else:
            raise click.UsageError("Must provide either --district or --lat/--lon")
            
        # Call prediction logic
        predict_yield(lat, lon, crop)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)

if __name__ == '__main__':
    main()
