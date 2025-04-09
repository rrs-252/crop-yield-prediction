import click
from src.utils.geospatial import GeoLocator
from src.utils.exceptions import GeoMatchError

@click.command()
@click.option('--district', help='District name (partial match allowed)')
@click.option('--lat', type=float, help='Latitude (-90 to 90)')
@click.option('--lon', type=float, help='Longitude (-180 to 180)')
@click.option('--crop', required=True, type=click.Choice([
    'Rice', 'Wheat', 'Maize', 'Pearl Millet', 'Finger Millet', 'Barley'
]))
def main(district, lat, lon, crop):
    """Crop Yield Prediction CLI"""
    geo = GeoLocator('data/geo/UnApportionedIdentifiers.csv')
    
    try:
        if district:
            if lat or lon:
                raise click.UsageError("Use either district OR coordinates")
            lat, lon = geo.coords_from_district(district)
            click.echo(f"Resolved {district} → {lat:.4f}, {lon:.4f}")
        elif lat and lon:
            district = geo.district_from_coords(lat, lon)
            click.echo(f"Resolved coordinates → {district}")
        else:
            raise click.UsageError("Must provide district or coordinates")
            
        # Proceed with prediction
        predict_yield(lat, lon, crop)
        
    except GeoMatchError as e:
        click.secho(f"Error: {str(e)}", fg='red')
