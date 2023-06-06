import geopandas as gpd
from shapely.geometry import LineString, Point
from pyproj import Proj, transform
import utm


def interpolate_linestring(line, point_spacing):

    # Calculate the line length
    length_m = line.length

    # Calculate the number of points that need to be created
    num_points = int(length_m / point_spacing)

    # Create points
    for i in range(num_points + 1):
        # Calculate the distance along the line of the current point
        distance = i * point_spacing

        # Ensure that the distance is not greater than the line's length
        distance = min(distance, length_m)

        # Create a point at this distance along the line
        # Yield the original line and the interpolated point
        yield line.interpolate(distance)


def main():
    # Load the geojson file
    gdf = gpd.read_file('/Users/max/Development/radar-trajectory/geo_data/lane_centers.geojson')

    # infer the UTM zone from the geometry
    crs = gdf.estimate_utm_crs()

    # Reproject the geometry to the UTM zone
    gdf = gdf.to_crs(crs)
    
    # Set the point spacing
    point_spacing = 0.05

    # explode into line strings
    gdf = gdf.explode(index_parts=True)

    # interpolate points along the lines
    gdf.geometry = gdf.geometry.apply(lambda x: LineString(interpolate_linestring(x, point_spacing)))

    # drop all hte indexes and make a new one
    gdf = gdf.reset_index(drop=True)

    # gdf.drop(columns=['level_0', 'level_1'], inplace=True)

    # reproject back to WGS84
    gdf = gdf.to_crs(epsg=4326)

    # save the geojson file
    gdf.to_file('/Users/max/Development/radar-trajectory/geo_data/lane_centers_interp.geojson', driver='GeoJSON')


if __name__ == "__main__":
    main()
