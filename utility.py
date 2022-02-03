import numpy as np
import xarray
import rioxarray
import os


def to_raster(in_xds, template_xds, out_file):
    # in_xds.rio.write_crs(template_xds.rio.crs)
    in_xds.rio.to_raster(out_file)


def to_tiff(input_tiff_path, data, model, name):
    rsc = xarray.open_rasterio(input_tiff_path)
    band = rsc.sel(band=1)
    data[np.isnan(data)] = rsc.attrs['nodatavals']
    band.data = data

    if not os.path.exists(path='./Preds/' + model):
        os.mkdir('./Preds/' + model)
    to_raster(band, rsc, './Preds/' + model + '/' + name)


def save_2d(row, col, index, value, model, name, path):
    mat = np.empty(row * col)
    mat[:] = np.nan
    mat[index] = value
    mat = mat.reshape(row, col)
    to_tiff(path + 'bio.tif', mat, model=model, name=name)
