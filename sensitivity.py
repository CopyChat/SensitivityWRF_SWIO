"""
to test WRF sensitivity at Reunion and Mauritius
"""

__version__ = f'Version 1.0  \nTime-stamp: <2021-05-10>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import xarray as xr
import hydra
from omegaconf import DictConfig
import GEO_PLOT


@hydra.main(config_path="configs", config_name="default")
def validation(cfg: DictConfig) -> None:
    # ----------------------------- compare -----------------------------
    if cfg.jobs.compare_sfc_var:
        for var in cfg.vars.sfc_var_list:
            for resolution in ['3km', '1km']:
                # example: 'U.addout_1km_d3.nc.temp.lastday.localtime.nc'
                d3_file = f'{cfg.dirs.local_data:s}/domain/d3/{var:s}.addout_{resolution:s}_d3.nc.temp.lastday.localtime.nc'
                d3 = GEO_PLOT.read_to_standard_da(d3_file, var)

                d4_file = f'{cfg.dirs.local_data:s}/domain/d4/{var:s}.addout_{resolution:s}_d4.nc.temp.lastday.localtime.nc'
                d4 = GEO_PLOT.read_to_standard_da(d4_file, var)

                hour = 12

                GEO_PLOT.plot_compare_2geo_maps(map1=d3.loc[d3.time.dt.hour == hour].squeeze(),
                                                map2=d4.loc[d4.time.dt.hour == hour].squeeze(),
                                                tag1='d3', tag2='d4',
                                                suptitle_add_word=f'resolution_{resolution:s} '
                                                                  f'hour_{hour:g}')
                print(f'good')

    if cfg.jobs.compare_vertical_var:
        for var in cfg.vars.vertical_var_list:
            for resolution in ['3km', '1km']:
                d3_file = f'{cfg.dirs.local_data:s}/domain/d3/{var:s}.addout_{resolution:s}_d3.nc.temp.lastday.localtime.nc'
                d3 = GEO_PLOT.read_to_standard_da(d3_file, var)

                d4_file = f'{cfg.dirs.local_data:s}/domain/d4/{var:s}.addout_{resolution:s}_d4.nc.temp.lastday.localtime.nc'
                d4 = GEO_PLOT.read_to_standard_da(d4_file, var)

                lev = 1  # 1000hPa, surface
                lev = 7  # 850hPa
                lev = 16  # 500hPa
                lev = 23  # 200hPa

                hour = 12

                map1 = d3.loc[d3.time.dt.hour == hour].sel(lev=lev).squeeze()
                map2 = d4.loc[d4.time.dt.hour == hour].sel(lev=lev).squeeze()

                GEO_PLOT.plot_compare_2geo_maps(map1=map1, map2=map2, tag1='d3', tag2='d4',
                                                suptitle_add_word=f'resolution_{resolution:s} '
                                                                  f'hour_{hour:g} '
                                                                  f'lev_{lev:g}')

                print(var, resolution, f'good')

    if cfg.jobs.compare_vertical_var_era5:

        era5_vertical_path = f'{cfg.dirs.local_data:s}/domain/era5/u.v.w.era5.20151229.nc'

        for resolution in ['3km', '1km']:
            for var, era5_var in zip(cfg.vars.vertical_var_list, cfg.vars.era5_vertical_var_list):

                d3_file = f'{cfg.dirs.local_data:s}/domain/d3/{var:s}.addout_{resolution:s}_d3.nc.temp.lastday.localtime.nc'
                d3 = GEO_PLOT.read_to_standard_da(d3_file, var)

                d4_file = f'{cfg.dirs.local_data:s}/domain/d4/{var:s}.addout_{resolution:s}_d4.nc.temp.lastday.localtime.nc'
                d4 = GEO_PLOT.read_to_standard_da(d4_file, var)

                era5 = GEO_PLOT.read_to_standard_da(era5_vertical_path, era5_var)
                era5 = GEO_PLOT.convert_da_shifttime(era5, 4 * 3600)

                for lev, era5_lev in zip([1, 7, 16, 23], [1000, 850, 500, 200]):
                    for model, tag in zip([d3, d4], ['d3', 'd4']):
                        print(var, era5_lev, resolution, tag)

                        hour = 12

                        map1 = model.loc[model.time.dt.hour == hour].sel(lev=lev).squeeze()
                        map2 = era5.loc[era5.time.dt.hour == hour].sel(lev=era5_lev).squeeze()

                        GEO_PLOT.plot_compare_2geo_maps(map1=map1, map2=map2, tag1=tag, tag2='era5',
                                                        suptitle_add_word=f'resolution_{resolution:s} '
                                                                          f'hour_{hour:g} '
                                                                          f'model_{tag:s} '
                                                                          f'lev_{era5_lev:g}')

                        print(f'end loop for domain settings')
                    print(f'end loop for lev')
                print(f'end loop for var')
            print(f'end loop for resolution')

    print(f'good')


if __name__ == "__main__":
    sys.exit(validation())
