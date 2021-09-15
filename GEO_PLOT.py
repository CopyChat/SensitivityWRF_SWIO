"""
functions used for process and plot geo data
hard linked with that in CODE/
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-02-21>'
__author__ = "ChaoTANG@univ-reunion.fr"

import os
import sys
from typing import List
import warnings
import hydra
import seaborn
from omegaconf import DictConfig
import cftime
import glob
import pandas as pd
import calendar
import numpy as np
from dateutil import tz
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def my_coding_rules():
    """

    :return:
    :rtype:
    """

    print(f'classification file in pd.DataFrame with DateTimeIndex')
    print(f'geo-field in DataArray with good name and units')
    print(f'see the function read_to_standard_da for the standard dim names')


# ----------------------------- definition -----------------------------

def get_possible_standard_coords_dims(name_for: str = 'coords', ndim: int = 1):
    """
    that the definition of all project, used by read nc to standard da
    :return:
    :rtype:
    """

    standard_coords = ['time', 'lev', 'lat', 'lon', 'number']
    standard_dims_1d = ['time', 'lev', 'y', 'x', 'num']
    standard_dims_2d = ['time', 'lev', ['y', 'x'], ['y', 'x'], 'num']
    # this definition is used for all projects, that's the best order do not change this
    # key word: order, dims, coords,

    if name_for == 'coords':
        return standard_coords
    if name_for == 'dims':

        if ndim == 1:
            return standard_dims_1d
        if ndim == 2:
            return standard_dims_2d
        else:
            return 0


# -----------------------------
class ValidationGrid:

    def __init__(self, vld_name: str, vld: xr.DataArray,
                 ref_name: str, ref: xr.DataArray):
        self.vld_name = vld_name
        self.ref_name = ref_name
        self.vld = vld
        self.ref = ref
        self.vld_var = vld.name
        self.ref_var = ref.name

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def stats(self) -> None:
        print('name', 'dimension', 'shape', 'max', 'min', 'std')
        print(self.vld_name, self.vld.dims, self.vld.shape,
              self.vld.max().values, self.vld.min().values, self.vld.std())
        print(self.ref_name, self.ref.dims, self.ref.shape,
              self.ref.max().values, self.ref.min().values, self.ref.std())
        print(f'-----------------------')

        return None

    @property
    def plot_vis_a_vis(self):
        a = self.vld.values.ravel()
        b = self.ref.values.ravel()

        plt.title(f'{self.vld_name:s} vs {self.ref_name}')

        # lims = [
        #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        # ]

        vmin = np.min([self.vld.min().values, self.ref.min().values])
        vmax = np.max([self.vld.max().values, self.ref.max().values])

        lims = [vmin, vmax]

        fig = plt.figure(dpi=220)
        ax = fig.add_subplot(1, 1, 1)

        plt.scatter(a, b)
        plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f'{self.ref_name:s} ({self.ref.assign_attrs().units:s})')
        ax.set_ylabel(f'{self.vld_name:s} ({self.vld.assign_attrs().units:s})')
        # fig.savefig('./plot/dd.png', dpi=300)

        plt.show()

        print(value_lonlatbox_from_area('d01'))

        return fig

    @property
    def plot_validation_matrix(self):
        """
        here plot maps of
        Returns
        -------
        """

        return 1


class CmipVarDir:

    def __init__(self, path: str):
        self.path = path

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def nc_file(self):
        return glob.glob(f'{self.path:s}/*.nc')

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def gcm(self):
        files = self.nc_file
        gcm = list(set([s.split('_')[2] for s in files]))
        gcm.sort()
        # todo: when produce new file in the dir, the results would be different/wrong
        return gcm

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def ssp(self):
        files = self.nc_file
        ssp = list(set([s.split('_')[3] for s in files]))
        return ssp

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def var(self):
        files = self.nc_file
        file_names = [s.split('/')[-1] for s in files]
        var = list(set([s.split('_')[0] for s in file_names]))
        return var

    @property
    def freq(self):
        files = self.nc_file
        freq = list(set([s.split('_')[1] for s in files]))
        return freq


def convert_multi_da_to_ds(list_da: list, list_var_name: list) -> xr.Dataset:
    ds = list_da[0].to_dataset(name=list_var_name[0])

    if len(list_da) > 1:
        # Save one DataArray as dataset
        for i in range(1, len(list_da)):
            # Add second DataArray to existing dataset (ds)
            ds[list_var_name[i]] = list_da[i]

    return ds


def read_binary_file(bf: str):
    """
    to read binary file
    :param bf:
    :type bf:
    :return:
    :rtype:
    """

    # example: f'./local_data/MSG+0415.3km.lon'

    data = np.fromfile(bf, '<f4')  # little-endian float32
    # data = np.fromfile(bf, '>f4')  # big-endian float32

    return data


def convert_ds_to_da(ds: xr.Dataset, varname: str = 'varname') -> xr.DataArray:
    """
    get all the value from ds and put them into a da, with the coordinates from ds

    Parameters
    ----------
    ds : with each variable < 2D
    varname: usually there are several var names in the ds, as numbers of ensemble, for example.

    Returns
    -------
    da: with dims=('var', 'latitude', 'longitude')
    """
    list_var = list(ds.keys())
    first_da = ds[list_var[0]]

    all_values = np.stack([ds[x].values for x in list_var], axis=0)

    da = 0

    if len(ds.coord_names) == 3:
        # there's another coord beside 'longitude' 'latitude'
        time_coord_name = [x for x in ds.coord_names if x not in ['longitude', 'latitude']][0]
        time_coord = ds.coords[time_coord_name]

        units = ds.attrs['units']
        da = xr.DataArray(data=all_values, coords={'var': list_var, time_coord_name: time_coord,
                                                   'latitude': first_da.latitude, 'longitude': first_da.longitude},
                          dims=('var', time_coord_name, 'latitude', 'longitude'), name=varname,
                          attrs={'units': units})

    if len(ds.coord_names) == 2:
        # only 'longitude' and 'latitude'

        da = xr.DataArray(data=all_values, coords={'var': list_var,
                                                   'latitude': first_da.latitude, 'longitude': first_da.longitude},
                          dims=('var', 'latitude', 'longitude'), name=varname)

    return da


def convert_da_to_360day_monthly(da: xr.DataArray) -> xr.DataArray:
    """
    Takes a DataArray. Change the
    calendar to 360_day and precision to monthly.
    @param da: input with time
    @return:
    @rtype:
    """
    val = da.copy()
    time1 = da.time.copy()
    i_time: int
    for i_time in range(val.sizes['time']):
        year = time1[i_time].dt.year.__int__()
        mon = time1[i_time].dt.month.__int__()
        day = time1[i_time].dt.day.__int__()
        # bb = val.time.values[i_time].timetuple()
        time1.values[i_time] = cftime.Datetime360Day(year, mon, day)

    val = val.assign_coords({'time': time1})

    return val


def find_two_bounds(vmin: float, vmax: float, n: int):
    """
    find the vmin and vmax in 'n' interval
    :param vmin:
    :param vmax:
    :param n:
    :return:
    """
    left = round(vmin / n, 0) * n
    right = round(vmax / n, 0) * n

    return left, right


@hydra.main(config_path="configs", config_name="config.ctang")
def query_data(cfg: DictConfig, mysql_query: str, remove_missing_data=True):
    """
    select data from DataBase
    :return: DataFrame
    """

    from sqlalchemy import create_engine
    import pymysql
    pymysql.install_as_MySQLdb()

    db_connection_str = cfg.MySQLdb.db_connection_str
    db_connection = create_engine(db_connection_str)

    df: pd.DataFrame = pd.read_sql(sql=mysql_query, con=db_connection)

    # ----------------------------- remove two stations with many missing data -----------------------------
    if remove_missing_data:
        df.drop(df[df['station_id'] == 97419380].index, inplace=True)
        df.drop(df[df['station_id'] == 97412384].index, inplace=True)

    # ----------------------------- remove two stations with many missing data -----------------------------
    df['Datetime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('Datetime')
    df = df.drop(['DateTime'], axis=1)

    return df


@hydra.main(config_path="configs", config_name="config.ctang")
def query_influxdb(cfg: DictConfig, query: str):
    """
    select data from DataBase
    :return: DataFrame
    """
    from influxdb import DataFrameClient

    host = cfg.SWIOdb.host
    user = cfg.SWIOdb.user
    password = cfg.SWIOdb.password
    # port = 8088
    dbname = cfg.SWIOdb.dbname

    client = DataFrameClient(host, user, password, dbname)

    df = client.query(query)

    df = df.set_index('DateTime')

    return df


def plot_wrf_domain(num_dom: int, domain_dir: str):
    """
    to plot the domain setting from the output of geogrid.exe
    Parameters
    ----------
    num_dom :
    domain_dir :

    Returns
    -------
    """

    # ----------------------------- Definition:
    Num_Domain = num_dom
    Title = 'Domain setting for WRF simulation'

    # ----------------------------- plot a empty map -----------------------------
    import matplotlib.patches as patches
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    fig = plt.figure(figsize=(10, 8), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    # set map
    swio = value_lonlatbox_from_area('detect')
    ax.set_extent(swio, crs=ccrs.PlateCarree())
    # lon_left, lon_right, lat_north, lat_north

    ax.coastlines('50m')
    ax.add_feature(cfeature.LAND.with_scale('10m'))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(list(range(0, 100, 2)))
    gl.ylocator = mticker.FixedLocator(list(range(-90, 90, 2)))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    # ----------------------------- end of plot a empty map -----------------------------

    # swio:
    # resolution = [27, 9, 3, 1]
    # Poovanum:
    # resolution = [10, 2, 0.4, 0.4]
    # Chandra:
    # resolution = [27, 9, 3, 1]
    # detect
    resolution = [27, 9, 3, 1]

    colors = ['black', 'blue', 'green', 'red']

    resolution = resolution[4 - Num_Domain:]
    colors = colors[4 - Num_Domain:]

    for d in range(Num_Domain):
        # ----------------------------- read domains -----------------------------
        # swio:
        nc_file = f'{domain_dir:s}/geo_em.d0{d + 1:g}.nc'

        # Poovanum:
        # nc_file = f'{DATA_DIR:s}/Poovanum/geo_em.d0{d + 1:g}.nc'
        # Chandra:
        # nc_file = f'{DATA_DIR:s}/Chandra/domain/geo_em.d0{d + 1:g}.nc'

        lat = xr.open_dataset(nc_file).CLAT
        lon = xr.open_dataset(nc_file).CLONG
        # map_factor = xr.open_dataset(nc_file).MAPFAC_M
        height = lat.max() - lat.min()
        width = lon.max() - lon.min()
        # --------------------- plot the domains ------------------------------

        p = patches.Rectangle((lon.min(), lat.min()), width, height,
                              fill=False, alpha=0.9, lw=2, color=colors[d])
        ax.add_patch(p)

        # domain names and resolution:
        plt.text(lon.max() - 0.05 * width, lat.max() - 0.05 * height,
                 f'd0{d + 1:g}: {resolution[d]:g}km',
                 fontsize=8, fontweight='bold', ha='right', va='top', color=colors[d])
        # grid points:

        plt.text(lon.min() + 0.5 * width, lat.min() + 0.01 * height,
                 f'({lon.shape[2]:g}x{lat.shape[1]})',
                 fontsize=8, fontweight='bold', ha='center', va='bottom', color=colors[d])

    aladin_domain = 0

    if aladin_domain:
        # add ALADIN domain:
        #  [34°E-74°E ; 2°S-28°S]:
        p1 = patches.Rectangle((34, -28), 40, 26, fill=False, alpha=0.9, lw=2, color='black')
        ax.add_patch(p1)

        plt.text(64, -29, f'ALADIN domain @12km', fontsize=8, fontweight='bold', ha='center', va='bottom',
                 color='black')

    plt.grid(True)
    plt.title(Title, fontsize=12, fontweight='bold')
    plt.show()

    return fig


# noinspection PyUnresolvedReferences
def plot_station_value(lon: pd.DataFrame, lat: pd.DataFrame, value: np.array, cbar_label: str,
                       fig_title: str, bias=False):
    """
    plot station locations and their values
    :param bias:
    :type bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :param value:
    :return: map show
    """
    import matplotlib as mpl

    fig = plt.figure(dpi=220)
    fig.suptitle(fig_title)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
    # lon_left, lon_right, lat_north, lat_north

    ax.add_feature(cfeature.LAND.with_scale('10m'))
    # ax.add_feature(cfeature.OCEAN.with_scale('10m'))
    # ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    # ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    # ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
    # ax.add_feature(cfeature.RIVERS.with_scale('10m'))

    # ax.coastlines()

    # ----------------------------- stations -----------------------------

    if np.max(value) - np.min(value) < 10:
        round_number = 2
    else:
        round_number = 0

    n_cbar = 10
    vmin = round(np.min(value) / n_cbar, round_number) * n_cbar
    vmax = round(np.max(value) / n_cbar, round_number) * n_cbar

    if bias:
        cmap = plt.cm.coolwarm
        vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
        vmax = max(np.abs(vmin), np.abs(vmax))
    else:
        cmap = plt.cm.YlOrRd

    bounds = np.linspace(vmin, vmax, n_cbar + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # ----------------------------------------------------------
    sc = plt.scatter(lon, lat, c=value, edgecolor='black',
                     # transform=ccrs.PlateCarree(),
                     zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

    # ----------------------------- color bar -----------------------------
    cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label)
    cb.ax.tick_params(labelsize=10)

    ax.gridlines(draw_labels=True)
    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)

    plt.show()
    print(f'got plot')


def get_zenith_angle(df: pd.DataFrame, datetime_col: str, utc: bool,
                     lon: np.ndarray, lat: np.ndarray, column_zenith: str):
    """
    get solar zenith angle at (lon, lat) according to df with DateTimeUTC
    :param column_zenith:
    :param utc:
    :param datetime_col:
    :param df:
    :param lon:
    :param lat:
    :return:
    """

    import pytz
    from pysolar.solar import get_altitude

    if utc:
        df['DateTimeUTC_2'] = df[datetime_col]
    else:
        # add timezone info, which is needed by pysolar
        df['DateTimeUTC_2'] = [df.index.to_pydatetime()[i].astimezone(pytz.timezone("UTC"))
                               for i in range(len(df.index))]

    print(f'starting calculation solar zenith angle')
    zenith = [90 - get_altitude(lat[i], lon[i], df['DateTimeUTC_2'][i]) for i in range(len(df))]
    # prime meridian in Greenwich, England

    # df_new = df.copy()
    # df_new['utc'] = df['DateTimeUTC_2']
    # df_new[column_zenith] = zenith
    df_new = pd.DataFrame(columns=[column_zenith], index=df.index, data=zenith)

    output_file = r'/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Prediction_PV/zenith.csv'
    df_new.to_csv(output_file)

    # ----------------------------- for test -----------------------------
    # import datetime
    # date = datetime.datetime(2004, 11, 1, 00, 00, 00, tzinfo=datetime.timezone.utc)
    #
    # for lat in range(100):
    #     lat2 = lat/100 - 21
    #     a = 90 - get_altitude(55.5, lat2, date)
    #     print(lat2, a)
    # ----------------------------- for test -----------------------------

    return df_new


def zenith_angle_reunion(df, ):
    """
    to get zenith angle @ la reunion
    input: df['DateTime']
    """
    from pysolar.solar import get_altitude

    lat = -22  # positive in the northern hemisphere
    lon = 55  # negative reckoning west from
    # prime meridian in Greenwich, England

    return [90 - get_altitude(lat, lon, df[i])
            for i in range(len(df))]


def get_color():
    """define some (8) colors to use for plotting ... """

    # return [plt.cm.Spectral(each)
    #         for each in np.linspace(0, 6, 8)]

    # return ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    return ['pink', 'darkviolet', 'blue', 'teal', 'forestgreen', 'darkorange', 'red',
            'deeppink', 'blueviolet', 'royalblue', 'lightseagreen', 'limegreen', 'yellowgreen', 'tomato',
            'silver', 'gray', 'black']


def convert_ttr_era5_2_olr(ttr: xr.DataArray, is_reanalysis: bool):
    """
    as the name of function
    :param is_reanalysis:
    :param ttr:
    :return:
    """

    # using reanalyse of era5:
    # The thermal (also known as terrestrial or longwave) radiation emitted to space at the top of the atmosphere
    # is commonly known as the Outgoing Longwave Radiation (OLR). The top net thermal radiation (this parameter)
    # is equal to the negative of OLR. This parameter is accumulated over a particular time period which depends on
    # the data extracted. For the reanalysis, the accumulation period is over the 1 hour up to the validity date
    # and time. For the ensemble members, ensemble mean and ensemble spread, the accumulation period is over the
    # 3 hours up to the validity date and time. The units are joules per square metre (J m-2). To convert to
    # watts per square metre (W m-2), the accumulated values should be divided by the accumulation period
    # expressed in seconds. The ECMWF convention for vertical fluxes is positive downwards.

    if is_reanalysis:
        factor = -3600
    else:
        factor = -10800

    if isinstance(ttr, xr.DataArray):
        # change the variable name and units
        olr = xr.DataArray(ttr.values / factor,
                           coords=[ttr.time, ttr.lat, ttr.lon],
                           dims=ttr.dims, name='OLR', attrs={'units': 'W m**-2',
                                                             'long_name': 'OLR'})

    return olr


def plot_class_occurrence_and_anomaly_time_series(classif: pd.DataFrame, anomaly: xr.DataArray):
    """
    as the title,
    project: MIALHE_2020 (ttt class occurrence series with spatial mean seasonal anomaly)
    :param classif:
    :type classif:
    :param anomaly:
    :type anomaly:
    :return:
    :rtype:
    """

    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, wspace=0, hspace=0)
    ax = fig.add_subplot(grid[0, 0])

    print(f'anomaly series')

    start = anomaly.index.year.min()
    end = anomaly.index.year.max()

    anomaly = anomaly.squeeze()

    ax.plot(range(start, end + 1), anomaly, marker='o')
    ax.set_ylim(-20, 20)

    ax.set_xticklabels(classif['year'])

    class_name = list(set(classif['class']))
    for i in range(len(class_name)):
        event = classif[classif['class'] == class_name[i]]['occurrence']
        cor = np.corrcoef(anomaly, event)[1, 0]

        ax.text(0.5, 0.95 - i * 0.08,
                f'cor with #{class_name[i]:g} = {cor:4.2f}', fontsize=12,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    print(f'bar plot')
    fig.add_subplot(grid[-1, 0])
    import seaborn as sns
    sns.set(style="whitegrid")
    sns.barplot(x='year', y="occurrence", hue=f'class', data=classif)

    fig.suptitle(f'reu spatial mean ssr anomaly and olr regimes number each season')

    plt.savefig('./plot/anomaly_seris_olr_regimes.png', dpi=300)
    plt.show()


def multi_year_daily_mean(var: xr.DataArray):
    """
    calculate multi_year_daily mean of variable: var.
    :param var:
    :return:
    """
    ydaymean = var[var.time.dt.year == 2000].copy()  # get structure
    ydaymean[:] = var.groupby(var.time.dt.dayofyear).mean(axis=0, dim=xr.ALL_DIMS).values

    return ydaymean


def anomaly_daily(da: xr.DataArray) -> xr.DataArray:
    """
    calculate daily anomaly, as the name, from the xr.DataArray
    :param da:
    :return:
    """

    print(f'daily anomaly ...')

    anomaly = da.groupby(da.time.dt.strftime('%m-%d')) - da.groupby(da.time.dt.strftime('%m-%d')).mean('time')

    return_anomaly = anomaly.assign_attrs({'units': da.assign_attrs().units, 'long_name': da.assign_attrs().long_name})

    return_anomaly.rename(da.name)

    return return_anomaly


def anomaly_hourly(da: xr.DataArray, percent: int = 0) -> xr.DataArray:
    """
    calculate hourly anomaly, as the name, from the xr.DataArray
    if the number of year is less than 30, better to smooth out the high frequency variance.
    :param da:
    :param percent: output in percentage
    :return: da with a coordinate named 'strftime' but do not have this dim
    """

    if len(set(da.time.dt.year.values)) < 30:
        warnings.warn('CTANG: input less than 30 years ... better to smooth out the high frequency variance, '
                      'for more see project Mialhe_2020/src/anomaly.py')

    print(f'calculating hourly anomaly ... {da.name:s}')
    # todo: print out the var name

    if percent:
        anomaly = (da.groupby(da.time.dt.strftime('%m-%d-%H')) -
                   da.groupby(da.time.dt.strftime('%m-%d-%H')).mean('time')) / \
                  da.groupby(da.time.dt.strftime('%m-%d-%H')).mean('time')
    else:
        anomaly = da.groupby(da.time.dt.strftime('%m-%d-%H')) - da.groupby(da.time.dt.strftime('%m-%d-%H')).mean('time')

    return_anomaly = anomaly.assign_attrs({'units': da.assign_attrs().units, 'long_name': da.assign_attrs().long_name})

    return_anomaly.rename(da.name)

    return return_anomaly


def remove_duplicate_list(mylist: list) -> list:
    """
    Remove Duplicates From a Python list
    :param mylist:
    :return:
    """

    list_return = list(dict.fromkeys(mylist))

    return list_return


def plot_geo_subplot_map(geomap, vmin, vmax, bias, ax,
                         domain: str, tag: str,
                         plot_cbar: bool = True,
                         statistics: bool = 1):
    """
    plot subplot
    Args:
        geomap ():
        vmin ():
        vmax ():
        bias ():
        ax ():
        domain ():
        tag ():
        plot_cbar ():
        statistics ():

    Returns:

    """

    plt.sca(ax)
    # active this subplot

    # set up map:
    set_basemap(ax, area=domain)

    # vmax = geomap.max()
    # vmin = geomap.min()
    cmap, norm = set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, bias=bias)

    cf: object = ax.contourf(geomap.lon, geomap.lat, geomap, levels=norm.boundaries,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')

    if plot_cbar:
        cbar_label = f'{geomap.name:s} ({geomap.assign_attrs().units:s})'
        plt.colorbar(cf, orientation='vertical', shrink=0.8, pad=0.05, label=cbar_label)

    ax.text(0.9, 0.95, f'{tag:s}', fontsize=12,
            horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    if statistics:
        ax.text(0.95, 0.05, f'mean = {geomap.mean().values:4.2f}', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    return cf


def get_data_in_classif(da: xr.DataArray, df: pd.DataFrame, significant: bool = 0, time_mean: bool = 0):
    """
    to get a new da with additional dim of classification df da and df may NOT in the same length of time.
    Note that, this will produce a lot of nan if a multi time steps are not belonging to the same class.
    :param time_mean:
    :param significant:
    :param da: with DataTimeIndex,
    :param df: class with DataTimeIndex
    :return: in shape of (:,:,class)
    :rtype: da with additional dim named with class number
    """

    # get info:
    class_column_name = df.columns[0]
    class_names = np.sort(list(set(df[class_column_name])))

    print(f'get data in class...')

    for i in range(len(class_names)):
        cls = class_names[i]
        date_class_one: pd.DatetimeIndex = df.loc[df[class_column_name] == cls].index
        if len(date_class_one) < 1:
            print(f'Sorry, I got 0 day in phase = {cls:g}')
            break
        class_1: xr.DataArray = \
            da.where(da.time.dt.strftime('%Y-%m-%d').isin(date_class_one.strftime('%Y-%m-%d')), drop=True)

        if significant:
            sig_map = value_significant_of_anomaly_2d_mask(field_3d=class_1)
            class_1 = filter_2d_by_mask(class_1, mask=sig_map)
            # class_1 is the only significant values of all the time steps in one class.

        if i == 0:
            data_in_class = class_1
        else:
            data_in_class = xr.concat([data_in_class, class_1], dim='class')

        print(f'class = {cls:g}', data_in_class.shape)

    # output:
    if time_mean:
        data_in_class = data_in_class.mean('time')

    output_da = data_in_class.assign_coords({'class': class_names}).rename(da.name).assign_attrs(
        {'units': da.attrs['units']}).transpose(..., 'class')

    return output_da


def convert_unit_era5_flux(flux: xr.DataArray, is_ensemble: bool = 0):
    """
    convert to W/m2
    :param is_ensemble:
    :type is_ensemble:
    :param flux:
    :type flux:
    :return:
    :rtype:
    """

    # ----------------------------- attention -----------------------------
    # For the reanalysis, the accumulation period is over the 1 hour
    # ending at the validity date and time. For the ensemble members,
    # ensemble mean and ensemble spread, the accumulation period is
    # over the 3 hours ending at the validity date and time. The units are
    # joules per square metre (J m-2 ). To convert to watts per square metre (W m-2 ),
    # the accumulated values should be divided by the accumulation period
    # expressed in seconds. The ECMWF convention for vertical fluxes is
    # positive downwards.

    print(f'convert flux unit to W/m**2 ...')
    if is_ensemble:
        factor = 3600 * 3
    else:
        factor = 3600 * 1

    da = flux / factor

    da = da.rename(flux.name).assign_attrs({'units': 'W/m**2',
                                            'long_name': flux.assign_attrs().long_name})

    return da


def plot_cyclone_in_classif(classif: pd.DataFrame,
                            radius: float = 3,
                            cen_lon: float = 55.5,
                            cen_lat: float = -21.1,
                            suptitle_add_word: str = ''
                            ):
    """
    to plot classification vs cyclone
    Args:
        cen_lat ():
        cen_lon ():
        radius ():
        classif (): classification in df with DateTimeIndex, and only one column of 'class',
                    the column name could be any string
        suptitle_add_word ():

    Returns:
        maps with cyclone path

    """

    # read cyclone
    cyclone_file = f'~/local_data/cyclones.2.csv'
    cyc = pd.read_csv(cyclone_file)
    cyc['Datetime'] = pd.to_datetime(cyc['DAT'])
    df_cyclone = cyc.set_index('Datetime')

    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    lat_min = cen_lat - radius
    lat_max = cen_lat + radius

    lon_min = cen_lon - radius
    lon_max = cen_lon + radius

    print(f'plot cyclone within {int(radius): g} degree ...')
    # ----------------------------- prepare fig -----------------------------
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex='row', sharey='col', figsize=(12, 10), dpi=220,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.9, wspace=0.09, hspace=0.01)
    axs = axs.ravel()

    # plot in class
    for c in range(n_class):
        c_name = class_names[c]
        print(f'plot class = {str(c_name):s}')

        class_one: pd.DataFrame = classif[classif == c_name].dropna()

        # ----------------------------- plotting -----------------------------
        ax = axs[c]
        plt.sca(axs[c])  # active this subplot
        set_basemap(area='m_r_m', ax=ax)

        total = 0
        # loop of day from classification:
        for i in range(len(class_one)):
            all_cyc_1day = df_cyclone[df_cyclone.index.date == class_one.index.date[i]]
            # could be one or more cyclone, so length < 4 * n_cyc

            if len(all_cyc_1day) < 1:  # if no cycle that day:
                pass
            else:  # if with cyclone records, one or more

                name = all_cyc_1day['NOM_CYC']
                # name could be the length > 1, since the cyclone file is 6-hourly.

                # sometimes there are 2 cyclones at the same day:
                cyclone_name_1day = set(list(name.values))
                num_cyclone_1day = len(cyclone_name_1day)
                if num_cyclone_1day > 1:
                    print(f'got more than one cyclones in one day')

                # to see if these cyclones pass within the $radius
                for cyc in cyclone_name_1day:

                    cyc_day = all_cyc_1day[all_cyc_1day['NOM_CYC'] == cyc]

                    lat1 = cyc_day[cyc_day['NOM_CYC'] == cyc]['LAT']
                    lon1 = cyc_day[cyc_day['NOM_CYC'] == cyc]['LON']

                    # if @ reunion
                    record_in_radius = 0
                    for record in range(len(lat1)):
                        if lat_min <= lat1[record] <= lat_max:
                            if lon_min <= lon1[record] <= lon_max:
                                record_in_radius += 1

                    if record_in_radius > 0:
                        # add this cyclone in today (do this in every day if satisfied)
                        total += 1
                        # plot path (up to 6 points) if one or more of these 6hourly records is within a radius
                        plt.plot(lon1, lat1, marker='.', label='path within a day')  # only path of the day
                        # plt.legend(loc='lower left', prop={'size': 8})

                        # full_path of this cyclone
                        # full_path_cyc = df_cyclone[df_cyclone['NOM_CYC'] == cyc]
                        # plt.plot(full_path_cyc['LON'], full_path_cyc['LAT'])

                        # output this cyclone:
                        print(i, total, record_in_radius, cyc_day)

        # ----------------------------- end of plot -----------------------------

        plt.title(f'#{c + 1:g}')
        ax.text(0.96, 0.95, f'cyclone@reu={total:g}\n'
                            f'total_day={len(class_one):g}\n'
                            f'{100 * total / len(class_one):4.1f}%',
                fontsize=12, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.06, 0.01, f'plot only the path within a day',
                fontsize=12, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

        # ----------------------------- end of plot -----------------------------
    title = f'cyclone within {radius:g} degree of Reunion'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)
    plt.savefig(f'./plot/{title.replace(" ", "_"):s}.radius_{radius:g}.deg'
                f'.png', dpi=300)

    plt.show()
    print(f'got plot')


def select_nearby_cyclone(cyc_df: pd.DataFrame,
                          lon_name: str = 'lon',
                          lat_name: str = 'lat',
                          radius: float = 3,
                          cen_lon: float = 55.5,
                          cen_lat: float = -21.1
                          ):
    """
    from cyclone record select nearby events
    Args:
        cyc_df ():
        lon_name ():
        lat_name ():
        radius ():
        cen_lon ():
        cen_lat ():

    Returns:
        df (): only nearby records

    key word: selecting, DataFrame, lonlat, radius, nearby, cyclone, df
    applied_project: Mialhe_2020
    """

    df = cyc_df.loc[
        (cyc_df[lat_name] >= cen_lat - radius) &
        (cyc_df[lat_name] <= cen_lat + radius) &
        (cyc_df[lon_name] >= cen_lon - radius) &
        (cyc_df[lon_name] <= cen_lon + radius)
        ]

    return df


def plot_diurnal_boxplot_in_classif(classif: pd.DataFrame, field_1D: xr.DataArray,
                                    suptitle_add_word: str = '',
                                    anomaly: int = 0,
                                    percent: int = 0,
                                    plot_big_data_test: int = 1):
    """

    Args:
        classif ():
        field_1D (): dims = time, in this func by 'data_in_class', get da in 'time' & 'class'

        suptitle_add_word ():
        anomaly (int): 0
        percent (int): 0 calculate relative anomaly
        plot_big_data_test ():

    Returns:

    Applied_project:
        Mialhe_2020
    """
    # ----------------------------- data -----------------------------
    data_in_class = get_data_in_classif(da=field_1D, df=classif, time_mean=False, significant=0)

    # to convert da to df: for the boxplot:
    print(f'convert DataArray to DataFrame ...')
    df = data_in_class.to_dataframe()
    df['Hour'] = df._get_label_or_level_values('time').hour
    df['Class'] = df._get_label_or_level_values('class')
    # key word: multilevel index, multi index

    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    # ----------------------------- plot -----------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w', edgecolor='k', dpi=300)

    seaborn.boxplot(x='Hour', y=data_in_class.name, hue='Class', data=df, ax=ax,
                    showmeans=False, showfliers=False)
    # Seaborn's showmeans=True argument adds a mark for mean values in each box.
    # By default, mean values are marked in green color triangles.

    if anomaly:
        plt.axhline(y=0.0, color='r', linestyle='-', zorder=-5)

    if percent:
        ax.set_ylim(-0.2, 0.2)
    else:
        ax.set_ylim(-120, 120)

    title = f'{field_1D.assign_attrs().long_name:s} percent={percent:g} anomaly={anomaly:g} in class'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)

    # ----------------------------- end of plot -----------------------------

    plt.savefig(f'./plot/diurnal.{title.replace(" ", "_"):s}.'
                f'test_{plot_big_data_test:g}.png', dpi=300)

    plt.show()
    print(f'got plot ')


def plot_diurnal_cycle_field_in_classif(classif: pd.DataFrame, field: xr.DataArray,
                                        area: str, vmax, vmin,
                                        bias: bool = True,
                                        plot_wind: bool = 0,
                                        only_significant_points: bool = 0,
                                        suptitle_add_word: str = '',
                                        plot_big_data_test: int = 1):
    """
        diurnal field in classification, data are processed before input to this question
    note:
        sfc circulation from era5 @27km, not anomaly, to show sea/land breeze during a day.
        the field, such as rsds, will be hourly anomaly, in one class, value at one hour - multiyear
        seasonal hourly mean, (%m-%d-%h), which is the difference between in-class value and all-day value.
    Args:
        classif (pandas.core.frame.DataFrame): DatetimeIndex with class name (1,2,3...)
        field (xarray.core.dataarray.DataArray): with time dim. field may in different days of classif,
            it will be selected by the available classif day by internal function.
        area (str):
        vmax (int):
        vmin (int):
        bias (bool):
        plot_wind (int):
        only_significant_points (int):
        suptitle_add_word (str):
        plot_big_data_test (int): if apply the data filter defined inside the function to boost the code,
            usually defined in the project level in the config file.

    Returns:

    """

    # ----------------------------- data -----------------------------
    data_in_class = get_data_in_classif(da=field, df=classif, time_mean=False,
                                        significant=only_significant_points)
    print(f'good')
    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # class_column_name = classif.columns.to_list()[0]

    fig, axs = plt.subplots(nrows=n_class, ncols=n_hour, sharex='row', sharey='col',
                            figsize=(19, 10), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.02, top=0.9, wspace=0.09, hspace=0.01)

    for cls in range(n_class):
        print(f'plot class = {cls + 1:g}')

        in_class = data_in_class.where(data_in_class['class'] == class_names[cls], drop=True).squeeze()
        hourly_mean = in_class.groupby(in_class.time.dt.hour).mean(keep_attrs=True)

        for hour in range(n_hour):

            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            cf = plot_geo_subplot_map(geomap=hourly_mean.sel(hour=hours[hour]),
                                      vmax=vmax, vmin=vmin, bias=bias,
                                      plot_cbar=False,
                                      statistics=0,
                                      ax=ax, domain=area, tag='')

            if cls == 0:
                ax.set_title(f'{hours[hour]:g}H00')
            if hour == 0:
                ax.set_ylabel(f'Class_{str(class_names[cls]):s}')
                plt.ylabel(f'Class_{str(class_names[cls]):s}')

    # ----------------------------- surface wind -----------------------------
    if plot_wind:
        print(f'plot surface wind ...')

        print(f'loading data ... ')
        warnings.warn('CTANG: load data from 1999-2016, make sure that the data period is correct,'
                      ' check and check it again')

        local_data = '/Users/ctang/local_data/era5/Mialhe_2020'
        # u = read_to_standard_da(f'{local_data:s}/'
        #                         f'u10.hourly.era5_land.1981-2018.{area:s}.NDJF.local_day_time.nc', 'u10')
        # v = read_to_standard_da(f'{local_data:s}/'
        #                         f'v10.hourly.era5_land.1981-2018.{area:s}.NDJF.local_day_time.nc', 'v10')
        u = read_to_standard_da(f'{local_data:s}/'
                                f'u10.hourly.era5.1979-2018.{area:s}.NDJF.local_day_time.nc', 'u10')
        v = read_to_standard_da(f'{local_data:s}/'
                                f'v10.hourly.era5.1979-2018.{area:s}.NDJF.local_day_time.nc', 'v10')
        # classif OLR is from 1979 to 2018.

        if plot_big_data_test:
            u = u.sel(time=slice('19990101', '20001201'))
            v = v.sel(time=slice('19990101', '20001201'))

        u_in_class = get_data_in_classif(u, classif, significant=False, time_mean=False)
        v_in_class = get_data_in_classif(v, classif, significant=False, time_mean=False)

        for cls in range(n_class):
            print(f'plot wind in class = {cls + 1:g}')

            u_in_1class = u_in_class.where(u_in_class['class'] == class_names[cls], drop=True).squeeze()
            v_in_1class = v_in_class.where(v_in_class['class'] == class_names[cls], drop=True).squeeze()

            u_hourly_mean = u_in_1class.groupby(u_in_1class.time.dt.hour).mean()
            v_hourly_mean = v_in_1class.groupby(v_in_1class.time.dt.hour).mean()

            for hour in range(n_hour):
                plt.sca(axs[cls, hour])
                ax = axs[cls, hour]

                u_1hour = u_hourly_mean.sel(hour=hours[hour])
                v_1hour = v_hourly_mean.sel(hour=hours[hour])

                plot_wind_subplot(area='bigreu',
                                  lon=u_1hour.lon, lat=v_1hour.lat,
                                  u=u_1hour,
                                  v=v_1hour,
                                  ax=ax, bias=0)

    # ----------------------------- end of plot -----------------------------
    cbar_label = f'{field.name:s} ({field.assign_attrs().units:s})'
    cb_ax = fig.add_axes([0.87, 0.2, 0.01, 0.7])
    plt.colorbar(cf, orientation='vertical', shrink=0.8, pad=0.05, cax=cb_ax, label=cbar_label)

    # ----------------------------- title -----------------------------
    if plot_wind:
        suptitle_add_word += ' (surface wind)'

    title = f'{field.assign_attrs().long_name:s} in class'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)

    # ----------------------------- end of plot -----------------------------

    plt.savefig(f'./plot/hourly_mean.{field.name:s}.in_class.{area:s}.'
                f'wind_{plot_wind:g}.only_sig_{only_significant_points:g}.'
                f'test_{plot_big_data_test:g}.png', dpi=300)
    plt.show()
    print(f'got plot ')


def plot_wind_subplot(area: str,
                      lon: xr.DataArray, lat: xr.DataArray,
                      u: xr.DataArray, v: xr.DataArray,
                      ax, bias: int = 0):
    """
    to plot circulation to a subplot
    Args:
        area ():
        lon ():
        lat ():
        u ():
        v ():
        ax ():
        bias (int):

    Returns:

    """

    # speed = np.sqrt(u10 ** 2 + v10 ** 2)
    # speed = speed.rename('10m_wind_speed').assign_coords({'units': u10.attrs['units']})

    # Set up parameters for quiver plot. The slices below are used to subset the data (here
    # taking every 4th point in x and y). The quiver_kwargs are parameters to control the
    # appearance of the quiver so that they stay consistent between the calls.

    if area == 'bigreu':
        n_sclice = None
        headlength = 5
        headwidth = 8

        if bias == 0:
            n_scale = 12
            key = 2
            # set the n_scale to define vector length, and key for sample vector in m/s
        else:
            n_scale = 3
            key = 0.5

    if area == 'SA_swio':
        headlength = 5
        headwidth = 3
        n_sclice = 8

        if bias == 0:
            n_scale = 3
            key = 10
        else:
            n_scale = 0.3
            key = 1

    quiver_slices = slice(None, None, n_sclice)
    quiver_kwargs = {'headlength': headlength,
                     'headwidth': headwidth,
                     'angles': 'uv', 'units': 'xy',
                     'scale': n_scale}
    # a smaller scale parameter makes the arrow longer.

    circulation = ax.quiver(lon.values[quiver_slices],
                            lat.values[quiver_slices],
                            u.values[quiver_slices, quiver_slices],
                            v.values[quiver_slices, quiver_slices],
                            linewidth=1.01, edgecolors='k',
                            # linewidths is only for controlling the outline thickness,
                            # when an outline of a different color is explicitly requested.
                            # it looks like you have to explicitly set the edgecolors kwarg
                            # to get what you want now
                            color='blue', zorder=2, **quiver_kwargs)

    ax.quiverkey(circulation, 0.08, 0.90, key, f'{key:g}' + r'$ {m}/{s}$',
                 labelpos='E',
                 coordinates='axes')
    # ----------------------------- end of plot wind -----------------------------


def plot_field_in_classif(field: xr.DataArray, classif: pd.DataFrame,
                          area: str, vmax, vmin,
                          bias: bool = 1,
                          plot_wind: bool = 0,
                          only_significant_points: bool = 0,
                          suptitle_add_word: str = ''):
    """
    to plot field in class.
    :type only_significant_points: object
    :param only_significant_points:
    :type only_significant_points: object
    :param field:
    :type field: xarray.core.dataarray.DataArray
    :param classif:
    :type classif: pandas.core.frame.DataFrame
    :param area:
    :type area: str
    :param vmax:
    :type vmax: int
    :param vmin:
    :type vmin: int
    :param plot_wind:
    :type plot_wind: int
    :param bias:
    :type bias: int
    :param suptitle_add_word:
    :type suptitle_add_word:
    :return:
    :rtype: None
    """

    # ----------------------------- data -----------------------------
    class_mean = get_data_in_classif(da=field, df=classif,
                                     time_mean=True,
                                     significant=only_significant_points)
    print(f'good')

    # ----------------------------- plot -----------------------------

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex='row', sharey='col',
                            figsize=(12, 15), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.9, wspace=0.09, hspace=0.01)
    # axs = axs.flatten()
    axs = axs.ravel()

    for c in range(len(class_mean['class'])):
        cls = class_mean['class'].values[c]
        print(f'plot in class {cls:g} ...')

        ax = set_active_axis(axs=axs, n=c)
        set_basemap(area=area, ax=ax)

        plt.title('#' + str(int(cls)), fontsize=14, pad=3)

        cmap, norm = set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, bias=bias)

        cf = plt.contourf(field.lon, field.lat, class_mean[:, :, c], levels=norm.boundaries,
                          cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')

    # ----------------------------- end of plot -----------------------------
    cb_ax = fig.add_axes([0.13, 0.1, 0.7, 0.015])
    cbar_label = f'{field.assign_attrs().long_name:s} ({field.assign_attrs().units:s})'
    plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label, cax=cb_ax)

    # ----------------------------- surface wind -----------------------------
    if plot_wind:
        print(f'plot surface wind ...')

        print(f'loading data ... ')

        local_data = '/Users/ctang/local_data/era5'
        u = read_to_standard_da(f'{local_data:s}/u10/u10.hourly.1999-2016.swio.day.nc', 'u10')
        v = read_to_standard_da(f'{local_data:s}/v10/v10.hourly.1999-2016.swio.day.nc', 'v10')

        u = anomaly_daily(u)
        v = anomaly_daily(v)

        u = get_data_in_classif(u, classif, significant=False, time_mean=True)
        v = get_data_in_classif(v, classif, significant=False, time_mean=True)

        # speed = np.sqrt(u10 ** 2 + v10 ** 2)
        # speed = speed.rename('10m_wind_speed').assign_coords({'units': u10.attrs['units']})

        # Set up parameters for quiver plot. The slices below are used to subset the data (here
        # taking every 4th point in x and y). The quiver_kwargs are parameters to control the
        # appearance of the quiver so that they stay consistent between the calls.

        if area == 'bigreu':
            n_sclice = None
            headlength = 5
            headwidth = 8

            if bias == 0:
                n_scale = 3
                key = 10
            else:
                n_scale = 3
                key = 0.5

        if area == 'SA_swio':
            headlength = 5
            headwidth = 3
            n_sclice = 8

            if bias == 0:
                n_scale = 3
                key = 10
            else:
                n_scale = 0.3
                key = 1

        quiver_slices = slice(None, None, n_sclice)
        quiver_kwargs = {'headlength': headlength,
                         'headwidth': headwidth,
                         'angles': 'uv', 'units': 'xy',
                         'scale': n_scale}
        # a smaller scale parameter makes the arrow longer.

        # plot in subplot:
        for c in range(len(class_mean['class'])):
            cls = class_mean['class'].values[c]
            print(f'plot wind in class {cls:g} ...')

            ax = set_active_axis(axs=axs, n=c)

            u_1 = u[:, :, c]
            v_1 = v[:, :, c]

            circulation = ax.quiver(u_1.lon.values[quiver_slices],
                                    u_1.lat.values[quiver_slices],
                                    u_1.values[quiver_slices, quiver_slices],
                                    v_1.values[quiver_slices, quiver_slices],
                                    color='blue', zorder=2, **quiver_kwargs)

            ax.quiverkey(circulation, 0.08, 0.90, key, f'{key:g}' + r'$ {m}/{s}$',
                         labelpos='E',
                         coordinates='axes')

        # ----------------------------- end of plot -----------------------------
        suptitle_add_word += ' (surface wind)'

    title = f'{field.assign_attrs().long_name:s} in class'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)
    plt.savefig(f'plot/{field.name:s}_{area:s}_sig{only_significant_points:g}_wind{plot_wind:g}_classif.png', dpi=220)
    plt.show()
    print(f'got plot')


def plot_ttt_regimes(olr_regimes: pd.DataFrame, olr: xr.DataArray,
                     only_significant_points: int = 0):
    """
    plot ttt phase by olr
    :param only_significant_points:
    :param olr_regimes:
    :param olr:
    :return:
    """
    # ----------------------------- use the regime in sarah-e period -----------------------------

    year_min = olr.indexes['time'].year.min()
    year_max = olr.indexes['time'].year.max()
    olr_regimes = olr_regimes[np.logical_and(
        olr_regimes.index.year > year_min - 1,
        olr_regimes.index.year < year_max + 1)]

    month = 'NDJF'

    print(f'anomaly ...')

    olr_anomaly = olr.groupby(olr.time.dt.strftime('%m-%d')) - \
                  olr.groupby(olr.time.dt.strftime('%m-%d')).mean('time')

    olr_anomaly = olr_anomaly.assign_attrs(
        {'units': olr.assign_attrs().units, 'long_name': olr.assign_attrs().long_name})

    # the regime is defined by ERA5 ensemble data (B. P.),  but for 18h UTC
    # shift time by +1 day to match ensemble data timestamp
    olr_anomaly = convert_da_shifttime(olr_anomaly, second=-3600 * 18)
    olr = convert_da_shifttime(olr, second=-3600 * 18)

    # ----------------------------- fig config -----------------------------
    fig, axs = plt.subplots(nrows=4, ncols=2, sharex='row', sharey='col',
                            figsize=(8, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    axs = axs.ravel()

    for regime in [1, 2, 3, 4, 5, 6, 7]:
        print(f'plot regime = {regime:g}')
        # ----------------------------- calculate mean in each phase -----------------------------
        date_phase_one: pd.DatetimeIndex = olr_regimes.loc[olr_regimes['class'] == regime].index
        if len(date_phase_one) < 1:
            print(f'Sorry, I got 0 day in phase = {regime:g}')
            print(olr_regimes)
            break

        anomaly_olr_1phase: xr.DataArray = olr_anomaly.sel(time=date_phase_one)  # filter
        # if there's a error: check
        # 1) if data_phase_one is empty
        # 2) if the Time is 00:00:00
        olr_1phase: xr.DataArray = olr.sel(time=date_phase_one)  # filter

        # nday = anomaly_olr_1phase.shape[0]
        anomaly_mean: xr.DataArray = anomaly_olr_1phase.mean(axis=0)
        olr_mean = olr_1phase.mean(axis=0)

        # anomaly_ccub = sio.loadmat('./src/regime_maps.mat')['regimes_maps']
        # if not (anomaly_mean - anomaly_ccub[:, :, regime - 1]).max():
        #     print('the same')

        if only_significant_points:
            sig_map: xr.DataArray = value_mjo_significant_map(phase=regime, grid=anomaly_mean, month=month)
            # olr_mean = filter_2d_by_mask(olr_mean, mask=sig_map)
            anomaly_mean = filter_2d_by_mask(anomaly_mean, mask=sig_map)

        ax = set_active_axis(axs=axs, n=regime - 1)
        set_basemap(area='SA_swio', ax=ax)

        # ----------------------------- start to plot -----------------------------
        plt.title('#' + str(regime) + '/' + str(7), pad=3)

        vmax = 70
        vmin = -70

        lon, lat = np.meshgrid(anomaly_mean.lon, anomaly_mean.lat)
        level_anomaly = np.arange(vmin, vmax + 1, 10)
        cf1 = plt.contourf(lon, lat, anomaly_mean, level_anomaly, cmap='PuOr_r', vmax=vmax, vmin=vmin)
        level_olr = np.arange(140, 280, 20)
        cf2 = plt.contour(lon, lat, olr_mean, level_olr, cmap='magma_r', vmax=280, vmin=140)
        ax.clabel(cf2, level_olr, inline=True, fmt='%2d', fontsize='xx-small')
        ax.coastlines()

        # cf = plt.pcolormesh(lon, lat, consistency_percentage_map,
        #                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

        # ----------------------------- end of plot -----------------------------

        ax.text(0.9, 0.95, f'{month:s}', fontsize=12,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.8, 0.1, f'{olr_anomaly.name:s}', fontsize=12,
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

        test = 1
        if test:
            print(f'save netcdf file @regimes to check')
            file = f'./ttt.regime.{regime}.nc'
            anomaly_mean.to_netcdf(file)

    cbar_label = f'OLR ({olr_anomaly.assign_attrs().units:s})'
    plt.colorbar(cf1, ticks=np.ndarray.tolist(level_anomaly), label=cbar_label, ax=axs)

    title = f'olr regimes'
    plt.suptitle(title)

    # tag: specify the location of the cbar
    # cb_ax = fig.add_axes([0.13, 0.1, 0.7, 0.015])
    # cb = plt.colorbar(cf1, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label, cax=cb_ax)

    plt.savefig(f'./plot/ttt_regimes_sig_{only_significant_points:g}.png', dpi=220)

    plt.show()
    print(f'got plot')


def plot_color_matrix(df: pd.DataFrame, ax, cbar_label: str, plot_number: bool = False):
    """
    plot matrix by df, where x is column, y is index,
    :param plot_number:
    :type plot_number:
    :param cbar_label:
    :type cbar_label: str
    :param df:
    :type df:
    :param ax:
    :type ax:
    :return:
    :rtype: ax
    """

    import math

    c = ax.pcolor(df, cmap=plt.cm.get_cmap('Blues', df.max().max() + 1))

    x_ticks_label = df.columns
    y_ticks_label = df.index

    # put the major ticks at the middle of each cell
    x_ticks = np.arange(df.shape[1]) + 0.5
    y_ticks = np.arange(df.shape[0]) + 0.5
    ax.set_xticks(x_ticks, minor=False)
    ax.set_yticks(y_ticks, minor=False)

    ax.set_xticklabels(x_ticks_label, minor=False)
    ax.set_yticklabels(y_ticks_label, minor=False)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)

    # ax.set_ylabel('year')
    # ax.set_xlabel('month')

    vmin = int(df.values.min())
    vmax = int(df.values.max())

    # print(vmin, vmax)

    if vmin + vmax < vmax:
        c = ax.pcolor(df, cmap=plt.cm.get_cmap('coolwarm', df.max().max() + 1))
        cbar_ticks = [x for x in range(vmin, vmax + 1, math.ceil((vmax - vmin) / 10))]
    else:
        cbar_ticks = [x for x in range(vmin, vmax, math.ceil((vmax - vmin) / 10))]

    if plot_number:
        for i in range(df.shape[1]):  # x direction
            for j in range(df.shape[0]):  # y direction
                c = df.iloc[j, i]
                # notice to the order of
                ax.text(x_ticks[i], y_ticks[j], f'{c:2.0f}', va='center', ha='center')
        # put cbar label
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(cbar_label)
    else:
        cb = plt.colorbar(c, ax=ax, label=cbar_label, ticks=cbar_ticks)
        loc = [x + 0.5 for x in cbar_ticks]
        cb.set_ticks(loc)
        cb.set_ticklabels(cbar_ticks)

    return ax


def find_symmetric_difference(list1, list2):
    """
    show difference
    :param list1:
    :type list1:
    :param list2:
    :type list2:
    :return:
    :rtype:
    """

    difference = set(list1).symmetric_difference(set(list2))
    list_difference = list(difference)

    return list_difference


def plot_matrix_class_vs_class(class_x: pd.DataFrame,
                               class_y: pd.DataFrame,
                               output_plot: str = 'class_vs_class_matrix',
                               occurrence: bool = 1,
                               suptitle_add_word: str = ""):
    """
    plot the matrix of class vs class, color bar is number of points
    class_df: DataFrame of one columns of classifications with DateTimeIndex, columns' names will be used.
    $$: if plot occurrence, impact of class_x on class_y

    :param class_y:
    :type class_y: pandas.core.frame.DataFrame
    :param class_x:
    :type class_x: pandas.core.frame.DataFrame
    :param output_plot:
    :type output_plot: str
    :param occurrence: if occurrence is True, will plot numbers in the matrix by default
    :type occurrence:
    :param suptitle_add_word:
    :type suptitle_add_word: str
    :return:
    :rtype: None
    """

    # the input DataFrames may have different index, so merge two classes with DataTimeIndex:
    class_df = class_x.merge(class_y, left_index=True, right_index=True)

    # y direction:
    name_y = class_df.columns[0]
    # x direction:
    name_x = class_df.columns[1]

    class_name_y = list(set(class_df.iloc[:, 0]))
    class_name_x = list(set(class_df.iloc[:, 1]))

    # get cross matrix
    cross = np.zeros((len(class_name_y), len(class_name_x)))
    for i in range(len(class_name_y)):
        class_one = class_df.loc[class_df[name_y] == class_name_y[i]]
        for j in range(len(class_name_x)):
            class_cross = class_one.loc[class_one[name_x] == class_name_x[j]]
            cross[i, j] = len(class_cross)
    cross_df = pd.DataFrame(data=cross, index=class_name_y, columns=class_name_x).astype(int)

    # ----------------------------- plot -----------------------------
    fig = plt.figure()
    widths = [1, 3]
    heights = [1, 2]
    gridspec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
    gridspec.update(wspace=0.1, hspace=0.2)  # set the spacing between axes.

    # matrix:
    ax = fig.add_subplot(gridspec[1, 1])
    cbar_label = 'count'
    plot_number = False

    if occurrence:
        print(f'occurrence: {name_y:s} introduced changes of {name_x:s} occurrence')
        occ = cross_df
        for i in range(len(class_name_x)):
            ssr_class = class_name_x[i]
            avg_freq = len(class_df[class_df[name_x] == ssr_class]) / len(class_df)
            print(len(class_df[class_df[name_x] == ssr_class]), len(class_df), avg_freq)

            for j in range(len(class_name_y)):
                large_class = class_name_y[j]
                freq = cross[j, i] / len(class_df[class_df[name_y] == large_class])
                occ.iloc[j, i] = (freq - avg_freq) * 100
                print(i, j, cross_df.iloc[j, i], len(class_df[class_df[name_y] == large_class]), freq)

        plot_number = True  # it's better to plot number with occurrence

        cross_df = occ

        cbar_label = 'occurrence (%)'

    plot_color_matrix(df=cross_df, ax=ax, cbar_label=cbar_label, plot_number=plot_number)

    ax.set_xlabel(name_x)

    # histogram in x direction:
    ax = fig.add_subplot(gridspec[0, 1])
    bars = class_name_x
    data = class_df[name_x]
    height = [len(data[data == x]) for x in class_name_x]

    y_pos = np.arange(len(bars))
    ax.bar(bars, height, align='center', color='red')

    ax.set_xlim(0.5, y_pos[-1] + 1.5)  # these limit is from test
    # x_ticks = np.arange(len(class_name_x)) + 0.5

    ax.set_xticks([], minor=False)

    ax.set_xticklabels([], minor=False)
    ax.set_ylabel('n_day')
    # ax.set_xlabel(name_x)

    # histogram in y direction:
    ax = fig.add_subplot(gridspec[1, 0])
    bars = class_name_y
    data = class_df[name_y]
    height = [len(data[data == x]) for x in class_name_y]

    y_pos = np.arange(len(bars))
    ax.barh(bars, height, align='center', color='orange')

    ax.set_ylim(0.5, y_pos[-1] + 1.5)
    # these limit is from test
    ax.invert_xaxis()
    ax.set_xlabel('n_day')
    ax.set_ylabel(name_y)

    # end of plotting:
    title = f'{name_x:s} vs {name_y:s}'
    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)

    plt.savefig(output_plot, dpi=300)

    plt.show()

    print(f'job done')


def plot_matrix_classification_at_year_and_month(class_df: pd.DataFrame, output_plot: str):
    """
    calculate classification at different month
    class_df: DataFrame of one class with DateTimeIndex
    """

    # get info from input:
    n_class = int(class_df.max())

    year_start = class_df.index.year.min()
    year_end = class_df.index.year.max()
    n_year = year_end - year_start + 1

    month_list = list(set(class_df.index.month))

    for i in range(n_class):
        class_1 = class_df.loc[class_df.values == i + 1]
        cross = np.zeros((n_year, len(month_list)))

        for y in range(n_year):
            for im in range(len(month_list)):
                cross[y, im] = class_1.loc[
                    (class_1.index.year == y + year_start) &
                    (class_1.index.month == month_list[im])].__len__()

        print(f'# ----------------- {n_class:g} -> {i + 1:g} -----------------------------')

        df = pd.DataFrame(data=cross, index=range(year_start, year_start + n_year), columns=month_list)
        df = df.astype(int)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w', edgecolor='k', dpi=300)

        plot_color_matrix(df=df, ax=ax, cbar_label='count')

        plt.suptitle(f'C{n_class:g} -> C{i + 1:g}')
        plt.savefig(f'./plot/{output_plot:s}', dpi=220)
        plt.show()

    print(f'job done')


def plot_12months_geo_map_significant(da: xr.DataArray, area: str, sig_dim: str, only_sig_point: bool):
    fig, axs = plt.subplots(nrows=4, ncols=3, sharex='row', sharey='col',
                            figsize=(14, 12), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    axs = axs.ravel()

    print(sig_dim)

    for imon in range(12):
        print(f'plot month = {imon + 1:g}')
        ax = set_active_axis(axs=axs, n=imon)

        # get data:
        month_ly = da.sel(month=imon + 1)

        if only_sig_point:
            sig_map: xr.DataArray = value_significant_of_anomaly_2d_mask(field_3d=month_ly, conf_level=0.05)
            print(f'only work for anomaly/change/trend/ !!! compare to ZERO!!')
            month_to_plot = filter_2d_by_mask(month_ly, mask=sig_map).mean(axis=0)
        else:
            month_to_plot = month_ly.mean(axis=0)

        # ----------------------------- plot -----------------------------
        set_basemap(area=area, ax=ax)

        lon, lat = np.meshgrid(month_to_plot.longitude, month_to_plot.latitude)

        vmax, vmin = value_cbar_max_min_of_da(month_to_plot)

        # TODO:
        # vmax = 20
        # vmin = -20
        level_anomaly = np.arange(vmin, vmax + 1, 2)

        cf1 = plt.contourf(lon, lat, month_to_plot, level_anomaly, cmap='PuOr_r', vmax=vmax, vmin=vmin)

        # ----------------------------- end of plot -----------------------------

        ax.text(0.9, 0.95, f'{calendar.month_abbr[imon + 1]:s}', fontsize=12,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.9, 0.1, f'{da.name:s}', fontsize=12,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        cbar_label = f'({da.assign_attrs().units:s})'
        plt.colorbar(cf1, ticks=np.ndarray.tolist(level_anomaly), label=cbar_label, ax=ax)

    return fig, axs


def select_area_from_str(da: xr.DataArray, area: str):
    lonlat = value_lonlatbox_from_area(area)
    da1 = da.where(np.logical_and(da.longitude > lonlat[0], da.longitude < lonlat[1]), drop=True)
    da2 = da1.where(np.logical_and(da1.latitude > lonlat[2], da1.latitude < lonlat[3]), drop=True)

    return da2


def plot_mjo_phase(mjo_phase: pd.DataFrame, olr: xr.DataArray, high_amplitude: bool, month: str,
                   only_significant_points: int = 0):
    """
    plot mjo phase by olr
    :param only_significant_points:
    :param month:
    :param mjo_phase:
    :param high_amplitude: if plot high amplitude > 1
    :param olr:
    :return:
    """
    # ----------------------------- prepare the data -----------------------------
    if high_amplitude:
        filtering = f'amplitude > 1'
        mjo_phase = data_filter_by_key_limit_value(data=mjo_phase, key='amplitude', how='gt', value=1)

    # ----------------------------- filtering data by season -----------------------------
    olr = filter_xr_by_month(data=olr, month=month)
    olr_daily_anomaly = anomaly_daily(olr)

    olr_daily_anomaly.to_netcdf(f'./mjo.anomaly.test.nc')

    mjo_phase = filter_df_by_month(data=mjo_phase, month=month)

    # ----------------------------- some predefined values of cbar limits -----------------------------

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex='row', sharey='col',
                            figsize=(8, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    # axs = axs.flatten()
    axs = axs.ravel()

    for phase in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(f'plot class = {phase:g}')
        # ----------------------------- calculate mean in each phase -----------------------------
        date_phase_one: pd.DatetimeIndex = mjo_phase.loc[mjo_phase['phase'] == phase].index
        if len(date_phase_one) < 1:
            print(f'Sorry, I got 0 day in phase = {phase:g}')
            print(mjo_phase)
            break
        anomaly_olr_1phase: xr.DataArray = olr_daily_anomaly.sel(time=date_phase_one)  # filter
        # if there's a error: check
        # 1) if data_phase_one is empty
        # 2) if the Time is 00:00:00
        olr_1phase: xr.DataArray = olr.sel(time=date_phase_one)  # filter

        # nday = anomaly_olr_1phase.shape[0]
        anomaly_mean: xr.DataArray = anomaly_olr_1phase.mean(axis=0)
        olr_mean = olr_1phase.mean(axis=0)

        if only_significant_points:
            sig_map: xr.DataArray = value_mjo_significant_map(phase=phase, grid=anomaly_mean, month=month)
            # olr_mean = filter_2d_by_mask(olr_mean, mask=sig_map)
            anomaly_mean = filter_2d_by_mask(anomaly_mean, mask=sig_map)
            # to fix type

        ax = set_active_axis(axs=axs, n=phase - 1)
        set_basemap(area='swio', ax=ax)
        # set_basemap(area='SA_swio', ax=ax)

        # ----------------------------- start to plot -----------------------------
        plt.title('#' + str(phase) + '/' + str(8), pad=3)

        lon, lat = np.meshgrid(anomaly_mean.longitude, anomaly_mean.latitude)
        level_anomaly = np.arange(-50, 51, 5)
        cf1 = plt.contourf(lon, lat, anomaly_mean, level_anomaly, cmap='PuOr_r', vmax=50, vmin=-50)
        level_olr = np.arange(140, 280, 20)
        cf2 = plt.contour(lon, lat, olr_mean, level_olr, cmap='magma_r', vmax=280, vmin=140)
        ax.clabel(cf2, level_olr, inline=True, fmt='%2d', fontsize='xx-small')
        ax.coastlines()

        # cf = plt.pcolormesh(lon, lat, consistency_percentage_map,
        #                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

        # ----------------------------- end of plot -----------------------------

        ax.text(0.9, 0.95, f'{month:s}', fontsize=12,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.8, 0.1, f'{olr.name:s}', fontsize=12,
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    cbar_label = f'OLR ({olr.assign_attrs().units:s})'
    plt.colorbar(cf1, ticks=np.ndarray.tolist(level_anomaly), label=cbar_label, ax=axs)

    title = f'MJO phase with {filtering:s} in {month:s}'
    plt.suptitle(title)

    # tag: specify the location of the cbar
    # cb_ax = fig.add_axes([0.13, 0.1, 0.7, 0.015])
    # cb = plt.colorbar(cf1, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label, cax=cb_ax)

    plt.savefig(f'./mjo_phases_sig_{only_significant_points:g}.png', dpi=220)

    plt.show()
    print(f'got plot')


def plot_hourly_curve_by_month(df: pd.DataFrame, columns: list, suptitle=' ', months=None):
    """
    plot hourly curves by /month/ for the columns in list
    :param months:
    :param suptitle:
    :param df:
    :param columns:
    :return:
    """

    if months is None:  # 👍
        months = [11, 12, 1, 2, 3, 4]

    # ----------------------------- set parameters -----------------------------
    # months = [11, 12, 1, 2, 3, 4]
    colors = ['black', 'green', 'orange', 'red']
    data_sources = columns

    # ----------------------------- set fig -----------------------------
    nrows = len(months)

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(16, 16),
                            facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    axs = axs.ravel()
    # ----------------------------- plotting -----------------------------

    for v in range(len(columns)):
        for i in range(nrows):
            plt.sca(axs[i])  # active this subplot

            month = months[i]
            data_slice = df[df.index.month == month]
            x = range(len(data_slice))

            label = data_sources[v]

            plt.plot(x, data_slice[[columns[v]]], color=colors[v], label=label)

            print(f'month = {months[i]:g}, var = {columns[v]:s}')

            # ----------------------------- format of fig -----------------------------
            # if input data is hourly, reform the fig axis
            nday = len(set(data_slice.index.day))

            if len(data_slice) > 31:
                custom_ticks = range(11, len(data_slice), 24)
                custom_ticks_labels = range(1, nday + 1)
            else:
                custom_ticks = x
                custom_ticks_labels = [y + 1 for y in custom_ticks]

            axs[i].set_xticks(custom_ticks)
            axs[i].set_xticklabels(custom_ticks_labels)

            axs[i].set_xlim(0, len(data_slice) * 1.2)

            # axs[i].xaxis.set_ticks_position('top')
            # axs[i].xaxis.set_ticks_position('bottom')

            plt.legend(loc='upper right', fontsize=8)
            plt.xlabel(f'day')
            plt.ylabel(r'$SSR\ (W/m^2)$')
            axs[i].text(0.5, 0.95, data_slice.index[0].month_name(), fontsize=20,
                        horizontalalignment='right', verticalalignment='top', transform=axs[i].transAxes)

    plt.suptitle(suptitle)
    plt.show()
    print(f'got the plot')


def get_T_value(conf_level: float = 0.05, dof: int = 10):
    """
    get value of T
    two tail = 0.95:
    :param conf_level:
    :param dof:
    :return:
    """

    print(conf_level)

    T_value = [
        12.71, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
        2.040, 2.037, 2.035, 2.032, 2.030, 2.028, 2.026, 2.024, 2.023, 2.021,
        2.020, 2.018, 2.017, 2.015, 2.014, 2.013, 2.012, 2.011, 2.010, 2.009,
        2.008, 2.007, 2.006, 2.005, 2.004, 2.003, 2.002, 2.002, 2.001, 2.000,
        2.000, 1.999, 1.998, 1.998, 1.997, 1.997, 1.996, 1.995, 1.995, 1.994,
        1.994, 1.993, 1.993, 1.993, 1.992, 1.992, 1.991, 1.991, 1.990, 1.990,
        1.990, 1.989, 1.989, 1.989, 1.988, 1.988, 1.988, 1.987, 1.987, 1.987,
        1.986, 1.986, 1.986, 1.986, 1.985, 1.985, 1.985, 1.984, 1.984, 1.984]

    # infinity:
    if dof > 100:
        return 1.960
    else:
        return T_value[dof - 1]


# ===================================================
# one tail t test table:
# dof       0.90    0.95   0.975    0.99   0.995   0.999
# 1.       3.078   6.314  12.706  31.821  63.657 318.313
# 2.       1.886   2.920   4.303   6.965   9.925  22.327
# 3.       1.638   2.353   3.182   4.541   5.841  10.215
# 4.       1.533   2.132   2.776   3.747   4.604   7.173
# 5.       1.476   2.015   2.571   3.365   4.032   5.893
# 6.       1.440   1.943   2.447   3.143   3.707   5.208
# 7.       1.415   1.895   2.365   2.998   3.499   4.782
# 8.       1.397   1.860   2.306   2.896   3.355   4.499
# 9.       1.383   1.833   2.262   2.821   3.250   4.296
# 10.       1.372   1.812   2.228   2.764   3.169   4.143
# 11.       1.363   1.796   2.201   2.718   3.106   4.024
# 12.       1.356   1.782   2.179   2.681   3.055   3.929
# 13.       1.350   1.771   2.160   2.650   3.012   3.852
# 14.       1.345   1.761   2.145   2.624   2.977   3.787
# 15.       1.341   1.753   2.131   2.602   2.947   3.733
# 16.       1.337   1.746   2.120   2.583   2.921   3.686
# 17.       1.333   1.740   2.110   2.567   2.898   3.646
# 18.       1.330   1.734   2.101   2.552   2.878   3.610
# 19.       1.328   1.729   2.093   2.539   2.861   3.579
# 20.       1.325   1.725   2.086   2.528   2.845   3.552
# 21.       1.323   1.721   2.080   2.518   2.831   3.527
# 22.       1.321   1.717   2.074   2.508   2.819   3.505
# 23.       1.319   1.714   2.069   2.500   2.807   3.485
# 24.       1.318   1.711   2.064   2.492   2.797   3.467
# 25.       1.316   1.708   2.060   2.485   2.787   3.450
# 26.       1.315   1.706   2.056   2.479   2.779   3.435
# 27.       1.314   1.703   2.052   2.473   2.771   3.421
# 28.       1.313   1.701   2.048   2.467   2.763   3.408
# 29.       1.311   1.699   2.045   2.462   2.756   3.396
# 30.       1.310   1.697   2.042   2.457   2.750   3.385
# 31.       1.309   1.696   2.040   2.453   2.744   3.375
# 32.       1.309   1.694   2.037   2.449   2.738   3.365
# 33.       1.308   1.692   2.035   2.445   2.733   3.356
# 34.       1.307   1.691   2.032   2.441   2.728   3.348
# 35.       1.306   1.690   2.030   2.438   2.724   3.340
# 36.       1.306   1.688   2.028   2.434   2.719   3.333
# 37.       1.305   1.687   2.026   2.431   2.715   3.326
# 38.       1.304   1.686   2.024   2.429   2.712   3.319
# 39.       1.304   1.685   2.023   2.426   2.708   3.313
# 40.       1.303   1.684   2.021   2.423   2.704   3.307
# 41.       1.303   1.683   2.020   2.421   2.701   3.301
# 42.       1.302   1.682   2.018   2.418   2.698   3.296
# 43.       1.302   1.681   2.017   2.416   2.695   3.291
# 44.       1.301   1.680   2.015   2.414   2.692   3.286
# 45.       1.301   1.679   2.014   2.412   2.690   3.281
# 46.       1.300   1.679   2.013   2.410   2.687   3.277
# 47.       1.300   1.678   2.012   2.408   2.685   3.273
# 48.       1.299   1.677   2.011   2.407   2.682   3.269
# 49.       1.299   1.677   2.010   2.405   2.680   3.265
# 50.       1.299   1.676   2.009   2.403   2.678   3.261
# 51.       1.298   1.675   2.008   2.402   2.676   3.258
# 52.       1.298   1.675   2.007   2.400   2.674   3.255
# 53.       1.298   1.674   2.006   2.399   2.672   3.251
# 54.       1.297   1.674   2.005   2.397   2.670   3.248
# 55.       1.297   1.673   2.004   2.396   2.668   3.245
# 56.       1.297   1.673   2.003   2.395   2.667   3.242
# 57.       1.297   1.672   2.002   2.394   2.665   3.239
# 58.       1.296   1.672   2.002   2.392   2.663   3.237
# 59.       1.296   1.671   2.001   2.391   2.662   3.234
# 60.       1.296   1.671   2.000   2.390   2.660   3.232
# 61.       1.296   1.670   2.000   2.389   2.659   3.229
# 62.       1.295   1.670   1.999   2.388   2.657   3.227
# 63.       1.295   1.669   1.998   2.387   2.656   3.225
# 64.       1.295   1.669   1.998   2.386   2.655   3.223
# 65.       1.295   1.669   1.997   2.385   2.654   3.220
# 66.       1.295   1.668   1.997   2.384   2.652   3.218
# 67.       1.294   1.668   1.996   2.383   2.651   3.216
# 68.       1.294   1.668   1.995   2.382   2.650   3.214
# 69.       1.294   1.667   1.995   2.382   2.649   3.213
# 70.       1.294   1.667   1.994   2.381   2.648   3.211
# 71.       1.294   1.667   1.994   2.380   2.647   3.209
# 72.       1.293   1.666   1.993   2.379   2.646   3.207
# 73.       1.293   1.666   1.993   2.379   2.645   3.206
# 74.       1.293   1.666   1.993   2.378   2.644   3.204
# 75.       1.293   1.665   1.992   2.377   2.643   3.202
# 76.       1.293   1.665   1.992   2.376   2.642   3.201
# 77.       1.293   1.665   1.991   2.376   2.641   3.199
# 78.       1.292   1.665   1.991   2.375   2.640   3.198
# 79.       1.292   1.664   1.990   2.374   2.640   3.197
# 80.       1.292   1.664   1.990   2.374   2.639   3.195
# 81.       1.292   1.664   1.990   2.373   2.638   3.194
# 82.       1.292   1.664   1.989   2.373   2.637   3.193
# 83.       1.292   1.663   1.989   2.372   2.636   3.191
# 84.       1.292   1.663   1.989   2.372   2.636   3.190
# 85.       1.292   1.663   1.988   2.371   2.635   3.189
# 86.       1.291   1.663   1.988   2.370   2.634   3.188
# 87.       1.291   1.663   1.988   2.370   2.634   3.187
# 88.       1.291   1.662   1.987   2.369   2.633   3.185
# 89.       1.291   1.662   1.987   2.369   2.632   3.184
# 90.       1.291   1.662   1.987   2.368   2.632   3.183
# 91.       1.291   1.662   1.986   2.368   2.631   3.182
# 92.       1.291   1.662   1.986   2.368   2.630   3.181
# 93.       1.291   1.661   1.986   2.367   2.630   3.180
# 94.       1.291   1.661   1.986   2.367   2.629   3.179
# 95.       1.291   1.661   1.985   2.366   2.629   3.178
# 96.       1.290   1.661   1.985   2.366   2.628   3.177
# 97.       1.290   1.661   1.985   2.365   2.627   3.176
# 98.       1.290   1.661   1.984   2.365   2.627   3.175
# 99.       1.290   1.660   1.984   2.365   2.626   3.175
# 100.       1.290   1.660   1.984   2.364   2.626   3.174
# infinity   1.282   1.645   1.960   2.326   2.576   3.090


def value_mjo_significant_map(phase: int, grid: xr.DataArray = 0, month: str = 0) -> xr.DataArray:
    """
    calculate significant map of mjo phase, depend on the input olr data from era5 analysis
    ONLY in the swio area
    :param month: like JJA and DJF, etc
    :param grid: output sig_map remapped to gird. if grid = 0 , no interp
    :param phase:
    :return:
    """
    mjo_phase: pd.DataFrame = read_mjo()

    # ----------------------------- read necessary data: era5 ttr reanalysis data
    # ttr_swio = xr.open_dataset(f'~/local_data/era5/ttr.era5.1999-2016.day.swio.nc')['ttr']
    ttr_swio = xr.open_dataset(f'~/local_data/era5/ttr.era5.1999-2016.day.reu.nc')['ttr']

    if isinstance(month, str):
        ttr_swio = filter_xr_by_month(ttr_swio, month=month)
        mjo_phase: pd.DataFrame = filter_df_by_month(mjo_phase, month=month)

    # ----------------------------- anomaly OLR -----------------------------
    olr_swio = convert_ttr_era5_2_olr(ttr=ttr_swio, is_reanalysis=True)
    olr_swio_anomaly = anomaly_daily(olr_swio)

    # select phase:
    date_index: pd.DatetimeIndex = mjo_phase.loc[mjo_phase['phase'] == phase].index
    olr_swio_anomaly_1phase: xr.DataArray = olr_swio_anomaly.sel(time=date_index)  # tag: filter

    # ----------------------------- calculate sig_map -----------------------------
    print(f'calculating significant map, dims={str(olr_swio_anomaly_1phase.shape):s}, waiting ... ')
    sig_map_olr: xr.DataArray = value_significant_of_anomaly_2d_mask(field_3d=olr_swio_anomaly_1phase, conf_level=0.05)

    # to see if remap is necessary:
    if isinstance(grid, int):
        new_sig_map = sig_map_olr

    else:

        new_sig_map = np.zeros(grid.shape)
        old_lon = olr_swio.longitude
        old_lat = olr_swio.latitude

        new_lon = grid.longitude.values
        new_lat = grid.latitude.values

        # get closest lon:
        for lon in range(grid.longitude.size):
            new_lon[lon] = old_lon[np.abs(old_lon - new_lon[lon]).argmin()]
        # get closest lat:
        for lat in range(grid.latitude.size):
            new_lat[lat] = old_lat[np.abs(old_lat - new_lat[lat]).argmin()]

        for lat in range(grid.latitude.size):
            for lon in range(grid.longitude.size):
                new_sig_map[lat, lon] = sig_map_olr.loc[dict(latitude=new_lat[lat], longitude=new_lon[lon])].values

    # return sig map in 2D xr.DataArray:
    sig = xr.DataArray(new_sig_map.astype(bool), coords=[grid.latitude, grid.longitude], dims=grid.dims)

    return sig


def value_significant_of_anomaly_2d_mask(field_3d: xr.DataArray, conf_level: float = 0.05) -> xr.DataArray:
    """
    calculate 2d map of significant of values in true false
    :param conf_level: default = 0.05
    :param field_3d: have to be in (time, lat, lon)
    :return: 2d array of true false xr.DataArray
    """

    # there's another coord beside 'longitude' 'latitude', which is the dim of significant !!!
    sig_coord_name = [x for x in field_3d.dims if x not in ['lon', 'lat']][0]

    # tag, note: change order of dim:
    field_3d = field_3d.transpose(sig_coord_name, "lat", "lon")

    sig_2d = np.zeros((field_3d.shape[1], field_3d.shape[2]))

    print(f'get significant map...')
    for lat in range(field_3d.shape[1]):
        print(f'significant ----- {lat * 100 / len(field_3d.lat): 4.2f} % ...')
        for lon in range(field_3d.shape[2]):
            grid = field_3d[:, lat, lon]

            if field_3d.name == 'SIS':
                # select value not nan, removing the nan value
                grid_nonnan = grid[np.logical_not(np.isnan(grid))]
            else:
                # good point
                grid_nonnan = grid

            if len(grid_nonnan) < 1:
                print("----------------- bad point")
                sig_2d[lat, lon] = 0
            else:
                t_statistic, p_value_2side = stats.ttest_1samp(grid_nonnan, 0)
                sig_2d[lat, lon] = p_value_2side

            # print(sig_2d.shape, lat, lon)

    # option 2:
    # 根据定义，p值大小指原假设H0为真的情况下样本数据出现的概率。
    # 在实际应用中，如果p值小于0.05，表示H0为真的情况下样本数据出现的概率小于5 %，
    # 根据小概率原理，这样的小概率事件不可能发生，因此我们拒绝H0为真的假设
    sig_map = sig_2d < conf_level

    # return sig map in 2D xr.DataArray:
    sig_map_da = field_3d.mean(sig_coord_name)
    sig = xr.DataArray(sig_map.astype(bool), coords=[field_3d.lat, field_3d.lon], dims=sig_map_da.dims)

    # sig.plot()
    # plt.show()

    return sig


def value_remap_a_to_b(a: xr.DataArray, b: xr.DataArray):
    """
    remap a to b, by method='cubic'
    :param a:
    :param b:
    :return:
    """

    # remap cmsaf to mf:
    # if remap by only spatial dimensions:
    lon_name = get_time_lon_lat_name_from_da(b)['lon']
    lat_name = get_time_lon_lat_name_from_da(b)['lat']

    interpolated = a.interp(lon=b[lon_name], lat=b[lat_name])
    # @you, please make sure in the you give the right dimension names (such as lon and lat).
    # cmsaf and mf are both xr.DataArray

    # if remap by all the dimensions:
    # interpolated = a.interp_like(b)
    # that's the same as line below
    # interpolated = a.interp(lon=b.lon, lat=b.lat, time=b.time)
    # we are not usually do the remap in time dimension,
    # if you want to do this, make sure that there is no duplicate index

    return interpolated


def value_season_mean_ds(ds, time_calendar='standard'):
    """

    :param ds:
    :type ds:
    :param time_calendar:
    :type time_calendar:
    :return:
    :rtype:
    """

    print(f'calendar is {time_calendar:s}')
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby('time.season').sum(dim='time')


def value_map_corner_values_from_coverage(coverage: str):
    """
    get location of lon/lat box:
    :param coverage:
    :return:
    """

    lon_low = lon_up = lat_low = lat_up = -999

    if coverage == 'reunion':
        lon_low, lon_up, lat_low, lat_up = 54.75, 56.25, -21.75, -20.25
    if coverage == 'swio':
        lon_low, lon_up, lat_low, lat_up = 0, 120, -50, 10
    if coverage == 'SouthernAfrica':
        lon_low, lon_up, lat_low, lat_up = 0, 60, -40, 0

    return lon_low, lon_up, lat_low, lat_up


def value_replace_in_xr(data, dim_name: str, new_values: np.ndarray):
    """
    replace values in xr.DataArray or xr.DataSet, by array
    :param dim_name:
    :type dim_name:
    :param data:
    :param new_values:
    :return:
    """

    data[dim_name] = new_values

    return data


def convert_utc2local_da(test: bool, da):
    """
    convert utc time to local time
    :param test:
    :type test:
    :param da:
    :return:
    """

    time_local = []
    for i in range(da.time.size):
        utc = da.indexes['time'][i].replace(tzinfo=tz.tzutc())
        local_time = utc.astimezone(tz.tzlocal())
        local_time = local_time.replace(tzinfo=None)
        time_local.append(local_time)

        if test:
            print(utc, local_time)

    value_replace_in_xr(data=da, dim_name='time', new_values=np.array(time_local))

    return da


def value_consistency_sign_with_mean_in_percentage_2d(field_3d: xr.DataArray):
    """
    get a map of percentage: days of the same signs with statistics.
    :param field_3d:
    :return: percentage_data_array in format ndarray in (lat,lon)
    """
    # note: change order of dim:
    field_3d = field_3d.transpose("time", "latitude", "longitude")
    # note: mean of one specific dim:
    time_mean_field = field_3d.mean(dim='time')

    num_time = field_3d.time.shape[0]
    num_lon = time_mean_field.longitude.shape[0]
    num_lat = time_mean_field.latitude.shape[0]

    compare_field = np.zeros(field_3d.shape)

    percentage_2d_map = np.zeros((num_lat, num_lon))

    for t in range(num_time):
        compare_field[t, :, :] = time_mean_field * field_3d[t]

    for lat in range(num_lat):
        for lon in range(num_lon):
            time_series = compare_field[:, lat, lon]
            positive_count = len(list(filter(lambda x: (x >= 0), time_series)))

            percentage_2d_map[lat, lon] = positive_count / num_time * 100

    return percentage_2d_map


def set_basemap(ax: plt.Axes, area: str):
    """
    set basemap
    :param ax:
    :type ax: cartopy.mpl.geoaxes.GeoAxesSubplot
    :param area:
    :return:
    """
    area_name = area

    area = value_lonlatbox_from_area(area_name)
    ax.set_extent(area, crs=ccrs.PlateCarree())
    # lon_left, lon_right, lat_north, lat_north

    ax.coastlines('50m')
    ax.add_feature(cfeature.LAND.with_scale('10m'))


def set_active_axis(axs: np.ndarray, n: int):
    """
    active this axis of plot or subplot
    :param axs: nd array of subplots' axis
    :param n:
    :return:
    """

    ax = axs[n]
    plt.sca(axs[n])
    # active this subplot

    return ax


# noinspection PyUnresolvedReferences
def set_cbar(vmax, vmin, n_cbar, bias):
    """
    set for color bar
    :param n_cbar:
    :param vmin:
    :param vmax:
    :param bias:
    :return: cmap, norm
    """
    import matplotlib as mpl

    if bias == 1:
        cmap = plt.cm.coolwarm
        vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
        vmax = max(np.abs(vmin), np.abs(vmax))
    if bias == 0:
        cmap = plt.cm.YlOrRd

    # using the input of min and max, but make (max-min/2) in the middle
    if bias == 3:
        cmap = plt.cm.coolwarm

    bounds = np.linspace(vmin, vmax, n_cbar + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def tag_from_str(string: str):
    """
    get monthly tag, such as JJA/DJF from string
    :param string: str
    :return:
    """

    tag_dict = dict(

        summer='JJA',
        winter='DJF',
        austral_summer='DJF',
        austral_winter='JJA',

    )

    return tag_dict[string]


def print_data(data, dim: int = 2):
    """
    print all data
    :param dim:
    :type dim:
    :param data:
    :return:
    """

    if dim == 1:
        for i in range(data.shape[0]):
            print(f'# {i:g} \t')
            print(data[i])

    if dim == 2:

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                print(data[i, j])


def read_mjo(match_ssr_avail_day: bool = 0):
    """
    to read mjo phase
    :return: pd.DataFrame
    """
    mjo_sel = f'SELECT ' \
              f'ADDTIME(CONVERT(left(dt, 10), DATETIME), MAKETIME(HOUR(dt), ' \
              f'floor(MINUTE(dt) / 10)*10,0)) as DateTime, ' \
              f'AVG(GHI_020_Avg) as ssr, ' \
              f'MONTH(dt) as Month, HOUR(dt) as Hour, ' \
              f'floor(MINUTE(dt) / 10)*10 as Minute ' \
              f'from GYSOMATE.GHI_T_P_Rain_RH_Moufia_1min ' \
              f'where GHI_020_Avg>=0 and ' \
              f'(dt>="2018-03-01" and dt<="2019-05-01") ' \
              f'group by date(dt), hour(dt), floor(minute(dt) / 10);'

    print(mjo_sel)
    mjo_query = f'SELECT dt as DateTime, rmm1, rmm2, phase, amplitude ' \
                f'from SWIO.MJO_index ' \
                f'where year(dt)>=1999 and year(dt)<=2016;'
    # NOTE: amplitude is the square root of rmm1^2 + rmm2^2

    df = query_data(mysql_query=mjo_query, remove_missing_data=False)

    if match_ssr_avail_day:
        return df
    else:
        return df


def sellonlatbox(da: xr.DataArray, lonlatbox: list):
    """
    used on dataarray, as that of cdo command
    Parameters
    ----------
    da : input data
    lonlatbox : [lon1, lon2, lat1, lat2], south and west are negative

    Returns
    -------
    DataArray
    """
    # TODO: consider the definition of negative values of lon/lat

    coords = da.coords.variables.mapping

    if 'lon' in coords.keys():
        lon_name = 'lon'
    if 'longitude' in coords.keys():
        lon_name = 'longitude'

    if 'lat' in coords.keys():
        lat_name = 'lat'
    if 'latitude' in coords.keys():
        lat_name = 'latitude'

    da1 = da.where(np.logical_and(da[lon_name] > min(lonlatbox[0], lonlatbox[1]),
                                  da[lon_name] < max(lonlatbox[0], lonlatbox[1])), drop=True)
    da2 = da1.where(np.logical_and(da1[lat_name] > min(lonlatbox[2], lonlatbox[3]),
                                   da1[lat_name] < max(lonlatbox[2], lonlatbox[3])), drop=True)

    return da2


def filter_2d_by_mask(data: xr.DataArray, mask: xr.DataArray):
    """
    filtering 2d data by mask
    :param data:
    :param mask:
    :return: do not change the data format
    """

    # check if the dims of data and mask is the same:
    check_lat = data.lat == mask.lat
    check_lon = data.lon == mask.lon

    if np.logical_or(False in check_lat, False in check_lon):
        print(f'maks and data in different lonlat coords...check ...')
        breakpoint()

    # if isinstance(data, np.ndarray):
    #     data_to_return: np.ndarray = data[mask]
    #     # Attention: if the mask is not square/rectangle of Trues, got 1d array;

    if isinstance(data, xr.DataArray):
        # build up a xr.DataArray as mask, only type of DataArray works.
        lookup = xr.DataArray(mask, dims=('lat', 'lon'))
        # use the standard dim names 'time', 'lat', 'lon'

        data_to_return = data.where(lookup)

    return data_to_return


def value_month_from_str(month: str):
    """
    get month as array from string such as 'JJA'
    :rtype: int
    :param month:
    :return: int
    """

    # TODO: from first letter to number

    if month == 'JJA':
        mon = (6, 7, 8)
    if month == 'DJF':
        mon = (12, 1, 2)
    if month == 'NDJF':
        mon = (11, 12, 1, 2)

    return mon


def filter_df_by_month(data: pd.DataFrame, month: str) -> pd.DataFrame:
    """
    filtering xr.DataArray by input string of season
    :param data:
    :param month: such as 'JJA'
    :return:
    """

    if isinstance(data, pd.DataFrame):
        # TODO: to be updated:
        season_index = ((data.index.month % 12 + 3) // 3).map({1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'})
        return_data = data[season_index == month]

    return return_data


def filter_xr_by_month(data: xr.DataArray, month: str) -> xr.DataArray:
    """
    filtering xr.DataArray by input string of season
    :param data:
    :param month: such as 'JJA', 'DJF', et 'NDJF'
    :return:
    """

    if isinstance(data, xr.DataArray):
        month = value_month_from_str(month)

        mask = [True if x in month else False for x in data.time.dt.month]
        lookup = xr.DataArray(mask, dims=data.dims[0])

        data_to_return = data.where(lookup, drop=True)

    if isinstance(data, xr.Dataset):
        # TODO: to be updated:
        print(f'function to update')

    return data_to_return


def filter_by_season_name(data: xr.DataArray, season_name: str):
    """
    filtering data by input string of season
    :param data:
    :param season_name: summer, winter, austral_summer, austral_winter, etc..
    :return:
    """

    # get tag such as 'JJA'
    season_tag = tag_from_str(season_name)

    if isinstance(data, xr.DataArray):
        return_data = data[data.time.dt.season == season_tag]

    if isinstance(data, pd.DataFrame):
        season_index = ((data.index.month % 12 + 3) // 3).map({1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'})
        return_data = data[season_index == season_tag]

    return return_data


def data_filter_by_key_limit_value(data, key: str, how: str, value: float):
    """
    filtering data by a value of keyword
    :param data:
    :param key: which column
    :param how: lt, gt, eq, lteq
    :param value: float
    :return: data after filtered
    """
    #
    # how_dict = dict(
    #     lt='<',
    #     gt='>',
    #     eq='==',
    #     lteq='<=',
    #     gteq='<=')

    # filter_str = f'{key:s} {how_dict[how]} {value:.2f}'

    if isinstance(data, pd.DataFrame):
        if how == 'gt':
            return_data = data[data[key] > value]

    if isinstance(data, xr.DataArray):
        return_data = data[data.time.dt.season == how]

    return return_data


def reduce_ndim_coord(coord: xr.DataArray, dim_name: str, random: bool = True,
                      max_check_len: int = 500, check_ratio: float = 0.1):
    """
    to check if the input da, usually a coord or dim of a geo map, is static or not,
    if it's static try to reduce the num of dim of this coord

    :param max_check_len:
    :type max_check_len:
    :param dim_name: to check if this dim is change as a function of others
    :type dim_name: such as 'south_north' if the input coord is lon
    :param check_ratio: percentage of the total size, for check if random is True
    :type check_ratio:
    :param coord: input coord such as lon or lat
    :type coord:
    :param random: to use random data, 10 percent of the total data
    :type random:
    :return: ndim-reduced coord as da
    :rtype:
    """

    original_coord = coord
    check_coord = coord

    other_dim = list(original_coord.dims)
    other_dim.remove(dim_name)

    check_coord = check_coord.stack(new_dim=other_dim)

    if random:
        from random import randint
        check_len = int(min(check_coord.shape[0] * check_ratio, max_check_len))
        check_index = [randint(0, check_len - 1) for x in range(check_len)]

        # select some sample to boost
        check_coord = check_coord.isel(new_dim=check_index)

    # starting to check if dim of dim_name is changing as a function of other_dims
    check_coord = check_coord.transpose(..., dim_name)

    diff = 0
    for i in range(check_coord.shape[0]):
        if np.array_equal(check_coord[i], check_coord[0]):
            pass
        else:
            diff += 1
            print('not same: ', i)
            print(f'random number is ', check_index)

    if diff:
        return original_coord
    else:
        # every lon, such as lon[i, :] is the same
        return check_coord[0]


def get_time_lon_lat_from_da(da: xr.DataArray):
    """

    Parameters
    ----------
    da :

    Returns
    -------
    dict :     return {'time': time, 'lon': lon, 'lat': lat, 'number': number}
    """
    coords_names: dict = get_time_lon_lat_name_from_da(da, name_from='coords')
    # attention: here the coords names maybe the dim names, if coords is missing in the *.nc file.

    coords = dict()

    for lonlat in ['lon', 'lat']:
        if lonlat in coords_names:
            lon_or_lat: xr.DataArray = da[coords_names[lonlat]]

            if lon_or_lat.ndim == 1:
                lon_or_lat = lon_or_lat.values

            if lon_or_lat.ndim > 1:
                dim_names = get_time_lon_lat_name_from_da(lon_or_lat, name_from='dims')
                lon_or_lat = reduce_ndim_coord(coord=lon_or_lat, dim_name=dim_names[lonlat], random=True).values

            if lonlat == 'lon':
                if lon_or_lat.ndim == 1:
                    lon_or_lat = np.array([x - 360 if x > 180 else x for x in lon_or_lat])

            coords[lonlat] = lon_or_lat

    if 'time' in coords_names:
        time = da[coords_names['time']].values
        coords.update(time=time)

    if 'lev' in coords_names:
        lev = da[coords_names['lev']].values
        coords.update(lev=lev)

    if 'number' in coords_names:
        number = da[coords_names['number']].values
        coords.update(number=number)

    return coords


def get_time_lon_lat_name_from_da(da: xr.DataArray,
                                  name_from: str = 'coords'):
    """

    Parameters
    ----------
    da ():
    name_from (): get name from coords by default, possible get name from dims.

    Returns
    -------
    dict :     return {'time': time, 'lon': lon, 'lat': lat, 'number': number}
    """
    # definitions:
    possible_coords_names = {
        'time': ['time', 'datetime', 'XTIME', 'Time'],
        'lon': ['lon', 'west_east', 'rlon', 'longitude', 'nx', 'x', 'XLONG', 'XLONG_U', 'XLONG_V'],
        'lat': ['lat', 'south_north', 'rlat', 'latitude', 'ny', 'y', 'XLAT', 'XLAT_U', 'XLAT_V'],
        'lev': ['height', 'bottom_top', 'lev', 'level', 'xlevel', 'lev_2'],
        'number': ['number', 'num', 'model']
        # the default name is 'number'
    }
    # attention: these keys will be used as the standard names of DataArray

    # ----------------------------- important:
    # coords = list(da.dims)
    # ATTENTION: num of dims is sometimes larger than coords: WRF has bottom_top but not such coords
    # ATTENTION: according to the doc of Xarray: len(arr.dims) <= len(arr.coords) in general.
    # ATTENTION: dims names are not the same as coords, WRF dim='south_north', coords=XLAT
    # so save to use the coords names.
    # -----------------------------
    if name_from == 'coords':
        da_names = list(dict(da.coords).keys())
        # CTANG: coords should be a list not a string.
        # since: 't' is in 'time', 'level' is in 'xlevel'
        # and: ['t'] is not in ['time']; 't' is not in ['time']

        dims = list(da.dims)

    if name_from == 'dims':
        da_names = list(da.dims)
        dims = list(da.dims)

    # construct output
    output_names = {}

    for key, possible_list in possible_coords_names.items():
        coord_name = [x for x in possible_list if x in da_names]
        if len(coord_name) == 1:
            output_names.update({key: coord_name[0]})
        else:
            # check if this coords is missing in dims
            dim_name = [x for x in possible_list if x in dims]
            if len(dim_name) == 1:
                output_names.update({key: dim_name[0]})

                print(f'coords {key:s} not found, using dimension name: {dim_name[0]}')
                warnings.warn('coords missing')

    return output_names


def value_cbar_max_min_of_da(da: xr.DataArray):
    max_value = np.float(max(np.abs(da.min()), np.abs(da.max())))

    return max_value, max_value * (-1)


def value_max_min_of_var(var: str, how: str):
    # data:
    # ----------------------------- do not change the following lines:
    var_name = ['sst', 'v10', 'msl', 'q', 'ttr', 'OLR', 'sp', 'SIS', 'ssrd', 'SWDOWN']
    max_mean = [30.00, 7.600, 103100, 0.0200, -190.00, -190.00, 400.0, 170.0, 170.00, 170.00]
    min_mean = [14.00, -6.60, 99900., 0.0000, -310.00, -310.00, 100.0, 0.000, 0.0000, 0.0000]
    max_anom = [0.500, 0.020, 900.00, 0.0020, 50.0000, 50.0000, 5.000, 30.00, 20.000, 20.000]
    min_anom = [-0.50, -0.02, -900.0, -0.002, -50.000, -50.000, -5.00, -30.0, -20.00, -20.00]

    # ----------------------------- do not change above lines.
    if how == 'time_mean':
        vmax = max_mean[var_name.index(var)]
        vmin = min_mean[var_name.index(var)]
    if how == 'anomaly_mean':
        vmax = max_anom[var_name.index(var)]
        vmin = min_anom[var_name.index(var)]

    return vmax, vmin


def get_min_max_ds(ds: xr.Dataset):
    """
    get min and max of ds
    Parameters
    ----------
    ds :
    -------
    """
    list_var = list(ds.keys())
    vmax = np.max([ds[v].max().values for v in list_var])
    vmin = np.min([ds[v].min().values for v in list_var])

    return vmin, vmax


def plot_hourly_boxplot_ds_by(list_da: list, list_var_name: list, by: str = 'Month', comment='no comment'):
    """
    plot hourly box plot by "Month" or "Season"
    :param comment:
    :type comment:
    :param list_da: the input da should be in the same coords
    :param list_var_name:
    :param by: 'Month' or 'season'
    :return:
    """

    import seaborn

    ds = convert_multi_da_to_ds(list_da=list_da, list_var_name=list_var_name)

    # define the variable to use:
    monthly = seasonal = None
    months = list(range(1, 13))
    seasons = ['DJF', 'MAM', 'JJA', 'SON']

    vmin, vmax = get_min_max_ds(ds)
    # vmax = 500
    # vmin = -500

    if by in ['Month', 'month', 'months']:
        monthly = True
        nrow = 6
        ncol = 2
        tags = months
    if by in ['season', 'Season', 'seasons']:
        print(f'plot seasonal plots')
        seasonal = True
        nrow = 2
        ncol = 2
        tags = seasons

    if by is None:
        nrow = 1
        ncol = 1
        tags = None

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            figsize=(ncol * 9, nrow * 3), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, hspace=0.9, top=0.95, wspace=0.2)

    if by is not None:
        axs = axs.ravel()

    # 1) prepare data:

    for i in range(len(tags)):
        if by is None:
            ax = axs
        else:
            # plt.sca(axs[i])  # active this subplot
            ax = axs[i]

        if monthly:
            data_slice: xr.Dataset = ds.where(ds.time.dt.month == tags[i], drop=True)
        if seasonal:
            data_slice: xr.Dataset = ds.where(ds.time.dt.season == tags[i], drop=True)

        if by is None:
            data_slice = ds.copy()

        # to convert da to df: for the boxplot:
        print(f'convert DataArray to DataFrame ...')
        all_var = pd.DataFrame()
        for col in range(len(list_var_name)):
            var = pd.DataFrame()
            da_slice = data_slice[list_var_name[col]]
            var['target'] = da_slice.values.ravel()
            var['Hour'] = da_slice.to_dataframe().index.get_level_values(0).hour
            var['var'] = [list_var_name[col] for _ in range(len(da_slice.values.ravel()))]
            # var['var'] = [list_var_name[col] for x in range(len(da_slice.values.ravel()))]
            all_var = all_var.append(var)

        seaborn.boxplot(x='Hour', y='target', hue='var', data=all_var, ax=ax, showmeans=True)
        # Seaborn's showmeans=True argument adds a mark for mean values in each box.
        # By default, mean values are marked in green color triangles.

        ax.set_xlim(4, 20)
        ax.set_ylim(vmin, vmax)
        if by is not None:
            ax.set_title(f'{by:s} = {str(tags[i]):s}', fontsize=18)

        if comment != 'no comment':
            ax.text(0.02, 0.95, f'{comment:s}', fontsize=14,
                    horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(0.95, 0.95), borderaxespad=0.,
                   loc="upper right", fontsize=18)

        ax.set_xlabel(f'Hour', fontsize=18)
        ax.set_ylabel(f'SSR ($W/m^2$)', fontsize=18)
        ax.tick_params(labelsize=16)

        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')

    print(f'save/show the plot ...')
    plt.savefig(f'./plot/{"-".join(list_var_name):s}.hourly_boxplot_by_{by:s}.png', dpi=200)

    plt.show()

    print(f'got this plot')

    return fig


def plot_hourly_boxplot_by(df: pd.DataFrame, columns: list, by: str):
    """
    plot hourly box plot by "Month" or "Season"
    :param df:
    :param columns:
    :param by:
    :return:
    """

    import seaborn

    if by == 'Month':
        nrow = 7
        ncol = 1
    if by is None:
        nrow = 1
        ncol = 1

    # n_plot = nrow * ncol

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            figsize=(10, 19), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)

    if by == 'Month':
        axs = axs.ravel()

    months = [11, 12, 1, 2, 3, 4, 4]
    for i in range(len(months)):
        if by == 'Month':
            # plt.sca(axs[i])  # active this subplot
            ax = axs[i]
        if by is None:
            ax = axs

        if by == 'Month':
            data_slice = df[df.index.month == months[i]]
        if by is None:
            data_slice = df.copy()

        all_var = pd.DataFrame()
        for col in range(len(columns)):
            # calculate normalised value:
            var = pd.DataFrame()
            var['target'] = data_slice[columns[col]]
            var['Hour'] = data_slice.index.hour
            var['var'] = [columns[col] for _ in range(len(data_slice))]
            all_var = all_var.append(var)

        seaborn.boxplot(x='Hour', y='target', hue='var', data=all_var, ax=ax, showmeans=True)

        ax.set_xlim(5, 20)
        # ax.set_ylim(0, 1.1)
        if by is not None:
            ax.set_title(f'{by:s} = {months[i]:g}')

        # plt.legend()

        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(0.95, 0.95), borderaxespad=0.,
                   loc="upper right", fontsize=18)

        plt.ylabel(f'distribution')
        plt.title(f'SSR distribution between 5AM - 8PM', fontsize=18)
        ax.set_xlabel(f'Hour', fontsize=18)
        ax.set_ylabel(f'SSR ($W/m^2$)', fontsize=18)
        ax.tick_params(labelsize=16)

        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')

    print(f'save/show the plot ...')

    plt.show()

    print(f'got this plot')


def plot_scatter_color_by(x: pd.DataFrame, y: pd.DataFrame, label_x: str, label_y: str,
                          color_by_column: str, size: float = 8):
    """

    :param size:
    :param x:
    :param y:
    :param label_x:
    :param label_y:
    :param color_by_column:
    :return:
    """

    # default is color_by = month

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    colors = ['pink', 'darkviolet', 'blue', 'forestgreen', 'darkorange', 'red',
              'deeppink', 'blueviolet', 'royalblue', 'lightseagreen', 'limegreen', 'yellowgreen', 'tomato',
              'silver', 'gray', 'black']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 6), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, hspace=0.4, top=0.8, wspace=0.05)

    if color_by_column is None:
        xx = x
        yy = y
        plt.scatter(xx, yy, c=colors[1], s=size, edgecolors=colors[1], alpha=0.8)

    if color_by_column == 'Month':
        for i in range(len(months)):
            xx = x[x.index.month == months[i]]
            yy = y[y.index.month == months[i]]

            # plt.plot(xx, yy, label=month_names[i], color=colors[i])
            plt.scatter(xx, yy, c=colors[i], label=month_names[i],
                        s=size, edgecolors=colors[i], alpha=0.8)

        plt.legend(loc="upper right", markerscale=6, fontsize=16)

    ax.set_xlabel(label_x, fontsize=18)
    ax.set_ylabel(label_y, fontsize=18)
    ax.tick_params(labelsize=16)

    plt.grid(True)

    return fig, ax


# ==================================
def get_random_color(num_color: int):
    """
    return color as a list
    :param num_color:
    :return:
    """
    import random

    number_of_colors = num_color

    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
             for _ in range(number_of_colors)]

    return color


# ==================================
def station_data_missing_map_hourly_by_month(df: pd.DataFrame, station_id: str):
    """
    plot hourly missing data map by month
    :param station_id:
    :param df:
    :return:
    """

    # ----------------------------- set parameters -----------------------------
    # TODO: read month directly
    months = [11, 12, 1, 2, 3, 4]
    station_id = list(set(df[station_id]))
    # ----------------------------- set fig -----------------------------
    nrows = len(months)

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 30),
                            facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    axs = axs.ravel()
    # ----------------------------- plotting -----------------------------

    # plot data in each month:
    for i in range(len(months)):
        month = months[i]

        plt.sca(axs[i])  # active this subplot

        month_data = df[df.index.month == month].sort_index()

        # get all time steps to check missing value
        first_timestep = month_data.index[0]
        last_timestep = month_data.index[-1]
        time_range = pd.date_range(first_timestep, last_timestep, freq='60min')
        daytime_range = [x for x in time_range if (8 <= x.hour <= 17)]
        all_daytime_index = pd.Index(daytime_range)

        # for v in range(2):
        for v in range(len(station_id)):

            data_slice = month_data[month_data['station_id'] == station_id[v]]
            nday = len(set(data_slice.index.day))

            print(f'month = {month:g}, station_id = {v:g}, day = {nday:g}')

            # find missing time steps:
            diff = all_daytime_index.difference(data_slice.index)

            if len(diff) == 0:
                print(f'all complete ...')
                plt.hlines(v, 0, 320, colors='blue', linewidth=0.2, linestyles='dashed', label='')
            else:
                print(f'there is missing data ...')

                plt.hlines(v, 0, 320, colors='red', linewidth=0.4, linestyles='dashed', label='')
                for k in range(len(all_daytime_index)):
                    if all_daytime_index[k] in diff:
                        plt.scatter(k, v, edgecolor='black', zorder=2, s=50)

        # ----------------------------- format of fig -----------------------------

        # ----------------------------- x axis -----------------------------
        # put the ticks in the middle of the day, means 12h00
        custom_ticks = range(4, len(data_slice), 10)

        custom_ticks_labels = range(1, nday + 1)
        axs[i].set_xticks(custom_ticks)
        axs[i].set_xticklabels(custom_ticks_labels)
        axs[i].set_xlim(0, 320)

        # axs[i].xaxis.set_ticks_position('top')
        # axs[i].xaxis.set_ticks_position('bottom')

        # ----------------------------- y axis -----------------------------
        custom_ticks = range(len(station_id))

        custom_ticks_labels = station_id

        axs[i].set_yticks(custom_ticks)
        axs[i].set_yticklabels(custom_ticks_labels)

        axs[i].set_ylim(-1, len(station_id) + 1)

        # plt.legend(loc='upper right', fontsize=8)
        plt.xlabel(f'day')
        plt.ylabel(f'station_id (blue (red) means (not) complete in this month)')
        plt.title(data_slice.index[0].month_name())

    suptitle = f'MeteoFrance missing data at each station during daytime (8h - 17h)'
    plt.suptitle(suptitle)

    # plt.show()
    print(f'got the plot')
    plt.savefig('./meteofrance_missing_map.png', dpi=200)


# noinspection PyUnresolvedReferences
def plot_station_value_by_month(lon: pd.DataFrame, lat: pd.DataFrame, value: pd.DataFrame,
                                cbar_label: str, fig_title: str, bias=False):
    """
    plot station locations and their values
    :param bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :param value:
    :return: map show
    """

    print(fig_title)
    import matplotlib as mpl

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    plt.figure(figsize=(5, 24), dpi=200)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):

        # data:
        monthly_data = value[value.index.month == months[m]]

        station_group = monthly_data.groupby('station_id')
        station_mean_bias = station_group[['bias']].mean().values[:, 0]

        # set map
        ax = plt.subplot(len(months), 1, m + 1, projection=ccrs.PlateCarree())
        ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
        # ax.set_extent([20, 110, -51, 9], crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))
        # ax.add_feature(cfeature.OCEAN.with_scale('10m'))
        # ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        # ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        # ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
        # ax.add_feature(cfeature.RIVERS.with_scale('10m'))
        # ax.coastlines()

        # ----------------------------- cbar -----------------------------
        if np.max(station_mean_bias) - np.min(station_mean_bias) < 10:
            round_number = 2
        else:
            round_number = 0

        n_cbar = 10
        vmin = round(np.min(station_mean_bias) / n_cbar, round_number) * n_cbar
        vmax = round(np.max(station_mean_bias) / n_cbar, round_number) * n_cbar

        if bias:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
            print(vmax)
        else:
            cmap = plt.cm.YlOrRd

        vmin = -340
        vmax = 340

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # ----------------------------------------------------------
        # plot:
        # ax.quiver(x, y, u, v, transform=vector_crs)
        sc = plt.scatter(lon, lat, c=station_mean_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.1, label=cbar_label)
        cb.ax.tick_params(labelsize=10)

        # ax.xaxis.set_ticks_position('top')

        ax.gridlines(draw_labels=False)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'monthly daytime (8h 17h) \n mean bias at MeteoFrance stations')

    plt.show()
    print(f'got plot')


def monthly_circulation(lon: xr.DataArray, lat: xr.DataArray,
                        u: xr.DataArray, v: xr.DataArray, p: xr.DataArray, domain: str,
                        cbar_label: str, fig_title: str, bias=False):
    """
    to plot monthly circulation, u, v winds, and mean sea level pressure (p)
    :param domain: one of ['swio', 'reu-mau', 'reu']
    :param p:
    :param v:
    :param u:
    :param bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :return: map show
    """

    print(cbar_label, fig_title, bias)

    months = [11, 12, 1, 2, 3, 4]
    dates = ['2004-11-01', '2004-12-01', '2005-01-01', '2005-02-01', '2005-03-01', '2005-04-01']
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    # nrows = len(months)

    plt.figure(figsize=(5, 24), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):
        ax = plt.subplot(len(months), 1, m + 1, projection=ccrs.PlateCarree())

        # ax.gridlines(draw_labels=False)

        print(f'plot month = {month_names[m]:s}')
        # ----------------------------- plot u and v winds -----------------------------
        # data:
        x = lon.longitude.values
        y = lat.latitude.values
        monthly_u = u.sel(time=dates[m]).values
        monthly_v = v.sel(time=dates[m]).values
        monthly_p = p.sel(time=dates[m]).values

        # set map
        area_name = domain

        if area_name == 'swio':
            n_slice = 1
            n_scale = 2
        if area_name == 'reu_mau':
            n_slice = 1
            n_scale = 10

        if area_name == 'reu':
            n_slice = 2
            n_scale = 10

        area = value_lonlatbox_from_area(area_name)
        ax.set_extent(area, crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))

        # ----------------------------- mean sea level pressure -----------------------------
        # Contour the heights every 10 m
        contours = np.arange(98947, 102427, 300)

        c = ax.contour(x, y, monthly_p, levels=contours, colors='green', linewidths=1)
        ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')

        # ----------------------------- wind -----------------------------
        # Set up parameters for quiver plot. The slices below are used to subset the data (here
        # taking every 4th point in x and y). The quiver_kwargs are parameters to control the
        # appearance of the quiver so that they stay consistent between the calls.
        quiver_slices = slice(None, None, n_slice)
        quiver_kwargs = {'headlength': 5, 'headwidth': 3, 'angles': 'uv', 'scale_units': 'xy', 'scale': n_scale}

        # Plot the wind vectors
        ax.quiver(x[quiver_slices], y[quiver_slices],
                  monthly_u[quiver_slices, quiver_slices], monthly_v[quiver_slices, quiver_slices],
                  color='blue', zorder=2, **quiver_kwargs)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        # ax.gridlines(draw_labels=False)
        # # ----------------------------------------------------------
        # # plot:

        clevs = np.arange(-10, 12, 2)
        # clevs = np.arange(200, 370, 15)
        cf = ax.contourf(x, y, monthly_u, clevs, cmap=plt.cm.coolwarm,
                         norm=plt.Normalize(-10, 10), transform=ccrs.PlateCarree())

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05,
                          # label='ssr')
                          label='east <-- 850hPa zonal wind --> west')

        cb.ax.tick_params(labelsize=10)

        # # ax.xaxis.set_ticks_position('top')

        # ax.text(0.53, 0.95, month_names[m] + 'from ERA5 2004-2005',
        #         horizontalalignment='right', verticalalignment='top',
        #         transform=ax.transAxes)

        plt.title(month_names[m] + ' (ERA5 2004-2005)')

    # plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'circulation wind at 850 hPa')

    plt.show()
    print(f'got plot')

    plt.savefig(f'./monthly_circulation.png', dpi=220)


def convert_multi_da_by_new_dim(list_da: list, new_dim: dict):
    """
    Args:
        list_da ():
        new_dim ():

    Returns: a new da

    """

    # merge data from all models:

    ensemble = xr.concat(list_da, list(new_dim.values())[0]).rename({'concat_dim': list(new_dim.keys())[0]})
    # change order of dims so that cdo could read correctly
    # import itertools
    # ensemble_da = ensemble.transpose(itertools.permutations(list(list_da[0].dims) + list(new_dim.keys())))
    # TODO: change dims order

    return ensemble


def get_gcm_list_in_dir(var: str, path: str):
    files: list = glob.glob(f'{path:s}/{var:s}*nc')

    gcm = list(set([s.split('_')[2] for s in files]))

    gcm.sort()

    return gcm


def convert_cmip6_ensemble_2_standard_da(
        var: str,
        ssp: str,
        freq: str,
        output_nc: str,
        year_start: int,
        year_end: int,
        raw_data_dir: str,
        output_dir: str,
        raw_file_tag: str,
        set_same_cal: bool = False,
        remapping: bool = False):
    """
    to read the raw cmip6 files, before mergetime, and do processes to save it to single netcdf file
    with the same lon lat, same calender.

    rules to follow: attention 1) make input and output separately in different dirs 2) temp file do not have .nc
    in the end.

    Args:
        output_nc ():
        set_same_cal ():
        remapping (bool):
        raw_file_tag ():
        year_end ():
        year_start ():
        var ():
        ssp ():
        raw_data_dir ():
        output_dir ():
        freq ():

    Returns:
        path of output file absolute

    """

    # definition:

    ensemble = 'r1i1p1f1'
    grid = 'gn'
    # ----------------------------- 1st, merge the data to 1970-2099.nc -----------------------------

    # attention this part is done with the code in ./src/
    # all model are merged with the same period, num of year for example

    # ----------------------------- 2nd, prepare the standard data xr.DataArray-----------------------------
    # with the same dims names

    # example of file: rsds_Amon_MPI-ESM1-2-LR_ssp585_r1i1p1f1_gn_2015-2099.year.global_mean.nc
    wildcard_raw_file = f'{raw_data_dir:s}/{var:s}*{freq:s}*{ssp:s}_{ensemble:s}_{grid:s}_' \
                        f'{year_start:g}-{year_end:g}*{raw_file_tag:s}.nc'

    raw_files: list = glob.glob(wildcard_raw_file)

    # gcm list
    gcm_list = list(set([a.split("_")[2] for a in raw_files]))
    gcm_list.sort()

    for ff in range(len(raw_files)):
        # attention: when using matching, careful that CESM2 ~ CESM2-XXX
        gcm_name = [model for model in gcm_list if raw_files[ff].find(f'_{model:s}_') > 0][0]
        # gcm_name = [model for model in gcm_list if model in raw_files[ff].split('_')][0]

        # find

        da = read_to_standard_da(raw_files[ff], var=var)
        da = da.assign_attrs(model_name='gcm_name')
        # here the name is saved to the final ensemble da
        # each name of gcm is in the coords of model_name (new dim)

        if not ff:
            lon = get_time_lon_lat_from_da(da)['lon']
            time = get_time_lon_lat_from_da(da)['time']
            lat = get_time_lon_lat_from_da(da)['lat']

        # create a new DataArray:
        new_da = xr.DataArray(data=da.values.astype(np.float32),
                              dims=('time', 'lat', 'lon'),
                              coords={'time': time, 'lat': lat, 'lon': lon},
                              name=var)
        new_da = new_da.assign_attrs({'model_name': gcm_name,
                                      'units': da.attrs['units'],
                                      'missing_value': np.NAN})

        # new_da = new_da.dropna(dim='latitude')

        if set_same_cal:
            # convert time to the same calendar:
            new_da = convert_da_to_360day_monthly(new_da)

        if remapping:
            # since the lon/lat are different in nc files, remap them to the same/smallest domain:
            # select smaller domain as "GERICS-REMO2015_v1"
            if ff == 0:
                ref_da = new_da.mean(axis=0)
            else:
                print(f'ref:', ref_da.shape, f'remap:', new_da.shape)
                new_da = value_remap_a_to_b(a=new_da, b=ref_da)
                print(new_da.shape)

        # temp data, in the raw_data dir

        temp_data = f'{raw_files[ff]:s}.temp'
        new_da.to_netcdf(temp_data)
        print(f'save to {temp_data:s}')

    # merge data from all models:
    da_list: List[xr.DataArray] = [xr.open_dataarray(file + '.temp') for file in raw_files]

    new_dim = [f'{aa.assign_attrs().model_name:s}' for aa in da_list]

    # noinspection PyTypeChecker
    ensemble_da = xr.concat(da_list, new_dim).rename({'concat_dim': 'model'})
    ensemble_da = ensemble_da.squeeze(drop=True)

    # rename attribute: model_name:
    ensemble_da = ensemble_da.assign_attrs(model_name='ensemble')

    # TODO: cdo sinfo do not works

    # make time the 1st dim
    # change order of dims so that cdo could read correctly
    # dims = list(ensemble_da.dims)
    # dims.remove('time')
    # new_order = ['time'] + dims

    ensemble_da = ensemble_da.transpose('time', 'model')

    # output file name, to save in data dir
    ensemble_da.to_netcdf(f'{output_dir:s}/{output_nc:s}')
    # ----------------------------- ok, clean -----------------------------
    # OK, remove the temp data:
    os.system(f'rm -rf *nc.temp')

    print(f'all done, the data is saved in {output_dir:s} as {output_nc:s}')

    return f'{output_dir:s}/{output_nc:s}'


def convert_cordex_ensemble_2_standard_da(
        var: str,
        domain: str,
        gcm: list,
        rcm: list,
        rcp: str,
        raw_data_dir: str,
        output_dir: str,
        output_tag: str,
        statistic: str,
        test: bool):
    """
    to read the original netcdf files, before mergetime, and do processes to save it to single netcdf file
    with the same lon lat, same calender.
    :param var:
    :type var:
    :param domain:
    :type domain:
    :param gcm:
    :type gcm:
    :param rcm:
    :type rcm:
    :param rcp:
    :type rcp:
    :param raw_data_dir:
    :type raw_data_dir:
    :param output_dir:
    :type output_dir:
    :param output_tag:
    :type output_tag:
    :param statistic:
    :type statistic:
    :param test:
    :type test:
    :return:
    :rtype:
    """

    # ----------------------------- 1st, merge the data to 1970-2099.nc -----------------------------
    # clean:
    # os.system(f'./local_data/{VAR:s}/merge.sfcWind.codex.hist.rcp85.sh -r')
    # merge:
    # os.system(f'./local_data/{VAR:s}/merge.sfcWind.codex.hist.rcp85.sh')
    # already done on CCuR
    # ----------------------------- 2nd, prepare the standard data xr.DataSet -----------------------------

    for window in ['1970-1999', '2036-2065', '2070-2099', '1970-2099']:

        # output file name, to save in data dir
        ensemble_file_output = f'{output_dir:s}/{var:s}/' \
                               f'{var:s}.{statistic:s}.{domain:s}.{rcp:s}.ensemble.{output_tag:s}.{window:s}.nc'

        raw_files: list = glob.glob(f'{raw_data_dir:s}/*{var:s}*{rcp:s}*{window:s}*.{output_tag:s}.{statistic:s}.nc')

        if len(raw_files) == 0:
            continue

        for ff in range(len(raw_files)):
            gcm_name = [model for model in gcm if raw_files[ff].find(model) > 0][0]
            rcm_name = [model for model in rcm if raw_files[ff].find(model) > 0][0]

            da = xr.open_dataset(raw_files[ff])
            da = da[var]
            da = da.assign_attrs(units='hour_per_month')

            if not ff:
                lon = get_time_lon_lat_from_da(da)['lon']
                time = get_time_lon_lat_from_da(da)['time']
                lat = get_time_lon_lat_from_da(da)['lat']

            # create a new DataArray:
            new_da = xr.DataArray(data=da.values.astype(np.float32), dims=('time', 'latitude', 'longitude'),
                                  coords={'time': time, 'latitude': lat, 'longitude': lon},
                                  name=var)
            new_da = new_da.assign_attrs({'model_id': rcm_name, 'driving_model_id': gcm_name,
                                          'units': da.attrs['units'],
                                          'missing_value': np.NAN})

            # new_da = new_da.dropna(dim='latitude')

            if test:
                print(ff, gcm_name, rcm_name)

            set_same_cal = 0
            remapping = 0

            if set_same_cal:
                # convert time to the same calendar:
                new_da = convert_da_to_360day_monthly(new_da)

            if remapping:
                # since the lon/lat are different in nc files, remap them to the same/smallest domain:
                # select smaller domain as "GERICS-REMO2015_v1"
                if ff == 0:
                    ref_da = new_da.mean(axis=0)
                else:
                    print(f'ref:', ref_da.shape, f'remap:', new_da.shape)
                    new_da = value_remap_a_to_b(a=new_da, b=ref_da)
                    print(new_da.shape)
                    print(new_da.shape)

            # temp data, in the raw_data dir
            output_file = f'{raw_data_dir:s}/' \
                          f'{var:s}.{statistic:s}.{domain:s}.{rcp:s}.{gcm_name:s}-{rcm_name:s}.{window:s}.nc.temp'
            new_da.to_netcdf(output_file)

            # ds = xr.merge([ds, new_da.to_dataset()])

        # merge data from all models:
        files_to_merge = f'{raw_data_dir:s}/{var:s}.{statistic:s}.{domain:s}.{rcp:s}.*.{window:s}.nc.temp'
        files = glob.glob(files_to_merge)
        da_list: List[xr.DataArray] = [xr.open_dataarray(file) for file in files]
        new_dim = [f'{aa.assign_attrs().driving_model_id:s}->{aa.assign_attrs().model_id:s}'
                   for aa in da_list]

        # noinspection PyTypeChecker
        ensemble_da = xr.concat(da_list, new_dim).rename({'concat_dim': 'model'})
        # change order of dims so that cdo could read correctly
        ensemble_da = ensemble_da.transpose('time', 'latitude', 'longitude', 'model')

        # drop nan dimension, created by cdo remap:
        ensemble_da = ensemble_da[:, 1:-1, 1:-1, :]
        ensemble_da = ensemble_da.dropna(dim='latitude')
        ensemble_da.to_netcdf(ensemble_file_output)

        # TODO: cdo sinfo do not works

    # ----------------------------- ok, clean -----------------------------
    # OK, remove the temp data:
    os.system(f'rm -rf *nc.temp')

    print(f'all done, the data {window:s} is in ./data/{var:s}/')


def plot_geo_map(data_map: xr.DataArray, bias: int, vmax: np.float = 100, vmin: np.float = 0,
                 suptitle_add_word: str = None):
    """
    plot geo map in the axis = ax
    :param suptitle_add_word:
    :type suptitle_add_word:
    :param vmin:
    :param vmax:
    :param bias:
    :param data_map:
    :return:
    """

    fig = plt.figure(dpi=220)
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

    set_basemap(ax, area='d_1km')
    if bias == 1:
        how = 'anomaly_mean'
    else:
        how = 'time_mean'

    # if max and min is default, then use the value in my table:
    if vmax == 100:
        if vmin == 0:
            vmax, vmin = value_max_min_of_var(var=str(data_map.name), how=how)

    cmap, norm = set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, bias=bias)

    lon = get_time_lon_lat_from_da(data_map)['lon']
    lat = get_time_lon_lat_from_da(data_map)['lat']

    cf: object = ax.contourf(lon, lat, data_map, levels=norm.boundaries,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')

    cbar_label = f'{data_map.name:s} ({data_map.assign_attrs().units:s})'
    # cb_ax = ax.add_axes([0.87, 0.2, 0.01, 0.7])
    # cb = plt.colorbar(cf, orientation='vertical', shrink=0.8, pad=0.05, label=cbar_label)
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, label=cbar_label)

    print(fig, cb)

    title = data_map.assign_attrs().long_name.replace(" ", "_") + f' {how:s}'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    plt.suptitle(title)
    # tag: additional word added to suptitle

    plt.savefig(f'./plot/{data_map.name:s}.{title.replace(" ", "_"):s}.png', dpi=220)
    plt.show()
    print(f'got plot ')


def if_same_coords(map1: xr.DataArray, map2: xr.DataArray, coords_to_check=None):
    """
    return yes or not if 2 maps in the same coordinates
    :param coords_to_check:
    :type coords_to_check:
    :param map2:
    :param map1:
    :return:

    Parameters
    ----------
    coords_to_check : list of the dims to check
    """

    if coords_to_check is None:
        coords_to_check = ['lat', 'lon']

    coords1 = get_time_lon_lat_from_da(map1)
    coords2 = get_time_lon_lat_from_da(map2)

    possible_coords = coords_to_check

    for k in possible_coords:
        if k in coords1.keys() & k in coords2.keys():
            one = coords1[k]
            two = coords2[k]
            if_same: bool = (one == two).all()
        else:
            if_same = False

        same = False and if_same

    return same


def read_to_standard_da(file_path: str, var: str):
    """
    read da and change the dim names/order to time, lon, lat, lev, number
    note: the coords may have several dims, which will be reduced to one, if the coord is not changing
            according to other dims, by the function reduce_ndim_coord
    attention: if the lon and lat could not be reduce to 2D, or if the time could not be reduce to 1D,
               this function will be crash.
    note: order/name of output da are defined in function get_possible_standard_coords_dims

    Parameters
    ----------
    file_path :
    var :

    Returns
    -------

    """

    ds = xr.open_dataset(file_path)

    da = ds[var]

    # change the order of dims: necessary
    # da = convert_da_standard_dims_order(da)

    coords = get_time_lon_lat_from_da(da)

    possible_coords = get_possible_standard_coords_dims()
    # ['time', 'lev', 'lat', 'lon', 'number']

    # prepare new coords with the order and standard names of function: get_possible_standard_coords_dims
    new_coords = dict()
    max_coords_ndim = 1
    for d in possible_coords:
        if d in coords:
            new_coords[d] = coords[d]
            max_coords_ndim = max(max_coords_ndim, coords[d].ndim)

    # to prepare dims names for data and for coords
    possible_dims_coords = get_possible_standard_coords_dims(name_for='dims', ndim=max_coords_ndim) # 1d or 2d
    possible_dims_data_1d = get_possible_standard_coords_dims(name_for='dims', ndim=1)

    # get new dim, for data (time, x, y) and for coords lon: [x,y] separately
    new_dims_data = dict()
    new_dims_coords = dict()
    for d in possible_coords:
        if d in coords:
            new_dims_coords[d] = possible_dims_coords[possible_coords.index(d)]
            new_dims_data[d] = possible_dims_data_1d[possible_coords.index(d)]

    # prepare coords parameters for da:
    coords_param = dict()
    for cod, dim in zip(new_coords.keys(), new_dims_coords.values()):
        coords_param[cod] = (dim, new_coords[cod])

        # example of coords_param:
        # kkk={"time": (new_coords['time']),
        # "lat": (['y', 'x'], new_coords['lat']),
        # "lon": (['y', 'x'], new_coords['lon'])}

    # create new da:
    new_da = xr.DataArray(da.values,
                          dims=list(new_dims_data.values()),
                          coords=coords_param,
                          name=var, attrs=da.attrs)

    # if max_coords_ndim == 2:
    #     new_da = xr.DataArray(da.values,
    #                           dims=list(new_dims.values()),
    #                           coords={
    #                               "time": (new_coords['time']),
    #                               "lat": (['y', 'x'], new_coords['lat']),
    #                               "lon": (['y', 'x'], new_coords['lon']),
    #                           },
    #                           name=var, attrs=da.attrs)

    return new_da


def read_csv_into_df_with_header(csv: str):
    """
    and also set datatimeindex csv has to have a column as 'DateTime'
    Parameters
    ----------
    csv :

    Returns
    -------
    pd.DataFrame

    """

    df = pd.read_csv(csv, na_values=['-9999'])

    df['DateTimeIndex'] = pd.to_datetime(df['DateTime'])

    df = df.set_index('DateTimeIndex')
    del df['DateTime']

    # TODO: test todo
    return df


def match_station_to_grid_data(df: pd.DataFrame, column: str, da: xr.DataArray):
    """
    matching in situ data, type csv with header, to gridded data in DataArray
    merge in space and time

    keyword: station, pixel, gridded, select, match, create
    note: the codes to get station values are lost. not found so far.


    Parameters
    ----------
    df : dataframe with DateTimeIndex
    column : column name of the values to match
    da :  make sure they're in the same timezone

    Returns
    -------
    xr.DataArray
    the output da will be in the same lon-lat grid,
    while the length of time dimension is the same as the in-situ data
    """
    # ----------------------------- read -----------------------------
    # loop in time
    dt_index_mf = df.index.drop_duplicates().dropna().sort_values()
    dt_index_mf = dt_index_mf.tz_localize(None)

    # initialize and make all the values as nan:
    matched_da = da.reindex(time=dt_index_mf)

    # do not touch the unit of the df
    matched_da = matched_da.assign_attrs(units='as original in df')

    matched_da = matched_da.where(matched_da.lon > 360)

    # rename the array with column name
    matched_da = matched_da.rename(column)

    for dt in range(len(dt_index_mf)):
        df_1 = df.loc[dt_index_mf[dt]]
        da_1 = da.sel(time=dt_index_mf[dt])

        for i in range(len(df_1)):
            lat = df_1.latitude[i]
            lon = df_1.longitude[i]
            da_sta = da_1.sel(lat=lat, lon=lon, method='nearest')
            df_sta = df_1.loc[(df_1['longitude'] == lon) & (df_1['latitude'] == lat)]

            nearest_lat = da_sta.lat
            nearest_lon = da_sta.lon

            # update value:
            matched_da.loc[dict(lon=nearest_lon, lat=nearest_lat, time=matched_da.time[dt])] \
                = float(df_sta[column])

            print(dt_index_mf[dt], f'lon={lon:4.2f}, lat={lat:4.2f}, {df_1.station_name[i]:s}')

    print(f'good')

    return matched_da


def plot_nothing(ax):
    # Hide axis
    # plt.setp(ax.get_xaxis().set_visible(False))
    # plt.setp(ax.get_yaxis().set_visible(False))

    # plt.setp(ax.get_xaxis().set_ticks([]))
    # plt.setp(ax.get_yaxis().set_ticks([]))

    # plt.setp(ax.get_yticklabels(),visible=False)
    # plt.setp(ax.get_xticklabels(),visible=False)

    # plt.tick_params(axis="x", which="both", bottom=False, top=False)
    # plt.tick_params(axis="y", which="both", left=False, right=False)
    # ax.tick_params(axis='both', which='both', length=0)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_diurnal_cycle_maps_dataset(list_da: list, bias: int, var_list, hour_list, title: str,
                                    lonlatbox=None, comment='no comment'):
    """
    plot diurnal cycle from dataset, the dataArray may have different coords, vmax and vmin, just a function to show
    all the diurnal cycle
    :param comment:
    :type comment:
    :param bias:
    :type bias:
    :param list_da:
    :type list_da:
    :param title:
    :param lonlatbox: default area: SWIO_1km domain
    :param hour_list:
    :param var_list:
    :return:
    """

    if lonlatbox is None:
        lonlatbox = [54.8, 58.1, -21.9, -19.5]

    # ----------------------------- filtering data by season -----------------------------
    fig, axs = plt.subplots(ncols=len(hour_list), nrows=len(var_list), sharex='row', sharey='col',
                            figsize=(len(hour_list) * 2, len(var_list) * 4), dpi=220,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.02, top=0.8, wspace=0.1, hspace=0.1)

    ds = convert_multi_da_to_ds(list_da, var_list)

    vmin, vmax = get_min_max_ds(ds)
    # vmin = -600
    # vmax = 600

    for var in range(len(var_list)):
        print(f'plot var = {var + 1:g}')

        da = ds[var_list[var]]

        coords = get_time_lon_lat_from_da(da)

        # comment these two lines when using different cbar over subplots
        # vmax = float(da.max(skipna=True))
        # vmin = float(da.min(skipna=True))
        # vmax, vmin = GEO_PLOT.value_max_min_of_var(var=field.name, how=statistic)

        for h in range(len(hour_list)):
            hour = hour_list[h]
            print(f'hour = {hour:g}')

            hourly = da[da.indexes['time'].hour == hour][0]

            # ----------------------------- plotting -----------------------------
            if len(axs.shape) == 1:
                # if the input dataset has only one DataArray
                ax = axs[h]
            else:
                ax = axs[var, h]
            # set map
            ax.set_extent(lonlatbox, crs=ccrs.PlateCarree())
            # lon_left, lon_right, lat_north, lat_north
            ax.coastlines('50m')
            ax.add_feature(cfeature.LAND.with_scale('10m'))

            cmap, norm = set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, bias=bias)
            cf = ax.contourf(coords['lon'], coords['lat'], hourly, levels=norm.boundaries,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')

            ax.text(0.02, 0.95, f'{var_list[var]:s}', fontsize=8,
                    horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            if comment != 'no comment':
                ax.text(0.92, 0.95, f'{comment:s}', fontsize=8,
                        horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

            if var == 0:
                ax.set_title(f'{hour:g}H00')
            # ----------------------------- end of plot -----------------------------
            if h + 1 == len(hour_list):
                cax = inset_axes(ax,
                                 width="5%",  # width = 5% of parent_bbox width
                                 height="100%",  # height : 50%
                                 loc='lower left',
                                 bbox_to_anchor=(1.05, 0., 1, 1),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0,
                                 )

                cbar_label = f'{var_list[var]:s} ({da.assign_attrs().units:s})'
                # ax.text(0.9, 0.9, cbar_label, ha='center', va='center', transform=ax.transAxes)

                cb = plt.colorbar(cf, orientation='vertical', cax=cax, shrink=0.8, pad=0.05, label=cbar_label)
                cb.ax.tick_params(labelsize=10)

    plt.suptitle(title)
    plt.savefig(f'./plot/{"-".join(var_list):s}.hourly_maps.png', dpi=200)
    plt.show()
    print(f'got plot ')


def plot_compare_2geo_maps(map1: xr.DataArray, map2: xr.DataArray, tag1: str = 'A', tag2: str = 'B',
                           suptitle_add_word: str = None):
    """
    to compare 2 geo-maps,
    :param suptitle_add_word:
    :type suptitle_add_word:
    :param tag2:
    :param tag1:
    :param map1: model, to be remapped if necessary
    :param map2: ref,
    :return:

    Parameters
    ----------
    suptitle_add_word :  str
    suptitle_add_word :  add word to the plot sup title

    """

    # to check the if remap is necessary:

    if not if_same_coords(map1, map2, coords_to_check=['lat', 'lon']):
        print(f'coords not same, have to perform remapping to compare...')
        map1 = value_remap_a_to_b(a=map1, b=map2)
        tag1 += f'_remap'

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='row', sharey='col',
                             figsize=(10, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    axes = axes.flatten()

    # ----------------------------- map 1 -----------------------------
    plot_geo_map(data_map=map1, bias=0,
                 # ax=axes[0],
                 vmax=max(map1.max(), map2.max()), vmin=min(map1.min(), map2.min()))
    axes[0].text(0.93, 0.95, tag1, fontsize=12,
                 horizontalalignment='right', verticalalignment='top', transform=axes[0].transAxes)

    axes[0].text(0.93, 0.05, f'mean: {map1.mean().values:4.2f}', fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[0].transAxes)

    # ----------------------------- map 2 -----------------------------
    plot_geo_map(data_map=map2, bias=0,
                 # ax=axes[1],
                 vmax=max(map1.max(), map2.max()), vmin=min(map1.min(), map2.min()))
    axes[1].text(0.93, 0.95, tag2, fontsize=12,
                 horizontalalignment='right', verticalalignment='top', transform=axes[1].transAxes)

    axes[1].text(0.93, 0.05, f'mean: {map2.mean().values:4.2f}', fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[1].transAxes)

    # ----------------------------- plot bias -----------------------------
    bias_map = xr.DataArray(map1.values - map2.values, coords=[map1.lat, map1.lon], dims=map1.dims,
                            name=map1.name, attrs={'units': map1.assign_attrs().units})
    plot_geo_map(data_map=bias_map, bias=1,
                 # ax=axes[2],
                 vmax=max(np.abs(bias_map.max()), np.abs(bias_map.min())),
                 vmin=min(-np.abs(bias_map.max()), -np.abs(bias_map.min())))
    axes[2].text(0.93, 0.95, f'{tag1:s}-{tag2:s}', fontsize=14,
                 horizontalalignment='right', verticalalignment='top', transform=axes[2].transAxes)

    axes[2].text(0.93, 0.05, f'mean: {bias_map.mean().values:4.2f}', fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[2].transAxes)

    # ----------------------------- plot bias in % -----------------------------
    bias_in_percent = xr.DataArray((map1.values - map2.values) / map2.values * 100,
                                   coords=[map1.lat, map1.lon],
                                   dims=map1.dims, name=map1.name, attrs={'units': f'%'})

    vmax = max(np.abs(bias_in_percent.max()), np.abs(bias_in_percent.min()))
    vmin = min(-np.abs(bias_in_percent.max()), -np.abs(bias_in_percent.min()))

    # set the max of %
    if vmax > 1000 or vmin < -1000:
        vmax = 1000
        vmin = -1000

    plot_geo_map(data_map=bias_in_percent, bias=1,
                 # ax=axes[3],
                 vmax=vmax, vmin=vmin)
    axes[3].text(0.93, 0.95, f'({tag1:s}-{tag2:s})/{tag2:s} %', fontsize=14,
                 horizontalalignment='right', verticalalignment='top', transform=axes[3].transAxes)

    axes[3].text(0.93, 0.05, f'mean: {bias_in_percent.mean().values:4.2f} %', fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[3].transAxes)

    date = convert_time_coords_to_datetime(map1).date()
    hour = convert_time_coords_to_datetime(map1).hour

    timestamp = str(date) + 'T' + str(hour)

    title = f'{tag1:s} vs {tag2:s}' + f' ({timestamp:s})'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    plt.suptitle(title)
    # tag: additional word added to suptitle

    plt.savefig(f'./plot/{map1.name:s}.{title.replace(" ", "_"):s}.{timestamp:s}.png', dpi=220)
    plt.show()
    print(f'got plot ')


def convert_df_shifttime(df: pd.DataFrame, second: int):
    """
    shift time by second
    Parameters
    ----------
    df :
    second :
    Returns
    -------
    """

    from datetime import timedelta

    time_shifted = df.index + timedelta(seconds=second)

    new_df = pd.DataFrame(data=df.values, columns=df.columns, index=time_shifted)

    return new_df


def convert_da_standard_dims_order(da: xr.DataArray):
    """
    read da and change the dim order to time, lon, lat, lev, number
    however, the names are not changed

    note: this function may take time when access values of da by da.values

    :param da:
    :type da:
    :return:
    :rtype:
    """

    dims_names = get_time_lon_lat_name_from_da(da, name_from='dims')

    dims_order = []

    possible_coords = get_possible_standard_coords_dims()
    # ['time', 'lev', 'lat', 'lon', 'number']

    possible_name = [x for x in possible_coords if x in dims_names.keys()]
    for i in range(len(dims_names)):
        dims_order.append(dims_names[possible_name[i]])

    new_da = da.transpose(*dims_order)

    return new_da


def convert_da_shifttime(da: xr.DataArray, second: int):
    """
    shift time by second
    Parameters
    ----------
    da :
    second :
    Returns
    -------
    """

    from datetime import timedelta

    coords = get_time_lon_lat_from_da(da)

    time_shifted = da.time.get_index('time') + timedelta(seconds=second)

    new_coords = dict(time=time_shifted)
    possible_coords = ['lev', 'lat', 'lon']  # do not change the order
    for d in possible_coords:
        if d in coords:
            # my_dict['name'] = 'Nick'
            new_coords[d] = coords[d]

    new_da = xr.DataArray(da.values, dims=tuple(new_coords.keys()),
                          coords=new_coords, name=da.name, attrs=da.attrs)

    return new_da


def convert_time_coords_to_datetime(da: xr.DataArray):
    """
    convert time coordinates in dataArray to datetime object
    :param da:
    :return:
    """

    dt_object = pd.Timestamp(da.time.values).to_pydatetime()

    return dt_object


def vis_a_vis_plot(x, y, xlabel: str, ylabel: str, title: str):
    """
    plot scatter plot
    :param title:
    :type title:
    :param xlabel:
    :param ylabel:
    :param x:
    :param y:
    :return:
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    plt.scatter(x, y, marker='^', c='b', s=50, edgecolors='blue', alpha=0.8, label=ylabel)

    plt.title(title)

    # plt.legend(loc="upper right", markerscale=1, fontsize=16)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)

    plt.grid(True)

    plt.show()


# noinspection PyUnresolvedReferences
def plot_altitude_bias_by_month(df: pd.DataFrame, model_column: str, obs_column: str,
                                cbar_label: str,
                                bias=False):
    """
    plot station locations and their values
    :param obs_column:
    :type obs_column:
    :param df:
    :type df:
    :param cbar_label:
    :type cbar_label:
    :param model_column:
    :type model_column:
    :param bias:
    :return: map show
    """
    import matplotlib as mpl

    # data:
    df['bias'] = df[model_column] - df[obs_column]

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    # nrows = len(months)

    plt.figure(figsize=(5, 24), dpi=200)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):
        ax = plt.subplot(len(months), 1, m + 1)

        # data:
        monthly_data = df[df.index.month == months[m]]
        station_group = monthly_data.groupby('station_id')
        station_mean_bias = station_group[['bias']].mean().values[:, 0]
        # station_mean_height = station_group[['altitude']].mean().values[:, 0]

        lon = df['longitude']
        lat = df['latitude']

        # ----------------------------- cbar -----------------------------
        if np.max(station_mean_bias) - np.min(station_mean_bias) < 10:
            round_number = 2
        else:
            round_number = 0

        n_cbar = 10
        vmin = round(np.min(station_mean_bias) / n_cbar, round_number) * n_cbar
        vmax = round(np.max(station_mean_bias) / n_cbar, round_number) * n_cbar

        if bias:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
        else:
            cmap = plt.cm.YlOrRd

        # human chosen values
        # vmin = -340
        # vmax = 340

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        sc = plt.scatter(lon, lat, c=station_mean_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.1, label=cbar_label)
        cb.ax.tick_params(labelsize=10)

        ax.gridlines(draw_labels=False)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'monthly daytime (8h 17h) \n mean bias at MeteoFrance stations')

    plt.show()
    print(f'got plot')


def plot_scatter_contourf(lon: xr.DataArray, lat: xr.DataArray, cloud: xr.DataArray, cbar_label: str,
                          lon_mf: np.ndarray, lat_mf: np.ndarray, value: pd.DataFrame, cbar_mf: str,
                          bias_mf=True):
    """
    to plot meteofrance stationary value and a color filled map.

    :param value:
    :type value:
    :param bias_mf:
    :type bias_mf:
    :param cbar_mf:
    :param lat_mf:
    :param lon_mf:
    :param cloud:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :return: map show
    """

    import matplotlib as mpl
    import datetime as dt

    hours = [x for x in range(8, 18, 1)]
    # dates = ['2004-11-01', '2004-12-01', '2005-01-01', '2005-02-01', '2005-03-01', '2005-04-01']
    # month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    plt.figure(figsize=(10, 20), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each hour:
    for h in range(len(hours)):

        # noinspection PyTypeChecker
        ax = plt.subplot(len(hours) / 2, 2, h + 1, projection=ccrs.PlateCarree())

        print(f'plot hour = {hours[h]:g}')

        # ----------------------------- mean cloud fraction -----------------------------
        # data:

        hourly_cloud = cloud.sel(time=dt.time(hours[h])).mean(axis=0)

        # set map
        reu = value_lonlatbox_from_area('reu')
        ax.set_extent(reu, crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))

        # Plot Color fill of hourly mean cloud fraction
        # normalize color to not have too dark of green at the top end
        clevs = np.arange(60, 102, 2)
        cf = ax.contourf(lon, lat, hourly_cloud, clevs, cmap=plt.cm.Greens,
                         norm=plt.Normalize(60, 102), transform=ccrs.PlateCarree())

        # cb = plt.colorbar(cf, orientation='horizontal', pad=0.1, aspect=50)
        plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label)
        # cb.set_label(cbar_label)

        # ----------------------------- hourly mean bias wrf4.1 - mf -----------------------------

        # data:

        hourly_bias = value[value.index.hour == hours[h]]
        hourly_bias = hourly_bias.groupby('station_id').mean().values.reshape((37,))

        vmax = 240
        vmin = vmax * -1

        if bias_mf:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
        else:
            cmap = plt.cm.YlOrRd

        n_cbar = 20

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        # noinspection PyUnresolvedReferences
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # ----------------------------------------------------------
        sc = plt.scatter(lon_mf, lat_mf, c=hourly_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='vertical', shrink=0.7, pad=0.05, label=cbar_mf)
        cb.ax.tick_params(labelsize=10)

        # ----------------------------- end of plot -----------------------------
        ax.xaxis.set_ticks_position('top')

        ax.gridlines(draw_labels=False)

        ax.text(0.98, 0.95, f'{hours[h]:g}h00\nDJF mean\n2004-2005', horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

        ax.text(0.01, 0.16, f'0.05x0.05 degree\nMVIRI/SEVIRI on METEOSAT',
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    # plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'cloud fraction at daytime hours \n as mean of DJF during 2004 - 2005')

    plt.show()
    print(f'got plot')


def calculate_climate_rolling_da(da: xr.DataArray):
    """
    calculate 30 year rolling mean
    Parameters
    ----------
    da :

    Returns
    -------
    mean
    """
    year = da.groupby(da.time.dt.year).mean(dim='time').rename({'year': 'time'})

    mean: xr.DataArray = year.rolling(time=30, center=True).mean().dropna("time")

    return mean


def get_lon_lat_from_area(area: str):
    """
    get the lon and lat of this position
    Args:
        area ():

    Returns:
        lon, lat

    """

    if area == 'reu':
        lon = 55.5
        lat = -21.1

    return lon, lat


def value_lonlatbox_from_area(area: str):
    """
    get lonlat box of an area by names
    :param area:
    :return: list
    """

    if area == 'southern_Africa':
        box = [0, 59, -40, 1]

    if area == 'AFR-22':
        box = [-24, 59, -47.5, 43.8]

    if area == 'SA_swio':
        box = [0, 90, -50, 10]

    if area == 'reu':
        box = [55, 56, -21.5, -20.8]

    if area == 'bigreu':
        box = [54, 57, -22, -20]

    if area == 'swio':
        box = [20, 110, -50, 9]

    if area == 'reu_mau':
        box = [52, 60, -17.9, -23]

    if area == 'swio-domain':
        box = [32, 76, -34, 4]

    if area == 'd01':
        box = [41, 70, -33, -6]

    if area == 'd02':
        box = [53.1, 60, -22.9, -18.1]

    if area == 'reu-mau':
        box = [52.1, 59.9, -22.9, -18.33]

    if area == 'd_1km':
        box = [54.8, 58.1, -21.9, -19.5]

    if area == 'detect':
        box = [44, 64, -28, -12]

    if area == 'm_r_m':
        box = [40, 64, -30, -10]

    return box


def cluster_mean_gaussian_mixture(var_history, n_components, max_iter, cov_type):
    """
    input days with similar temp profile, return a dataframe with values = most common cluster mean.

    :param var_history: pd.DateFrame
    :param n_components:
    :param max_iter:
    :param cov_type:
    :return: pd.DateFrame of DateTimeIndex
    """

    from sklearn.mixture import GaussianMixture

    # clustering by Gaussian Mixture
    gm = GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type=cov_type)

    var_clusters = gm.fit(var_history)

    cluster_mean = var_clusters.means_

    labels = gm.predict(var_history)

    return cluster_mean, labels


def plot_daily_cluster_mean(mean, locations, labels, ylabel, title):
    fig = plt.figure(figsize=(10, 6), dpi=220)
    # fig.suptitle(fig_title)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect(aspect=0.015)
    # ----------------------------- plotting -----------------------------

    colors = ['blue', 'red', 'orange']
    markers = ['o', '^', 's']
    group_names = ['group 1', 'group 2', 'group 3']

    # get x in hours, even when only have sunny hours:
    x = range(8, 18)

    for c in range(mean.shape[0]):
        plt.plot(x, mean[c, :], color=colors[c], marker=markers[c], label=group_names[c])

    plt.hlines(0, 8, 17, colors='black')

    # plt.text(0.98, 0.95, 'text',
    #          horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title(title)
    # ----------------------------- location of group members -----------------------------

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
    ax.coastlines('50m')

    # ----------------------------- plot cloud fraction from CM SAF -----------------------------

    # cloud fraction cover
    cfc_cmsaf = f'~/local_data/obs/CFC.cmsaf.hour.reu.DJF.nc'
    cloud = xr.open_dataset(cfc_cmsaf).CFC

    mean_cloud = cloud.mean(dim='time')

    clevs = np.arange(60, 82, 2)
    cf = ax.contourf(cloud.lon, cloud.lat, mean_cloud, clevs, cmap=plt.cm.Greens,
                     norm=plt.Normalize(60, 82), transform=ccrs.PlateCarree(), zorder=1)

    # cb = plt.colorbar(cf, orientation='horizontal', pad=0.1, aspect=50)
    plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05, label='daily mean cloud fraction CM SAF')

    # ----------------------------- locations -----------------------------
    # plot location of stations
    for i in range(len(locations)):
        label = labels[i]

        lon = locations['longitude'].values[i]
        lat = locations['latitude'].values[i]

        plt.scatter(lon, lat, color=colors[label],
                    edgecolor='black', zorder=2, s=50, label=group_names[label] if i == 0 else "")

    # sc = plt.scatter(locations['longitude'], locations['latitude'], c=labels,
    #                  edgecolor='black', zorder=2, s=50)

    ax.gridlines(draw_labels=True)

    plt.xlabel(u'$hour$')
    plt.ylabel(ylabel)
    plt.legend(loc='upper right', prop={'size': 8})


def plot_cordex_ensemble_monthly_changes_map(past: xr.DataArray, future: xr.DataArray,
                                             vmax: float, vmin: float,
                                             significance, big_title: str):
    """
    to plot climate changes based on ensemble of model outputs
    1. the input model outputs are in the same shape

    Parameters
    ----------
    big_title :
    past : past, windows defined before in the cfg file
    future :
    Returns
    -------
    :param big_title:
    :type big_title:
    :param significance:
    :type significance:
    :param past:
    :type past:
    :param future:
    :type future:
    :param vmin:
    :type vmin:
    :param vmax:
    :type vmax:

    """

    # windows = {
    #     'past': f'{past.time.dt.year.values[0]:g}-{past.time.dt.year.values[-1]:g}',
    #     'future': f'{future.time.dt.year.values[0]:g}-{future.time.dt.year.values[-1]:g}',
    # }

    fig, axs = plt.subplots(nrows=4, ncols=3, sharex='row', sharey='col',
                            figsize=(15, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    axs = axs.flatten()
    axs = axs.ravel()

    # ----------------------------- subplot data:
    changes = (future - past.assign_coords(time=future.time))

    if significance:
        print(f'calculate significant area ... ')
        ens_mean = []
        for i in range(len(changes)):
            sig_mask: xr.DataArray = value_significant_of_anomaly_2d_mask(field_3d=changes[i], conf_level=0.05)
            masked_map = filter_2d_by_mask(changes[i], mask=sig_mask).mean(axis=2)
            masked_map = masked_map.assign_attrs(units=future.assign_attrs().units)
            ens_mean.append(masked_map)
    else:
        ens_mean = changes.mean(dim=['model'], keep_attrs=True)
        ens_mean = ens_mean.assign_attrs(units=future.assign_attrs().units)

    # vmax = np.max(changes)
    # vmin = np.min(changes)

    print(vmin, vmax)

    for i in range(12):
        print(i)
        plot_geo_subplot_map(geomap=ens_mean[i],
                             vmin=vmin, vmax=vmax,
                             bias=1, ax=axs[i], domain='reu-mau', tag=f'month={i + 1:g}',
                             statistics=True,
                             )

    # ----------------------------- plot 4: ensemble std

    plt.suptitle(big_title)

    plt.savefig(f'./plot/{big_title.replace(" ", "_"):s}.png', dpi=220)
    # add the line blow to the command parameter, to disable the output_dir default by hydra:
    # "hydra.run.dir=. hydra.output_subdir=null"
    # or:
    # add hydra.run.dir = .
    # in the default.yaml.

    plt.show()

    print(f'done')


def plot_multi_scenario_ensemble_time_series(da: xr.DataArray,
                                             plot_every_model: int = 0,
                                             suptitle_add_word: str = '',
                                             highlight_model_list=None,
                                             ):
    """
    plot time series, the input
    Args:
        da (): only have 3 dims: time and number and SSP, with model names in coords
        suptitle_add_word ():
        highlight_model_list (): if plot highlight model, so every_model is off.
        plot_every_model ():

    Returns:
        plot
    """
    if highlight_model_list is None:
        highlight_model_list = []
    if len(highlight_model_list):
        plot_every_model = False

    plt.subplots(figsize=(9, 6), dpi=220)

    scenario = list(da.SSP.data)

    colors = ['blue', 'darkorange', 'green', 'red']

    x = da.time.dt.year

    for s in range(len(scenario)):

        data = da.sel(SSP=scenario[s]).dropna('number')
        num = len(data.number)
        scenario_mean = data.mean('number')

        if plot_every_model:
            for i in range(len(da.number)):
                model_name = list(da.number.data)[i]

                print(f'{model_name:s}, {str(i + 1):s}/{len(da.number):g} model')
                model = str(da.number[i].data)
                data_one_model = da.sel(number=model, SSP=scenario[s])

                plt.plot(x, data_one_model, linestyle='-', linewidth=1.0,
                         alpha=0.2, color=colors[s], zorder=1)

        else:
            # plot range of std
            scenario_std = data.std('number')

            # 95% spread
            low_limit = np.subtract(scenario_mean, 1.96 * scenario_std)
            up_limit = np.add(scenario_mean, 1.96 * scenario_std)

            plt.plot(x, low_limit, '-', color=colors[s], linewidth=0.1, zorder=1)
            plt.plot(x, up_limit, '-', color=colors[s], linewidth=0.1, zorder=1)
            plt.fill_between(x, low_limit, up_limit, color=colors[s], alpha=0.2, zorder=1)

        if len(highlight_model_list):
            j = 0
            for i in range(len(data.number)):
                model_name = list(data.number.data)[i]

                if model_name in highlight_model_list:
                    print(f'highlight this model: {model_name:s}')
                    j += 1

                    data_one_model = da.sel(number=model_name, SSP=scenario[s])

                    plt.plot(x, data_one_model, linestyle=get_linestyle_list()[j][1], linewidth=2.0,
                             alpha=0.8, label=model_name, color=colors[s], zorder=1)

        plt.plot(x, scenario_mean, label=f'{scenario[s]:s} ({num:g} GCMs)', linestyle='-', linewidth=2.0,
                 alpha=1, color=colors[s], zorder=2)

    plt.legend(loc='upper left', prop={'size': 14})

    plt.ylim(-10, 10)

    plt.ylabel(f'{da.name:s} ({da.units:s})')
    plt.xlabel('year')
    # plt.pause(0.05)
    # for interactive plot model, do not works for remote interpreter, do not work in not scientific mode.

    title = f'projected changes, 95% multi model spread'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    plt.suptitle(title)

    plt.savefig(f'{title.replace(" ", "_"):s}.every_model_{plot_every_model:g}.png', dpi=300)

    plt.show()


def get_linestyle_list():
    """
    to use like this linestyle=get_linestyle_list()[i][1]
    Returns:

    """
    linestyles = [
        ('solid', 'solid'),  # Same as (0, ()) or '-'
        ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
        ('dashed', 'dashed'),  # Same as '--'
        ('dashdot', 'dashdot'),  # Same as '-.
        ('loosely dotted', (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),

        ('loosely dashed', (0, (5, 10))),
        ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),

        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    return linestyles


def plot_cordex_ensemble_changes_map(past: xr.DataArray, mid: xr.DataArray, end: xr.DataArray, big_title: str):
    """
    to plot climate changes based on ensemble of model outputs
    1. the input model outputs are in the same shape

    Parameters
    ----------
    big_title :
    past : past, windows defined before in the cfg file
    mid :
    end :
    Returns
    -------

    """

    # windows = {
    #     'past': f'{past.time.dt.year.values[0]:g}-{past.time.dt.year.values[-1]:g}',
    #     'mid': f'{mid.time.dt.year.values[0]:g}-{mid.time.dt.year.values[-1]:g}',
    #     'end': f'{end.time.dt.year.values[0]:g}-{end.time.dt.year.values[-1]:g}',
    # }

    fig, axs = plt.subplots(nrows=4, ncols=3, sharex='row', sharey='col',
                            figsize=(15, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    axs = axs.flatten()
    axs = axs.ravel()

    # ----------------------------- subplot data:
    # 1, 2, 3
    ensemble_yearmean = [a.mean(dim=['time', 'model'], keep_attrs=True) for a in [past, mid, end]]
    # 4
    past_ensemble_std = past.mean(dim='time', keep_attrs=True).std(dim='model', keep_attrs=True)
    # 5
    mid_change = (mid - past.assign_coords(time=mid.time)).mean(dim=['time', 'model'], keep_attrs=True)
    mid_change = mid_change.assign_attrs(units=mid.assign_attrs().units)
    # 6
    end_change = (end - past.assign_coords(time=end.time)).mean(dim=['time', 'model'], keep_attrs=True)
    end_change = end_change.assign_attrs(units=mid.assign_attrs().units)
    # 7
    # emergence = value_time_of_emergence(std=past_ensemble_std, time_series=end_change)
    # TODO: running mean needed, to rewrite this function
    # emergence = past_ensemble_std
    # emergence = emergence.assign_attrs(units='time of emergence')

    # 8,
    past_mean = past.mean(dim=['time', 'model'], keep_attrs=True)
    mid_change_p = mid_change * 100 / (past_mean + 0.001)
    mid_change_p = mid_change_p.assign_attrs(units='%')

    # 9
    end_change_p = end_change * 100 / (past_mean + 0.001)
    end_change_p = end_change_p.assign_attrs(units='%')

    # 10: nothing
    nothing = past_ensemble_std
    # 11
    mid_ensemble_std = mid.mean(dim='time', keep_attrs=True).std(dim='model', keep_attrs=True)
    end_ensemble_std = end.mean(dim='time', keep_attrs=True).std(dim='model', keep_attrs=True)

    # running mean needed.
    emergence = past_ensemble_std

    maps = ensemble_yearmean + [
        past_ensemble_std, mid_change, end_change,
        emergence, mid_change_p, end_change_p,
        nothing, mid_ensemble_std, end_ensemble_std,
    ]
    # -----------------------------
    # for wind changes the max and min :
    # vmin = [np.min(ensemble_yearmean)] * 3 + [0.2, ] + [-0.3, ] * 2 + [2000, -0.15, -0.25, 0, 0.3, 0.3]
    # vmax = [np.max(ensemble_yearmean)] * 3 + [1.2, ] + [+0.3, ] * 2 + [2100, 0.15, 0.25, 1, 1.2, 1.2]
    vmin = [3.8, ] * 3 + [0.2, ] + [-0.3, ] * 2 + [2000, -0.15, -0.25, 0, 0.3, 0.3]
    vmax = [9.5, ] * 3 + [1.2, ] + [+0.3, ] * 2 + [2100, 0.15, 0.25, 1, 1.2, 1.2]
    # for wind changes the max and min :
    # -----------------------------

    # for duration changes, the max and min:
    # vmin = [500, ] * 3 + [5, ] + [-15, ] * 2 + [2000, -25, -25, 0, ] + [5, ] * 2
    # vmax = [800, ] * 3 + [120, ] + [15, ] * 2 + [2100, 25, 25, 10000] + [120, ] * 2

    # -----------------------------

    tags = ['past', 'mid', 'end', 'ensemble_std_past'] + ['mid-change', 'end-change', ] + \
           ['time of emergence', 'mid-change percentage', 'end-change percentage',
            'nothing', 'mid_ensemble_std', 'end_ensemble_std']
    bias = [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0]

    # vmax = np.max(ensemble_yearmean)
    # vmin = np.min(ensemble_yearmean)

    for i in range(12):
        subplot_kwargs = {
            'geomap': maps[i],
            'vmin': vmin[i],
            'vmax': vmax[i],
            'ax': axs[i],
            'domain': 'reu-mau',
            'tag': tags[i],
            'bias': bias[i]}

        print(i)
        plot_geo_subplot_map(**subplot_kwargs)

    # ----------------------------- plot 4: ensemble std

    plt.suptitle(big_title)
    plt.savefig(f'./plot/{big_title.replace(" ", "_"):s}.png', dpi=200)
    plt.show()
    print(f'done')
# change from simu_detect
